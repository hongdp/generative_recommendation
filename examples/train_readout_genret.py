"""E2 arm: generative retrieval on the shared readout backbone.

Fair generative-vs-dense comparison (design doc §13.12 campaign): identical
MaskedHSTU encoder, identical K-fork supervision positions/targets, identical
item-level context tokens as the dense begin arm (train_readout_hstu.py).
The ONLY delta is the output parameterization:

  E1 (dense):      1-token branch, readout . out_table^T, sampled softmax
  E2 (this file):  4-token branch [<begin>, c1, c2, c3] teacher-forcing the
                   target's 4-level Semantic ID; CE over 256 codes per level;
                   beam search over the code trie at eval.

Branch construction generalizes the anchor mask to branch length 4 (§2.3 tree
attention): branch token (fork j, step s) sees context <= j plus same-fork
steps <= s; branches remain mutually invisible; context never sees branches.
Serving equivalence: eval uses a single 4-token branch anchored at the last
real position — identical computation to any training fork (§4).

Run (PYTHONPATH=src):
  python examples/train_readout_genret.py --dataset beauty
"""

import argparse
import json
import os
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from datasets import AmazonDataLoader, SteamDataLoader
from evaluation.metrics import calculate_metrics_from_ranks
from models.readout_hstu import GenReadoutHSTUModel
from train_readout_hstu import build_multi_position_data

DEFAULT_SIDS = {
    "beauty": "./data/semantic_ids_xxl_mlprqvae_dedup_beauty.json",
    "steam": "./data/semantic_ids_steam_mlprqvae_dedup.json",
}


def gen_layout(max_len, branch_len, num_rel_buckets=64, eval_mode=False):
    """Anchor/fork/step/pos vectors. Training: one branch per context position
    (fork-major, branch j at rows L + j*Lb ...). Eval: single branch anchored
    at the last position."""
    ctx = np.arange(max_len)
    if eval_mode:
        forks = np.array([max_len - 1])
    else:
        forks = ctx
    anchor = np.concatenate([ctx, np.repeat(forks, branch_len)])
    fork = np.concatenate([np.full(max_len, -1), np.repeat(np.arange(len(forks)), branch_len)])
    step = np.concatenate([np.zeros(max_len, np.int64), np.tile(np.arange(branch_len), len(forks))])
    pos = np.concatenate([ctx, (np.repeat(forks, branch_len) + 1 + np.tile(np.arange(branch_len), len(forks)))])
    rel_idx = np.clip(pos[:, None] - pos[None, :], 0, num_rel_buckets - 1).astype(np.int32)
    return (jnp.array(anchor, jnp.int32), jnp.array(fork, jnp.int32),
            jnp.array(step, jnp.int32), jnp.array(rel_idx))


def gen_mask(x, anchor, fork, step, max_len):
    """M[q,k] = (context key <= anchor(q), non-pad) OR (same fork, step_k <= step_q) OR k == q."""
    n = anchor.shape[0]
    k_ids = jnp.arange(n)
    valid_key = jnp.concatenate(
        [x != 0, jnp.zeros((x.shape[0], n - max_len), dtype=bool)], axis=1)
    ctx_reach = ((k_ids[None, :] < max_len) & (k_ids[None, :] <= anchor[:, None]))
    same_fork = (fork[:, None] >= 0) & (fork[:, None] == fork[None, :]) & (step[None, :] <= step[:, None])
    mask = (ctx_reach[None, :, :] & valid_key[:, None, :]) | same_fork[None, :, :]
    return mask | jnp.eye(n, dtype=bool)[None, :, :]


def main():
    p = argparse.ArgumentParser(description="Generative-retrieval arm on the shared readout backbone (E2).")
    p.add_argument("--dataset", type=str, default="beauty", choices=["beauty", "sports", "toys", "steam"])
    p.add_argument("--semantic_ids_path", type=str, default="")
    p.add_argument("--num_levels", type=int, default=4)
    p.add_argument("--num_codes", type=int, default=256)
    p.add_argument("--beam", type=int, default=30)
    p.add_argument("--max_len", type=int, default=20)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--attention_dim", type=int, default=128)
    p.add_argument("--linear_dim", type=int, default=512)
    p.add_argument("--dropout_rate", type=float, default=0.2)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_batch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_dir", type=str, default="")
    p.add_argument("--tb_log_dir", type=str, default="")
    args = p.parse_args()

    dataset = args.dataset.lower()
    args.semantic_ids_path = args.semantic_ids_path or DEFAULT_SIDS[dataset]
    tag = f"readout_genret_{dataset}"
    args.checkpoint_dir = args.checkpoint_dir or f"./data/{tag}_checkpoints"

    print(f"--- GenRet E2 | dataset={dataset} | SIDs={args.semantic_ids_path} ---")
    print("Device list:", jax.devices())

    loader = (AmazonDataLoader(category=dataset, data_dir="./data", min_rating=0)
              if dataset in ["beauty", "sports", "toys"] else SteamDataLoader(data_dir="./data"))
    num_items, L, Lb, K = loader.num_items, args.max_len, args.num_levels, args.num_codes
    begin_id, code_base = num_items + 1, num_items + 2

    sem = {int(k): v for k, v in json.load(open(args.semantic_ids_path)).items()}
    code_arr = np.zeros((num_items + 1, Lb), dtype=np.int32)
    for item, codes in sem.items():
        if 0 < item <= num_items:
            code_arr[item] = codes
    path_to_item = {tuple(v): k for k, v in sem.items() if k != 0}
    assert len(path_to_item) == num_items, "SIDs must be collision-free (dedup level)"
    code_arr_j = jnp.array(code_arr)

    train_inputs, train_labels = build_multi_position_data(loader.user_history, L)
    val_inputs, val_targets = loader.get_split("val", max_len=L, format_type="index").to_numpy()
    test_inputs, test_targets = loader.get_split("test", max_len=L, format_type="index").to_numpy()
    print(f"Train users {len(train_inputs)} | positions {(train_labels > 0).sum()} | val {len(val_targets)} | test {len(test_targets)}")

    anchor_t, fork_t, step_t, rel_t = gen_layout(L, Lb)
    anchor_e, fork_e, step_e, rel_e = gen_layout(L, Lb, eval_mode=True)
    n_train, n_eval = int(anchor_t.shape[0]), int(anchor_e.shape[0])

    model = GenReadoutHSTUModel(
        num_items=num_items, num_levels=Lb, num_codes=K,
        embedding_dim=args.embedding_dim, num_blocks=args.num_blocks,
        num_heads=args.num_heads, attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        attn_dropout_rate=args.dropout_rate, linear_dropout_rate=args.dropout_rate)
    params = model.init(jax.random.PRNGKey(args.seed),
                        jnp.zeros((1, n_train), jnp.int32),
                        jnp.ones((1, n_train, n_train), bool), rel_t)["params"]
    print(f"Parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6:.2f}M")

    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    level_off = jnp.arange(Lb) * K + code_base            # token id of code c at level l = level_off[l] + c

    @jax.jit
    def train_step(params, opt_state, x, labels, dropout_key):
        codes = code_arr_j[labels]                        # [B, L, Lb]
        # branch tokens: [<begin>, tok(c1), tok(c2), ..., tok(c_{Lb-1})] teacher-forced
        branch = jnp.concatenate(
            [jnp.full((x.shape[0], x.shape[1], 1), begin_id, jnp.int32),
             codes[:, :, : Lb - 1] + level_off[None, None, : Lb - 1]], axis=2)
        tokens = jnp.concatenate([x, branch.reshape(x.shape[0], -1)], axis=1)
        mask = gen_mask(x, anchor_t, fork_t, step_t, L)

        def loss_fn(pp):
            h, heads = model.apply({"params": pp}, tokens, mask, rel_t,
                                   rngs={"dropout": dropout_key}, deterministic=False)
            hb = h[:, L:].reshape(x.shape[0], L, Lb, -1)  # [B, L, Lb, d]
            logits = jnp.einsum("blsd,sdk->blsk", hb, heads)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, codes)  # [B, L, Lb]
            w = (labels > 0).astype(jnp.float32)[:, :, None]
            return jnp.sum(ce * w) / jnp.maximum(jnp.sum(w) * Lb, 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def step_logits(params, x, branch_tokens, s):
        """Forward the single-fork eval layout; return level-s logits at branch step s."""
        tokens = jnp.concatenate([x, branch_tokens], axis=1)
        mask = gen_mask(x, anchor_e, fork_e, step_e, L)
        h, heads = model.apply({"params": params}, tokens, mask, rel_e, deterministic=True)
        return jax.nn.log_softmax(h[:, L + s] @ heads[s], axis=-1)

    B = args.beam

    def beam_decode(params, x):
        """Returns paths [b, B, Lb] and beam order (already score-sorted)."""
        b = x.shape[0]
        pad = jnp.full((b, Lb), begin_id, jnp.int32)
        lp = step_logits(params, x, pad, 0)                                  # [b, K]
        sc, c0 = jax.lax.top_k(lp, B)                                        # [b, B]
        beams = c0[:, :, None]                                              # [b, B, 1]
        x_rep = jnp.repeat(x, B, axis=0)
        for s in range(1, Lb):
            bt = jnp.full((b * B, Lb), begin_id, jnp.int32)
            bt = bt.at[:, 1 : s + 1].set(beams.reshape(b * B, s) + np.array(level_off)[None, :s])
            lp = step_logits(params, x_rep, bt, s).reshape(b, B, K)
            cum = (sc[:, :, None] + lp).reshape(b, B * K)
            sc, flat = jax.lax.top_k(cum, B)
            prev, c = flat // K, flat % K
            beams = jnp.concatenate(
                [jnp.take_along_axis(beams, prev[:, :, None], axis=1), c[:, :, None]], axis=2)
        return np.array(beams)

    def run_eval(params, inputs, targets):
        ranks, valid_frac = [], []
        for i in range(0, len(inputs), args.eval_batch):
            xb = jnp.array(inputs[i : i + args.eval_batch])
            paths = beam_decode(params, xb)                                  # [b, B, Lb]
            tgt = targets[i : i + args.eval_batch]
            for row in range(len(tgt)):
                tgt_path = tuple(code_arr[tgt[row]])
                r = 10**6
                nv = 0
                for j in range(B):
                    pth = tuple(paths[row, j])
                    if pth in path_to_item:
                        nv += 1
                    if pth == tgt_path and r == 10**6:
                        r = j + 1
                ranks.append(r)
                valid_frac.append(nv / B)
        m = calculate_metrics_from_ranks(np.array(ranks), [1, 5, 10, 20])
        m["Valid@Beam"] = float(np.mean(valid_frac))
        return m

    writer = None
    if args.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)

    num_samples = len(train_inputs)
    rng = np.random.RandomState(args.seed)
    drop_rng = jax.random.PRNGKey(args.seed + 1)
    best_params, best_val, best_epoch, patience_counter, gstep = params, -1.0, 0, 0, 0

    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(num_samples)
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, num_samples - args.batch_size + 1, args.batch_size):
            idx = perm[i : i + args.batch_size]
            drop_rng, sk = jax.random.split(drop_rng)
            params, opt_state, loss = train_step(
                params, opt_state, jnp.array(train_inputs[idx]), jnp.array(train_labels[idx]), sk)
            tot += float(loss); nb += 1; gstep += 1
        print(f"Epoch {epoch:03d}/{args.epochs} | loss {tot / max(nb, 1):.4f} | {time.time() - t0:.1f}s")
        if writer is not None:
            writer.add_scalar("Loss/train", tot / max(nb, 1), gstep)

        vm = run_eval(params, val_inputs, val_targets)
        print(f"  Val | NDCG@10 {vm['NDCG@10']:.5f} | HR@10 {vm['HR@10']:.5f} | Valid@Beam {vm['Valid@Beam']:.4f}")
        if writer is not None:
            for k, v in vm.items():
                writer.add_scalar(f"Val/{k.replace('@', '_')}", v, gstep)
        if vm["NDCG@10"] > best_val:
            best_val, best_params, best_epoch, patience_counter = vm["NDCG@10"], params, epoch, 0
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            with open(os.path.join(args.checkpoint_dir, "best_checkpoint.msgpack"), "wb") as f:
                f.write(flax.serialization.to_bytes(
                    {"params": params, "epoch": epoch, "best_val_ndcg": best_val}))
            print(f"  >>> new best (val NDCG@10 {best_val:.5f}), saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop at epoch {epoch} (best {best_epoch})")
                break

    print("\nTest evaluation on best checkpoint...")
    tm = run_eval(best_params, test_inputs, test_targets)
    for k, v in tm.items():
        print(f"{k}: {v:.5f}")

    from datetime import datetime
    row = (
        f"| {datetime.now().strftime('%Y-%m-%d')} | GenRet E2 (shared backbone, 4-level SID branch, "
        f"beam {B}, HSTU {args.num_blocks}x{args.embedding_dim}) on {dataset.upper()} | Local (GeForce RTX 4080) | "
        f"{tm['HR@5']:.5f} | {tm['NDCG@5']:.5f} | {tm['HR@10']:.5f} | {tm['NDCG@10']:.5f} | "
        f"{tm['HR@20']:.5f} | {tm['NDCG@20']:.5f} | {tm['MRR']:.5f} | "
        f"Best val NDCG@10={best_val:.5f} (epoch {best_epoch}); Valid@Beam={tm['Valid@Beam']:.4f}; "
        f"seed={args.seed}; SIDs={os.path.basename(args.semantic_ids_path)} |"
    )
    with open("experiment_results.md", "a") as f:
        f.write(row + "\n")
    print("Row appended to experiment_results.md")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
