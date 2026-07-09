"""Dedicated readout-token A/B for two-tower sequential recommendation.

Tests the hypothesis that reading the user summary out of an item's own
residual stream (dual role: KV semantics + prefix aggregation) hurts a shallow
retrieval model, and that a dedicated ``<begin>`` token with anchor-based
masking removes the interference at negligible cost.

Both arms share one backbone (:class:`models.readout_hstu.ReadoutHSTUModel`),
one data layout, one sampled-softmax loss, and — by construction — the exact
same supervision positions and targets. The only difference is *where* the
readout happens:

  --readout item   (arm A, baseline)
      tokens [x_0..x_{L-1}], causal mask; position j predicts label[j] from
      h_j — the item's own stream is the readout.
  --readout begin  (arm B, treatment)
      tokens [x_0..x_{L-1}, b_0..b_{L-1}]; branch token b_j anchors to context
      position j: it sees context keys k <= j (plus itself), carries logical
      position j+1, and is invisible to everyone else. b_j predicts label[j].
      Item positions carry no loss.

Training is multi-position (one sample per user, loss at every next-item
position of the train prefix), unlike examples/train_hstu.py's one-prefix-
per-sample full-softmax CE — so arm A here is itself a re-baseline; compare
arms against each other, not against the old HSTU rows.

Serving equivalence (§4 of the design doc): each b_j is trained fully isolated,
so eval-time readout at b_{L-1} (anchored to the last real item) reproduces a
single ``<begin>`` appended to the user history. Verified numerically in
scratchpad test_readout_equiv.py.

Run (PYTHONPATH=src):
  python examples/train_readout_hstu.py --dataset beauty --readout item
  python examples/train_readout_hstu.py --dataset beauty --readout begin
"""

import argparse
import os
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from datasets import AmazonDataLoader, SteamDataLoader, MovieLensDataLoader
from evaluation.evaluator import Evaluator
from models.readout_hstu import ReadoutHSTUModel


# ---------------------------------------------------------------------------
# Data: one sample per user, labels at every train-prefix position
# ---------------------------------------------------------------------------
def build_multi_position_data(user_history, max_len):
    """Input x = train items[:-1] (last max_len), labels y = train items[1:].

    Train items are seq[:-2] (leave-one-out protocol: seq[-2] is val,
    seq[-1] is test). Left-padded with 0; label 0 = no loss at that position.
    """
    inputs, labels = [], []
    for _, seq in user_history.items():
        train_items = seq[:-2]
        if len(train_items) < 2:
            continue
        x = train_items[:-1][-max_len:]
        y = train_items[1:][-max_len:]
        pad = max_len - len(x)
        inputs.append([0] * pad + x)
        labels.append([0] * pad + y)
    return np.array(inputs, dtype=np.int32), np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Anchor-based layout: mask, logical positions, relative-distance buckets
# ---------------------------------------------------------------------------
def make_layout(max_len, readout, num_rel_buckets):
    """Returns (anchor [N], pos [N], rel_idx [N, N]) for the chosen arm.

    anchor[q] = last context index visible to query q (§3.3's f);
    pos[q]    = logical position used for relative attention distances.
    """
    ctx = np.arange(max_len)
    if readout == "item":
        anchor, pos = ctx, ctx
    else:  # begin: branch token b_j appended for every context position j
        anchor = np.concatenate([ctx, ctx])
        pos = np.concatenate([ctx, ctx + 1])
    rel = pos[:, None] - pos[None, :]
    rel_idx = np.clip(rel, 0, num_rel_buckets - 1).astype(np.int32)
    return (
        jnp.array(anchor, dtype=jnp.int32),
        jnp.array(pos, dtype=jnp.int32),
        jnp.array(rel_idx),
    )


def make_mask(x, anchor, max_len):
    """Visibility mask [batch, N, N] from the anchor vector.

    M[q, k] = (k < L  and  k <= anchor[q]  and  x[k] != 0)  or  k == q.
    Covers both arms: causal attention over real context tokens; branch
    tokens see only their prefix plus themselves; branches are invisible
    to everyone (they only appear as k via the k == q clause).
    """
    n = anchor.shape[0]
    k_ids = jnp.arange(n)
    is_ctx_key = k_ids < max_len                                   # [N]
    valid_key = jnp.concatenate(
        [x != 0, jnp.zeros((x.shape[0], n - max_len), dtype=bool)], axis=1
    )                                                              # [batch, N]
    reach = k_ids[None, :] <= anchor[:, None]                      # [N, N]
    mask = (is_ctx_key[None, :] & reach)[None, :, :] & valid_key[:, None, :]
    return mask | jnp.eye(n, dtype=bool)[None, :, :]


def main():
    parser = argparse.ArgumentParser(description="Readout-token A/B for two-tower HSTU retrieval.")
    parser.add_argument("--readout", type=str, required=True, choices=["item", "begin"])
    parser.add_argument("--dataset", type=str, default="beauty", choices=["ml-1m", "beauty", "sports", "toys", "steam"])
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--linear_dim", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--tie_output", action="store_true", help="Tie output table to input embeddings.")
    parser.add_argument("--reinject", type=str, default="none", choices=["none", "scalar", "vector"],
                        help="Gated transition-prior re-injection (design doc §1.5): readout becomes "
                             "h_b + alpha * e_in(anchored item), alpha learnable (excluded from weight decay). "
                             "begin arm only.")
    parser.add_argument("--alpha_init", type=float, default=0.0,
                        help="Initial alpha for --reinject; small positive value breaks the alpha-sign "
                             "symmetry toward the empirically better positive basin.")
    parser.add_argument("--num_negatives", type=int, default=1024, help="Sampled-softmax negatives per step (shared across batch).")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--tb_log_dir", type=str, default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume_path", type=str, default="")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    if args.reinject != "none" and args.readout != "begin":
        raise ValueError("--reinject only makes sense for --readout begin (item arm has the identity path built in)")
    rj = "" if args.reinject == "none" else f"_reinj{args.reinject}"
    tag = f"readout_{args.readout}{rj}_{dataset}"
    args.checkpoint_dir = args.checkpoint_dir or f"./data/{tag}_checkpoints"
    args.tb_log_dir = args.tb_log_dir or f"./data/tensorboard/{tag}"

    print(f"--- Readout A/B | arm={args.readout} | dataset={dataset} ---")
    print("Device list:", jax.devices())

    data_dir = "./data"
    if dataset == "ml-1m":
        loader = MovieLensDataLoader(dataset_name="ml-1m", data_dir=data_dir, min_rating=0)
    elif dataset in ["beauty", "sports", "toys"]:
        loader = AmazonDataLoader(category=dataset, data_dir=data_dir, min_rating=0)
    else:
        loader = SteamDataLoader(data_dir=data_dir)
    num_items = loader.num_items
    print(f"Users = {loader.num_users}, Items = {num_items}")

    max_len = args.max_len
    train_inputs, train_labels = build_multi_position_data(loader.user_history, max_len)
    n_positions = int((train_labels > 0).sum())
    print(f"Train: {len(train_inputs)} users, {n_positions} supervised positions")

    val_inputs, val_targets = loader.get_split("val", max_len=max_len, format_type="index").to_numpy()
    test_inputs, test_targets = loader.get_split("test", max_len=max_len, format_type="index").to_numpy()
    print(f"Val: {len(val_targets)} | Test: {len(test_targets)}")

    begin_id = num_items + 1
    anchor, _, rel_idx = make_layout(max_len, args.readout, num_rel_buckets=64)
    n_total = int(anchor.shape[0])

    model = ReadoutHSTUModel(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
        tie_output=args.tie_output,
    )

    def to_tokens(x):
        if args.readout == "item":
            return x
        branches = jnp.full((x.shape[0], max_len), begin_id, dtype=jnp.int32)
        return jnp.concatenate([x, branches], axis=1)

    key = jax.random.PRNGKey(args.seed)
    dummy = jnp.zeros((1, n_total), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, n_total, n_total), dtype=bool)
    params = model.init(key, dummy, dummy_mask, rel_idx)["params"]
    # With re-injection, the trainable tree gains a gate alpha alongside the model
    # params; alpha is masked out of weight decay so its fitted value reflects the
    # data's transition-mixture weight, not the regularizer.
    if args.reinject != "none":
        alpha0 = jnp.full(() if args.reinject == "scalar" else (args.embedding_dim,), args.alpha_init)
        params = {"model": params, "alpha": alpha0}
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {n_params / 1e6:.2f}M")

    if args.reinject != "none":
        wd_mask = {"model": jax.tree_util.tree_map(lambda _: True, params["model"]), "alpha": False}
        optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay, mask=wd_mask)
    else:
        optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    def model_tree(p):
        return p["model"] if args.reinject != "none" else p

    @jax.jit
    def train_step(params, opt_state, x, labels, negs, dropout_key):
        def loss_fn(p):
            mp = model_tree(p)
            tokens = to_tokens(x)
            mask = make_mask(x, anchor, max_len)
            h, out_table = model.apply(
                {"params": mp}, tokens, mask, rel_idx,
                rngs={"dropout": dropout_key}, deterministic=False,
            )
            # Readout states aligned with labels [batch, L, d]
            h_ro = h[:, :max_len] if args.readout == "item" else h[:, max_len:]
            if args.reinject != "none":
                # Fork j's anchored item is x[:, j]; pad rows are loss-masked below.
                h_ro = h_ro + p["alpha"] * mp["item_embedding"]["embedding"][x]
            pos_e = out_table[labels]                              # [B, L, d]
            pos_logit = jnp.sum(h_ro * pos_e, axis=-1)             # [B, L]
            neg_logits = jnp.einsum("bld,md->blm", h_ro, out_table[negs])
            # Kill accidental negatives that equal the row's own positive
            collide = negs[None, None, :] == labels[:, :, None]
            neg_logits = jnp.where(collide, -1e9, neg_logits)
            logits = jnp.concatenate([pos_logit[:, :, None], neg_logits], axis=-1)
            ce = -jax.nn.log_softmax(logits, axis=-1)[:, :, 0]
            w = (labels > 0).astype(jnp.float32)
            return jnp.sum(ce * w) / jnp.maximum(jnp.sum(w), 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def predict_scores(params, x):
        """Full-catalog scores from the serving readout (last real position)."""
        mp = model_tree(params)
        tokens = to_tokens(x)
        mask = make_mask(x, anchor, max_len)
        h, out_table = model.apply({"params": mp}, tokens, mask, rel_idx, deterministic=True)
        # item arm: stream of the most recent item; begin arm: b_{L-1}
        h_q = h[:, max_len - 1] if args.readout == "item" else h[:, -1]
        if args.reinject != "none":
            h_q = h_q + params["alpha"] * mp["item_embedding"]["embedding"][x[:, -1]]
        return h_q @ out_table.T                                   # [B, num_items + 1]

    evaluator = Evaluator(k_list=[1, 5, 10, 20])
    writer = None
    if args.tb_log_dir and not args.eval_only:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)

    best_params, best_val_ndcg, best_epoch = params, -1.0, 0
    if args.resume_path:
        template = {"params": params, "epoch": 0, "best_val_ndcg": 0.0}
        with open(args.resume_path, "rb") as f:
            state = flax.serialization.from_bytes(template, f.read())
        params = best_params = state["params"]
        best_val_ndcg = float(state["best_val_ndcg"])
        print(f"Resumed epoch {state['epoch']} (val NDCG@10 {best_val_ndcg:.5f})")

    if not args.eval_only:
        num_samples = len(train_inputs)
        rng = np.random.RandomState(args.seed)
        drop_rng = jax.random.PRNGKey(args.seed + 1)
        patience_counter, global_step = 0, 0

        for epoch in range(1, args.epochs + 1):
            perm = rng.permutation(num_samples)
            epoch_loss, num_batches = 0.0, 0
            t0 = time.time()
            for i in range(0, num_samples - args.batch_size + 1, args.batch_size):
                idx = perm[i : i + args.batch_size]
                negs = jnp.array(rng.randint(1, num_items + 1, size=args.num_negatives), dtype=jnp.int32)
                drop_rng, step_rng = jax.random.split(drop_rng)
                params, opt_state, loss = train_step(
                    params, opt_state, jnp.array(train_inputs[idx]), jnp.array(train_labels[idx]), negs, step_rng
                )
                epoch_loss += float(loss)
                num_batches += 1
                global_step += 1
            avg_loss = epoch_loss / max(num_batches, 1)
            alpha_str = ""
            if args.reinject != "none":
                a = np.array(params["alpha"])
                alpha_str = f" | alpha {float(a):.4f}" if a.ndim == 0 else f" | |alpha| {float(np.linalg.norm(a)):.4f} mean {float(a.mean()):.4f}"
            print(f"Epoch {epoch:03d}/{args.epochs} | loss {avg_loss:.4f}{alpha_str} | {time.time() - t0:.1f}s")
            if writer is not None:
                writer.add_scalar("Loss/train", avg_loss, global_step)
                if args.reinject != "none":
                    a = np.array(params["alpha"])
                    writer.add_scalar("Alpha/value", float(a) if a.ndim == 0 else float(np.linalg.norm(a)), global_step)

            val_results = evaluator.evaluate_index_based(
                lambda b: predict_scores(params, jnp.array(b)), val_inputs, val_targets, batch_size=512
            )
            print(f"  Val | NDCG@10 {val_results['NDCG@10']:.5f} | HR@10 {val_results['HR@10']:.5f} | MRR {val_results['MRR']:.5f}")
            if writer is not None:
                for m, s in val_results.items():
                    writer.add_scalar(f"Val/{m}", s, global_step)

            if val_results["NDCG@10"] > best_val_ndcg:
                best_val_ndcg, best_params, best_epoch = val_results["NDCG@10"], params, epoch
                patience_counter = 0
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                with open(os.path.join(args.checkpoint_dir, "best_checkpoint.msgpack"), "wb") as f:
                    f.write(flax.serialization.to_bytes(
                        {"params": params, "epoch": epoch, "best_val_ndcg": best_val_ndcg}
                    ))
                print(f"  >>> new best (val NDCG@10 {best_val_ndcg:.5f}), saved")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stop at epoch {epoch} (best epoch {best_epoch})")
                    break

    print("\nRunning test evaluation on best checkpoint...")
    test_results = evaluator.evaluate_index_based(
        lambda b: predict_scores(best_params, jnp.array(b)), test_inputs, test_targets, batch_size=512
    )
    print("\n--- Test results ---")
    for m, s in test_results.items():
        print(f"{m}: {s:.5f}")

    # -----------------------------------------------------------------------
    # §9.2 diagnostics: identity leakage + table alignment
    # -----------------------------------------------------------------------
    print("\n--- Diagnostics (best params) ---")
    best_model_params = model_tree(best_params)
    sample = jnp.array(val_inputs[: min(4096, len(val_inputs))])
    tokens = to_tokens(sample)
    mask = make_mask(sample, anchor, max_len)
    h, out_table = model.apply({"params": best_model_params}, tokens, mask, rel_idx, deterministic=True)
    emb_table = best_model_params["item_embedding"]["embedding"]

    def centered_cos(a, b):
        a = a - a.mean(axis=0, keepdims=True)
        b = b - b.mean(axis=0, keepdims=True)
        num = jnp.sum(a * b, axis=-1)
        den = jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-8
        return float(jnp.mean(num / den))

    last_items = sample[:, -1]
    e_last = emb_table[last_items]
    e_rand = emb_table[np.random.RandomState(0).randint(1, num_items + 1, size=len(sample))]
    h_item_stream = h[:, max_len - 1]     # last item's own stream (KV source)
    leak_self = centered_cos(h_item_stream, e_last)
    leak_rand = centered_cos(h_item_stream, e_rand)
    print(f"identity leakage @ last item stream: cos(h, e_in(x_t)) = {leak_self:.4f} vs random {leak_rand:.4f}")

    h_readout = h[:, max_len - 1] if args.readout == "item" else h[:, -1]
    ro_self = centered_cos(h_readout, e_last)
    ro_rand = centered_cos(h_readout, e_rand)
    print(f"identity leakage @ readout vector:   cos(h_ro, e_in(x_t)) = {ro_self:.4f} vs random {ro_rand:.4f}")

    align = centered_cos(emb_table[1 : num_items + 1], out_table[1:])
    print(f"in/out table alignment: mean diag cos = {align:.4f}")

    alpha_note = ""
    if args.reinject != "none":
        a = np.array(best_params["alpha"])
        alpha_note = (f", alpha={float(a):.4f}" if a.ndim == 0
                      else f", |alpha|={float(np.linalg.norm(a)):.4f} (mean {float(a.mean()):.4f})")
        print(f"final alpha (best ckpt): {alpha_note[2:]}")

    from datetime import datetime
    row = (
        f"| {datetime.now().strftime('%Y-%m-%d')} | Readout A/B arm={args.readout}{rj} "
        f"(HSTU {args.num_blocks}x{args.embedding_dim}, sampled-softmax M={args.num_negatives}, "
        f"{'tied' if args.tie_output else 'untied'}) on {dataset.upper()} | Local (GeForce RTX 4080) | "
        f"{test_results['HR@5']:.5f} | {test_results['NDCG@5']:.5f} | {test_results['HR@10']:.5f} | "
        f"{test_results['NDCG@10']:.5f} | {test_results['HR@20']:.5f} | {test_results['NDCG@20']:.5f} | "
        f"{test_results['MRR']:.5f} | Best val NDCG@10={best_val_ndcg:.5f} (epoch {best_epoch}); "
        f"leak self/rand={leak_self:.3f}/{leak_rand:.3f}, readout leak={ro_self:.3f}, table align={align:.3f}{alpha_note} |"
    )
    with open("experiment_results.md", "a") as f:
        f.write(row + "\n")
    print("\nResults row appended to experiment_results.md")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
