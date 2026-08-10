"""Conditional flow matching as a multi-interest retrieval head on KuaiRand.

Stage-2 of the generative-retrieval design (experiments/kuairand_flow_retrieval_*):
stage-1 is the FROZEN item-arm checkpoint from the readout A/B (MaskedHSTU
backbone + 100K x 512 out_table trained with sampled softmax). This script
trains only a small FlowHead v(h_user, z_t, t) with the OT-CFM objective

    x1 = z-scored out_table[target],  x0 ~ N(0, I),  xt = (1-t) x0 + t x1,
    loss = || v_theta(h, xt, t) - (x1 - x0) ||^2

and evaluates retrieval: sample M noise vectors per user, integrate each with
K-step Euler, score all 100K embeddings by L2 in the normalized space, merge
the M candidate lists by min-distance, report recall@K against the SAME 500K
test draw as eval_kuairand_deepk.py (rank arrays directly comparable).

Diagnostics: mean pairwise cosine among the M generated vectors (mode-collapse
meter) and mean distance to the nearest embedding (manifold hit rate).

Run (PYTHONPATH=src):
  python examples/train_kuairand_flow.py --exp_dir experiments/kuairand_flow_retrieval_20260807
"""

import argparse
import json
import os
import time
from datetime import datetime

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from datasets.kuairand import KuaiRandSequences, make_layout, make_mask
from models.readout_hstu import ReadoutHSTUModel
from models.tiger_flow import FlowHead


def main():
    ap = argparse.ArgumentParser(description="Flow-matching retrieval head on frozen KuaiRand stage-1.")
    ap.add_argument("--stage1", type=str,
                    default="experiments/kuairand_readout_ab_20260801/item/best_checkpoint.msgpack")
    ap.add_argument("--stage1_config", type=str,
                    default="experiments/kuairand_readout_ab_20260801/item/config.json")
    ap.add_argument("--cache", type=str, default="./data/kuairand/kuairand27k_top100000.npz")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--targets_per_epoch", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--val_cap", type=int, default=10_000)
    ap.add_argument("--test_cap", type=int, default=500_000)
    ap.add_argument("--grid_cap", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp_dir", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--anchor", action="store_true",
                    help="Arm 2: fit closed-form ridge W: h -> x1 and flow from x0 = Wh + sigma*eps "
                         "(per-dim residual sd) instead of global N(0,I). The generator then only "
                         "learns the local multimodal spread; W carries the global geometry.")
    ap.add_argument("--anchor_fit_n", type=int, default=200_000)
    args = ap.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    metrics_path = os.path.join(args.exp_dir, "metrics.jsonl")

    def log_metrics(**kw):
        kw["ts"] = datetime.now().isoformat(timespec="seconds")
        with open(metrics_path, "a") as f:
            f.write(json.dumps(kw) + "\n")

    cfg1 = json.load(open(args.stage1_config))
    max_len = cfg1["max_len"]
    num_items = cfg1["data_hash_meta"]["vocab_size"]
    dim = cfg1["embedding_dim"]

    data = KuaiRandSequences(args.cache, max_len)
    # Replicate the A/B + deepk draw order exactly: val draw first, then test.
    rng = np.random.RandomState(args.seed)
    # (the A/B used val_cap=50K; deepk consumed the same draw — reproduce it,
    # then subsample for the cheaper per-epoch flow eval)
    val_idx_full = rng.choice(data.val_pool, min(50_000, len(data.val_pool)), replace=False)
    test_idx = rng.choice(data.test_pool, min(args.test_cap, len(data.test_pool)), replace=False)
    val_idx = val_idx_full[: args.val_cap]
    print(f"val={len(val_idx):,} (of {len(val_idx_full):,} draw) | test={len(test_idx):,}")

    # ---- frozen stage-1 -----------------------------------------------------
    anchor, rel_idx = make_layout(max_len, "item")
    backbone = ReadoutHSTUModel(
        num_items=num_items, embedding_dim=dim, num_blocks=cfg1["num_blocks"],
        num_heads=cfg1["num_heads"], attention_dim=cfg1["attention_dim"],
        linear_dim=cfg1["linear_dim"], attn_dropout_rate=0.0, linear_dropout_rate=0.0,
        tie_output=False,
    )
    init_p = backbone.init(jax.random.PRNGKey(0), jnp.zeros((1, max_len), dtype=jnp.int32),
                           jnp.ones((1, max_len, max_len), dtype=bool), rel_idx)["params"]
    with open(args.stage1, "rb") as f:
        frozen = flax.serialization.from_bytes(
            {"params": init_p, "epoch": 0, "best_val_ndcg": 0.0}, f.read())["params"]
    out_table = np.asarray(frozen["out_embedding"])           # [num_items+1, dim]
    emb_mu = out_table[1:].mean(axis=0)
    emb_sd = out_table[1:].std(axis=0) + 1e-6
    table_n = jnp.asarray((out_table[1:] - emb_mu) / emb_sd)  # [num_items, dim], row i = item i+1
    print(f"stage-1 loaded: table {out_table.shape}, per-dim sd mean {emb_sd.mean():.4f}")

    @jax.jit
    def readout(x):
        h, _ = backbone.apply({"params": frozen}, x, make_mask(x, anchor, max_len),
                              rel_idx, deterministic=True)
        return h[:, max_len - 1]                              # [B, dim]

    # ---- optional anchor map (arm 2) ---------------------------------------
    anchor_W = anchor_s = None
    if args.anchor:
        print(f"Fitting anchor map W on {args.anchor_fit_n:,} train targets...")
        fit_idx = rng.choice(data.train_pool, args.anchor_fit_n, replace=False)
        Hs, Xs = [], []
        for i in range(0, len(fit_idx), 1024):
            x, y = data.gather(fit_idx[i:i + 1024])
            Hs.append(np.asarray(readout(jnp.array(x))))
            Xs.append(np.asarray(table_n)[y - 1])
        H = np.concatenate(Hs); X = np.concatenate(Xs)
        H1 = np.concatenate([H, np.ones((len(H), 1), dtype=H.dtype)], axis=1)  # bias column
        A = H1.T @ H1 + 1e-3 * len(H1) * np.eye(H1.shape[1], dtype=H1.dtype)
        W = np.linalg.solve(A, H1.T @ X)
        resid = X - H1 @ W
        anchor_W = jnp.asarray(W)                              # [dim+1, dim]
        anchor_s = jnp.asarray(resid.std(axis=0) + 1e-6)       # [dim]
        print(f"anchor fit: residual per-dim sd mean {float(anchor_s.mean()):.3f} "
              f"(unconditional would be ~1.0)")

    def anchor_point(h):
        ones = jnp.ones((h.shape[0], 1), dtype=h.dtype)
        return jnp.concatenate([h, ones], axis=1) @ anchor_W

    def draw_x0(h, key, shape):
        """Flow source: N(0,I) (arm 1) or anchor + residual-scaled noise (arm 2)."""
        eps = jax.random.normal(key, shape)
        if not args.anchor:
            return eps
        a = anchor_point(h)
        if len(shape) == 3:                                    # [B, M, dim]
            a = a[:, None, :]
        return a + anchor_s * eps

    # ---- trainable flow head ------------------------------------------------
    flow = FlowHead(hidden_dim=args.hidden, output_dim=dim)
    key = jax.random.PRNGKey(args.seed)
    fparams = flow.init(key, jnp.zeros((1, dim)), jnp.zeros((1, dim)), jnp.zeros((1,)))["params"]
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(fparams))
    print(f"FlowHead params: {n_params / 1e6:.2f}M")

    optimizer = optax.chain(optax.clip_by_global_norm(args.grad_clip),
                            optax.adamw(args.learning_rate))
    opt_state = optimizer.init(fparams)

    @jax.jit
    def train_step(fparams, opt_state, h, x1n, key):
        k0, kt = jax.random.split(key)
        x0 = draw_x0(h, k0, x1n.shape)
        t = jax.random.uniform(kt, (x1n.shape[0],))
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1n

        def loss_fn(p):
            v = flow.apply({"params": p}, h, xt, t)
            return jnp.mean(jnp.sum((v - (x1n - x0)) ** 2, axis=-1))

        loss, grads = jax.value_and_grad(loss_fn)(fparams)
        updates, opt_state = optimizer.update(grads, opt_state, fparams)
        return optax.apply_updates(fparams, updates), opt_state, loss

    import functools

    @functools.partial(jax.jit, static_argnums=(3, 4))
    def generate(fparams, h, key, num_samples, num_steps):
        """Euler-integrate num_samples noise seeds per user: [B, M, dim]."""
        B = h.shape[0]
        z = draw_x0(h, key, (B, num_samples, dim))
        hM = jnp.repeat(h[:, None, :], num_samples, axis=1).reshape(B * num_samples, dim)
        z = z.reshape(B * num_samples, dim)
        dt = 1.0 / num_steps
        for s in range(num_steps):
            t = jnp.full((z.shape[0],), s * dt)
            z = z + flow.apply({"params": fparams}, hM, z, t) * dt
        return z.reshape(B, num_samples, dim)

    @jax.jit
    def merged_rank_of_target(z, y):
        """Min-L2-merged rank of the target item across M samples.

        z: [B, M, dim] generated (normalized space); y: [B] item ids (1-based).
        dist^2 to all items, min over M, rank = #items strictly closer.
        """
        d = (jnp.sum(z ** 2, -1)[:, :, None] - 2.0 * jnp.einsum("bmd,nd->bmn", z, table_n)
             + jnp.sum(table_n ** 2, -1)[None, None, :])      # [B, M, N]
        dmin = d.min(axis=1)                                  # [B, N]
        dt_ = dmin[jnp.arange(y.shape[0]), y - 1]
        return (dmin < dt_[:, None]).sum(axis=1) + 1

    @jax.jit
    def sample_diags(z):
        """(mean pairwise cosine among M samples, mean nearest-embedding dist)."""
        zn = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        g = jnp.einsum("bmd,bkd->bmk", zn, zn)
        M = z.shape[1]
        off = (g.sum(axis=(1, 2)) - M) / jnp.maximum(M * (M - 1), 1)
        d = (jnp.sum(z ** 2, -1)[:, :, None] - 2.0 * jnp.einsum("bmd,nd->bmn", z, table_n)
             + jnp.sum(table_n ** 2, -1)[None, None, :])
        return off.mean(), jnp.sqrt(jnp.maximum(d.min(axis=-1), 0.0)).mean()

    KS = [10, 50, 100, 500, 1000]

    def evaluate(fparams, idx, num_samples, num_steps, key, chunk=64, diags=False):
        ranks, cos_l, near_l = [], [], []
        for i in range(0, len(idx), chunk):
            x, y = data.gather(idx[i:i + chunk])
            h = readout(jnp.array(x))
            key, k = jax.random.split(key)
            z = generate(fparams, h, k, num_samples, num_steps)
            ranks.append(np.array(merged_rank_of_target(z, jnp.array(y))))
            if diags and (i // chunk) % 8 == 0:
                c, nd = sample_diags(z)
                cos_l.append(float(c)); near_l.append(float(nd))
        ranks = np.concatenate(ranks)
        out = {f"recall@{k}": float((ranks <= k).mean()) for k in KS}
        out["median_rank"] = float(np.median(ranks))
        if diags:
            out["pairwise_cos"] = round(float(np.mean(cos_l)), 4)
            out["nearest_dist"] = round(float(np.mean(near_l)), 3)
        return out, ranks

    # ---- training loop ------------------------------------------------------
    if args.smoke:
        args.targets_per_epoch, args.epochs = 100 * args.batch_size, 1
        val_idx, test_idx = val_idx[:1024], test_idx[:2048]

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(args.exp_dir, "tensorboard"))

    best_val, best_epoch, patience, gstep = -1.0, 0, 0, 0
    best_path = os.path.join(args.exp_dir, "best_flow_head.msgpack")
    key = jax.random.PRNGKey(args.seed + 7)

    for epoch in range(1, args.epochs + 1):
        pool = rng.choice(data.train_pool, args.targets_per_epoch, replace=False)
        t0, tot, nb = time.time(), 0.0, 0
        for i in range(0, len(pool) - args.batch_size + 1, args.batch_size):
            x, y = data.gather(pool[i:i + args.batch_size])
            h = readout(jnp.array(x))
            x1n = table_n[jnp.array(y) - 1]
            key, k = jax.random.split(key)
            fparams, opt_state, loss = train_step(fparams, opt_state, h, x1n, k)
            loss = float(loss)
            if not np.isfinite(loss):
                log_metrics(fatal="nan_loss", step=gstep)
                raise SystemExit(2)
            tot += loss; nb += 1; gstep += 1
            if gstep % 100 == 0:
                writer.add_scalar("loss/flow", loss, gstep)
        dt = time.time() - t0

        key, k = jax.random.split(key)
        val_res, _ = evaluate(fparams, val_idx, num_samples=4, num_steps=2, key=k, diags=True)
        print(f"Epoch {epoch:02d} | loss {tot/max(nb,1):.4f} | {nb*args.batch_size/dt:.0f} tgt/s | "
              f"val recall@500 {val_res['recall@500']:.4f} recall@50 {val_res['recall@50']:.4f} | "
              f"cos {val_res['pairwise_cos']} near {val_res['nearest_dist']}")
        log_metrics(epoch=epoch, step=gstep, loss=round(tot/max(nb,1), 5), val=val_res,
                    targets_per_sec=round(nb*args.batch_size/dt))
        for mk, mv in val_res.items():
            writer.add_scalar(f"val/{mk}", mv, gstep)

        if val_res["recall@500"] > best_val:
            best_val, best_epoch, patience = val_res["recall@500"], epoch, 0
            if not args.smoke:
                with open(best_path, "wb") as f:
                    f.write(flax.serialization.to_bytes({"params": fparams, "epoch": epoch}))
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stop at epoch {epoch} (best {best_epoch})")
                break

    if not args.smoke and os.path.exists(best_path):
        with open(best_path, "rb") as f:
            fparams = flax.serialization.from_bytes({"params": fparams, "epoch": 0}, f.read())["params"]

    # ---- final test: headline cells + M x steps grid ------------------------
    print("\nTest eval (headline cells on full draw)...")
    final = {"best_epoch": best_epoch, "best_val_recall500": best_val}
    if args.anchor:
        # W-only baseline: rank of the bare anchor point, no flow, no noise.
        ranks = []
        for i in range(0, len(test_idx), 64):
            x, y = data.gather(test_idx[i:i + 64])
            a = anchor_point(readout(jnp.array(x)))
            ranks.append(np.array(merged_rank_of_target(a[:, None, :], jnp.array(y))))
        ranks = np.concatenate(ranks)
        final["test_anchor_only"] = {f"recall@{k}": float((ranks <= k).mean()) for k in KS}
        final["test_anchor_only"]["median_rank"] = float(np.median(ranks))
        print(f"  anchor-only (W, no flow): {final['test_anchor_only']}")
    for M, S in [(1, 2), (16, 2)]:
        key, k = jax.random.split(key)
        res, ranks = evaluate(fparams, test_idx, M, S, k, diags=True)
        final[f"test_M{M}_S{S}"] = res
        np.save(os.path.join(args.exp_dir, f"test_ranks_M{M}_S{S}.npy"), ranks)
        print(f"  M={M} S={S}: {res}")
    grid_idx = test_idx[: args.grid_cap]
    for M in [1, 4, 16]:
        for S in [1, 2, 4]:
            if (M, S) in [(1, 2), (16, 2)]:
                continue
            key, k = jax.random.split(key)
            res, _ = evaluate(fparams, grid_idx, M, S, k)
            final[f"grid_M{M}_S{S}"] = res
            print(f"  grid M={M} S={S}: recall@500 {res['recall@500']:.4f}")
    log_metrics(final=True, **final)


if __name__ == "__main__":
    main()
