"""Production-regime readout A/B on KuaiRand-27K (single-target GTS).

Replicates the production setup that motivated this experiment line:
  - vocab: top-N videos by last-train-day watch count (examples/prepare_kuairand.py)
  - single-target supervision: each sample is (history window, next watch);
    exactly ONE readout per forward pass — the regime where the dual-duty
    conflict should NOT exist (design doc §14 / production leak curve 0.4->0.08)
  - GTS: train = days 1..28, val = test-day-1 subsample, test = test days 2-3;
    eval context may include earlier test-window events (serving semantics)
  - sampled-softmax retrieval over the vocab (production SSM analog)

Arms share the backbone (models.readout_hstu.ReadoutHSTUModel):
  --readout item   readout = last item's own final-layer stream (production baseline)
  --readout begin  readout = a single <begin> branch token anchored at the last
                   real position (invisible to context, logical position L)

Key diagnostic: per-layer identity-leak curve cos(h_l[readout_pos], e_in(x_last)),
the quantity that was 0.4->0.08 in the production 5x512 model. Pre-registered
predictions live in the run's EXPERIMENT.md.

Run (PYTHONPATH=src):
  python examples/train_kuairand_readout.py --readout item --num_blocks 5 --embedding_dim 512
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


# ---------------------------------------------------------------------------
# Data: ragged cache -> single-target samples
# ---------------------------------------------------------------------------
class KuaiRandSequences:
    """Wraps the prepare_kuairand.py npz cache for single-target sampling."""

    def __init__(self, path, max_len):
        d = np.load(path)
        self.video_ids = d["video_ids"]
        self.time_ms = d["time_ms"]
        self.is_test = d["is_test"]
        self.offsets = d["offsets"]
        self.max_len = max_len
        counts = np.diff(self.offsets)
        # Per-event start offset of its user's sequence (for window clamping).
        self.event_start = np.repeat(self.offsets[:-1], counts)
        pos_in_user = np.arange(len(self.video_ids)) - self.event_start
        has_history = pos_in_user >= 1
        self.train_pool = np.where(~self.is_test & has_history)[0]

        # Split test events into val day (first) and test days (rest) by calendar day.
        test_days = np.unique(self.day(self.time_ms[self.is_test]))
        self.val_day, self.test_days = test_days[0], test_days[1:]
        ev_day = self.day(self.time_ms)
        self.val_pool = np.where(self.is_test & has_history & (ev_day == self.val_day))[0]
        self.test_pool = np.where(self.is_test & has_history & (ev_day > self.val_day))[0]

    @staticmethod
    def day(ms):
        return (ms // 86_400_000).astype(np.int64)

    def gather(self, event_idx):
        """Left-padded histories [B, max_len] + targets [B] for event indices.

        Vectorized: the window for target e is events [e-L, e) clipped at the
        user's sequence start — contiguous, so invalid slots are naturally the
        left-pad positions.
        """
        e = np.asarray(event_idx)
        L = self.max_len
        cols = e[:, None] + np.arange(-L, 0)[None, :]
        valid = cols >= self.event_start[e][:, None]
        x = np.where(valid, self.video_ids[np.clip(cols, 0, None)], 0).astype(np.int32)
        return x, self.video_ids[e].astype(np.int32)


# ---------------------------------------------------------------------------
# Single-target layout: context tokens + (optionally) ONE begin branch token
# ---------------------------------------------------------------------------
def make_layout(max_len, readout, num_rel_buckets):
    ctx = np.arange(max_len)
    if readout == "item":
        anchor, pos = ctx, ctx
    else:  # one branch token anchored at the last context position
        anchor = np.concatenate([ctx, [max_len - 1]])
        pos = np.concatenate([ctx, [max_len]])
    rel = pos[:, None] - pos[None, :]
    rel_idx = np.clip(rel, 0, num_rel_buckets - 1).astype(np.int32)
    return jnp.array(anchor, dtype=jnp.int32), jnp.array(rel_idx)


def make_mask(x, anchor, max_len):
    """M[q,k] = (k is context and k <= anchor[q] and x[k] != 0) or k == q."""
    n = anchor.shape[0]
    k_ids = jnp.arange(n)
    is_ctx_key = k_ids < max_len
    valid_key = jnp.concatenate(
        [x != 0, jnp.zeros((x.shape[0], n - max_len), dtype=bool)], axis=1
    )
    reach = k_ids[None, :] <= anchor[:, None]
    mask = (is_ctx_key[None, :] & reach)[None, :, :] & valid_key[:, None, :]
    return mask | jnp.eye(n, dtype=bool)[None, :, :]


def main():
    parser = argparse.ArgumentParser(description="Production-regime readout A/B on KuaiRand.")
    parser.add_argument("--readout", type=str, required=True, choices=["item", "begin"])
    parser.add_argument("--cache", type=str, default="./data/kuairand/kuairand27k_top100000.npz")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--num_blocks", type=int, default=5)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--attention_dim", type=int, default=256)
    parser.add_argument("--linear_dim", type=int, default=2048, help="4x embedding_dim, matching production FFN ratio.")
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--normalize", action="store_true",
                        help="Cosine retrieval scoring (s = 20*cos(h, e)) instead of raw dot, train and eval.")
    parser.add_argument("--cowatch_weight", type=float, default=0.0,
                        help="Weight of the co-watch item-item InfoNCE auxiliary loss on the out_table "
                             "(aligns e(last watch) -> e(target) vs in-batch negatives). Gives the table "
                             "metric locality — the property the flow-retrieval postmortem showed is absent.")
    parser.add_argument("--cowatch_temp", type=float, default=0.07)
    parser.add_argument("--num_negatives", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Peak LR. 1e-3 (the 4x256 academic default) diverges to NaN by step ~2500 "
                             "at 5x512/seq128: unnormalized silu attention over 129 keys inflates "
                             "activations (init loss 14 vs ln(1025)=6.9 random floor).")
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--targets_per_epoch", type=int, default=2_000_000,
                        help="Train targets subsampled per epoch (pool is ~18.5M; ~30 epochs = ~3 full passes).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--val_cap", type=int, default=50_000)
    parser.add_argument("--test_cap", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory (EXPERIMENT.md must already exist there).")
    parser.add_argument("--smoke", action="store_true", help="Tiny run: 200 steps, small eval, no checkpoints.")
    args = parser.parse_args()

    from models.readout_hstu import ReadoutHSTUModel

    os.makedirs(args.exp_dir, exist_ok=True)
    metrics_path = os.path.join(args.exp_dir, "metrics.jsonl")

    def log_metrics(**kw):
        kw["ts"] = datetime.now().isoformat(timespec="seconds")
        with open(metrics_path, "a") as f:
            f.write(json.dumps(kw) + "\n")

    print(f"--- KuaiRand readout A/B | arm={args.readout} | {args.num_blocks}x{args.embedding_dim} ---")
    print("Devices:", jax.devices())

    data = KuaiRandSequences(args.cache, args.max_len)
    with open(args.cache.replace(".npz", ".meta.json")) as f:
        meta = json.load(f)
    num_items = meta["vocab_size"]
    print(f"vocab={num_items:,} | train pool={len(data.train_pool):,} | "
          f"val pool={len(data.val_pool):,} (day {data.val_day}) | test pool={len(data.test_pool):,}")

    rng = np.random.RandomState(args.seed)
    val_idx = rng.choice(data.val_pool, min(args.val_cap, len(data.val_pool)), replace=False)
    test_idx = rng.choice(data.test_pool, min(args.test_cap, len(data.test_pool)), replace=False)
    val_x, val_y = data.gather(val_idx)
    test_x, test_y = data.gather(test_idx)

    begin_id = num_items + 1
    anchor, rel_idx = make_layout(args.max_len, args.readout, num_rel_buckets=64)
    n_total = int(anchor.shape[0])
    ro_pos = args.max_len - 1 if args.readout == "item" else n_total - 1

    model = ReadoutHSTUModel(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
        tie_output=False,
    )

    def to_tokens(x):
        if args.readout == "item":
            return x
        return jnp.concatenate(
            [x, jnp.full((x.shape[0], 1), begin_id, dtype=jnp.int32)], axis=1)

    key = jax.random.PRNGKey(args.seed)
    params = model.init(
        key, jnp.zeros((1, n_total), dtype=jnp.int32),
        jnp.ones((1, n_total, n_total), dtype=bool), rel_idx,
    )["params"]
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {n_params / 1e6:.2f}M")

    import optax
    sched = optax.join_schedules(
        [optax.linear_schedule(0.0, args.learning_rate, args.warmup_steps),
         optax.constant_schedule(args.learning_rate)],
        [args.warmup_steps],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=sched, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x, labels, negs, dropout_key):
        def loss_fn(p):
            mask = make_mask(x, anchor, args.max_len)
            h, out_table = model.apply(
                {"params": p}, to_tokens(x), mask, rel_idx,
                rngs={"dropout": dropout_key}, deterministic=False,
            )
            h_ro = h[:, ro_pos]                                    # [B, d]
            if args.normalize:
                h_ro = 20.0 * h_ro / (jnp.linalg.norm(h_ro, axis=-1, keepdims=True) + 1e-8)
                tbl = out_table / (jnp.linalg.norm(out_table, axis=-1, keepdims=True) + 1e-8)
            else:
                tbl = out_table
            pos_logit = jnp.sum(h_ro * tbl[labels], axis=-1)        # [B]
            neg_logits = h_ro @ tbl[negs].T                         # [B, M]
            neg_logits = jnp.where(negs[None, :] == labels[:, None], -1e9, neg_logits)
            logits = jnp.concatenate([pos_logit[:, None], neg_logits], axis=-1)
            loss = -jax.nn.log_softmax(logits, axis=-1)[:, 0].mean()
            if args.cowatch_weight > 0.0:
                # Item-item locality: align e(last watch) with e(target) against
                # in-batch negatives, in cosine space. Normalize only the rows
                # used (normalizing the full 100K table per step costs ~25% throughput).
                a = out_table[x[:, -1]]
                b = out_table[labels]
                a = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
                b = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
                sim = (a @ b.T) / args.cowatch_temp                 # [B, B]
                cw = -jax.nn.log_softmax(sim, axis=-1).diagonal().mean()
                loss = loss + args.cowatch_weight * cw
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def predict_scores(params, x):
        mask = make_mask(x, anchor, args.max_len)
        h, out_table = model.apply({"params": params}, to_tokens(x), mask, rel_idx, deterministic=True)
        h_q = h[:, ro_pos]
        if args.normalize:
            h_q = h_q / (jnp.linalg.norm(h_q, axis=-1, keepdims=True) + 1e-8)
            out_table = out_table / (jnp.linalg.norm(out_table, axis=-1, keepdims=True) + 1e-8)
        return h_q @ out_table.T                                    # [B, num_items+1]

    def adjacency_gap(params, xs, ys, cap=2048):
        """Locality probe: mean cos(e_last, e_target) - cos(e_last, e_random) on the out_table."""
        tbl = np.asarray(params["out_embedding"])
        en = tbl / (np.linalg.norm(tbl, axis=1, keepdims=True) + 1e-8)
        last, tgt = xs[:cap, -1], ys[:cap]
        rnd = np.random.RandomState(0).randint(1, tbl.shape[0], size=len(tgt))
        return float((en[last] * en[tgt]).sum(1).mean() - (en[last] * en[rnd]).sum(1).mean())

    def run_eval(params, xs, ys, chunk=256):
        hits, ndcgs, mrrs = {k: 0.0 for k in (1, 5, 10, 20)}, {k: 0.0 for k in (1, 5, 10, 20)}, 0.0
        n = len(xs)
        for i in range(0, n, chunk):
            s = np.array(predict_scores(params, jnp.array(xs[i:i + chunk])))
            s[:, 0] = -np.inf
            y = ys[i:i + chunk]
            rank = (s > s[np.arange(len(y)), y][:, None]).sum(axis=1) + 1
            for k in hits:
                hit = rank <= k
                hits[k] += hit.sum()
                ndcgs[k] += (hit / np.log2(rank + 1)).sum()
            mrrs += (1.0 / rank).sum()
        out = {f"HR@{k}": hits[k] / n for k in hits}
        out.update({f"NDCG@{k}": ndcgs[k] / n for k in ndcgs})
        out["MRR"] = mrrs / n
        return out

    # Filtered capture: block outputs only — a bare capture_intermediates=True
    # also stores every Dense/LayerNorm output ([B,N,2048] tensors) and OOMs.
    blocks_only = lambda mdl, _method: (mdl.name or "").startswith("hstu_block")

    @jax.jit
    def _readout_states(params, x):
        """Per-layer hidden state AT the readout position only: [num_blocks, B, d]."""
        mask = make_mask(x, anchor, args.max_len)
        _, state = model.apply(
            {"params": params}, to_tokens(x), mask, rel_idx, deterministic=True,
            capture_intermediates=blocks_only, mutable=["intermediates"],
        )
        inter = state["intermediates"]
        return jnp.stack([inter[f"hstu_block_{i}"]["__call__"][0][:, ro_pos]
                          for i in range(args.num_blocks)])

    def layer_leak_curve(params, xs, cap=1024, chunk=128):
        """Per-layer cos(h_l[readout_pos], e_in(last item)), centered — the
        production diagnostic (0.4 -> 0.08 across 5 layers on the prod model).

        Chunked: the [B, heads, N, N] attention tensors stay per-chunk and only
        the [B, d] readout states accumulate. A single 1024-row forward OOMs on
        16GB for the begin arm (N=129 pads past the 128 tile boundary).
        """
        xs = xs[:cap]
        parts = [np.asarray(_readout_states(params, jnp.array(xs[i:i + chunk])))
                 for i in range(0, len(xs), chunk)]
        h_all = np.concatenate(parts, axis=1)                     # [L, B, d]
        emb = np.asarray(params["item_embedding"]["embedding"])
        e_last = emb[xs[:, -1]]                                   # [B, d]

        def centered_cos(a, b):
            a, b = a - a.mean(0, keepdims=True), b - b.mean(0, keepdims=True)
            return float(np.mean(np.sum(a * b, -1) /
                                 (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)))

        return [round(centered_cos(h_all[i], e_last), 4) for i in range(args.num_blocks)]

    # -----------------------------------------------------------------------
    if args.smoke:
        args.targets_per_epoch, args.epochs = 200 * args.batch_size, 1
        val_x, val_y = val_x[:2048], val_y[:2048]
        test_x, test_y = test_x[:2048], test_y[:2048]

    drop_rng = jax.random.PRNGKey(args.seed + 1)
    best_val_ndcg, best_epoch, patience_counter, global_step = -1.0, 0, 0, 0
    best_path = os.path.join(args.exp_dir, "best_checkpoint.msgpack")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(args.exp_dir, "tensorboard"))

    for epoch in range(1, args.epochs + 1):
        pool = rng.choice(data.train_pool, args.targets_per_epoch, replace=False)
        t0, epoch_loss, nb = time.time(), 0.0, 0
        for i in range(0, len(pool) - args.batch_size + 1, args.batch_size):
            x, y = data.gather(pool[i:i + args.batch_size])
            negs = jnp.array(rng.randint(1, num_items + 1, size=args.num_negatives), dtype=jnp.int32)
            drop_rng, step_rng = jax.random.split(drop_rng)
            params, opt_state, loss = train_step(
                params, opt_state, jnp.array(x), jnp.array(y), negs, step_rng)
            loss = float(loss)
            if not np.isfinite(loss):
                print(f"FATAL: non-finite loss at global step {global_step} — aborting")
                log_metrics(fatal="nan_loss", step=global_step, epoch=epoch)
                raise SystemExit(2)
            epoch_loss += loss
            nb += 1
            global_step += 1
            if global_step % 100 == 0:
                writer.add_scalar("loss/train", loss, global_step)
            if nb % 1000 == 0:
                print(f"  step {nb} | loss {epoch_loss / nb:.4f} | {nb * args.batch_size / (time.time() - t0):.0f} tgt/s")
        dt = time.time() - t0
        avg_loss = epoch_loss / max(nb, 1)

        val_results = run_eval(params, val_x, val_y)
        leak = layer_leak_curve(params, val_x)
        adj = adjacency_gap(params, val_x, val_y)
        print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | {dt:.0f}s ({nb * args.batch_size / dt:.0f} tgt/s) | "
              f"val NDCG@10 {val_results['NDCG@10']:.5f} HR@10 {val_results['HR@10']:.5f} | "
              f"adjΔ {adj:.4f} | leak {leak}")
        log_metrics(epoch=epoch, step=global_step, loss=round(avg_loss, 5),
                    val={k: round(v, 6) for k, v in val_results.items()},
                    leak_curve=leak, adjacency_gap=round(adj, 5),
                    targets_per_sec=round(nb * args.batch_size / dt))
        writer.add_scalar("val/adjacency_gap", adj, global_step)
        for m, s in val_results.items():
            writer.add_scalar(f"val/{m}", s, global_step)
        for li, lv in enumerate(leak):
            writer.add_scalar(f"leak/layer_{li + 1}", lv, global_step)
        writer.add_scalar("throughput/targets_per_sec", nb * args.batch_size / dt, global_step)

        if val_results["NDCG@10"] > best_val_ndcg:
            best_val_ndcg, best_epoch, patience_counter = val_results["NDCG@10"], epoch, 0
            if not args.smoke:
                with open(best_path, "wb") as f:
                    f.write(flax.serialization.to_bytes(
                        {"params": params, "epoch": epoch, "best_val_ndcg": best_val_ndcg}))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop at epoch {epoch} (best {best_epoch})")
                break

    if not args.smoke and os.path.exists(best_path):
        template = {"params": params, "epoch": 0, "best_val_ndcg": 0.0}
        with open(best_path, "rb") as f:
            params = flax.serialization.from_bytes(template, f.read())["params"]

    print("\nTest evaluation (best checkpoint)...")
    test_results = run_eval(params, test_x, test_y)
    leak = layer_leak_curve(params, test_x)
    for m, s in sorted(test_results.items()):
        print(f"{m}: {s:.5f}")
    print(f"per-layer leak curve @ readout pos: {leak}")
    log_metrics(final=True, best_epoch=best_epoch, best_val_ndcg=round(best_val_ndcg, 6),
                test={k: round(v, 6) for k, v in test_results.items()}, leak_curve=leak)


if __name__ == "__main__":
    main()
