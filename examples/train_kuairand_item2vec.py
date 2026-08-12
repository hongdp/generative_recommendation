"""Standalone item2vec space for KuaiRand — locality as the PRIMARY objective.

The locality-stage1 experiment showed a co-watch auxiliary loss cannot retrofit
metric locality onto a dot-trained two-tower table (v2-v4 all plateaued at
adjacency gap 0.009/0.026/0.053 — equilibrium against the main loss). This
trains the locality space with NO competing objective: skip-gram InfoNCE over
co-watch windows, embeddings only (no transformer).

Objective: for each watch event pair (i, j) within a window of W events in the
same user's stream, align e_i with e_j against S uniform negatives (cosine,
temperature tau). Symmetric by construction (both orders sampled).

Gates for the flow rerun (run examples-side probes after training):
  gate 1: adjacency gap  cos(last,target) - cos(last,random)  >= 0.2
  gate 2: oracle top1-as-query cos-rank of target             <  5000

Run (PYTHONPATH=src):
  python examples/train_kuairand_item2vec.py --exp_dir experiments/<...>/item2vec
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

from datasets.kuairand import KuaiRandSequences


def main():
    ap = argparse.ArgumentParser(description="item2vec locality space for KuaiRand.")
    ap.add_argument("--cache", type=str, default="./data/kuairand/kuairand27k_top100000.npz")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--window", type=int, default=5, help="Co-watch window (events apart <= W).")
    ap.add_argument("--num_negatives", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--learning_rate", type=float, default=3e-3)
    ap.add_argument("--batch_size", type=int, default=4096, help="Pairs per step (embeddings only — go big).")
    ap.add_argument("--pairs_per_epoch", type=int, default=20_000_000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp_dir", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    metrics_path = os.path.join(args.exp_dir, "metrics.jsonl")

    def log_metrics(**kw):
        kw["ts"] = datetime.now().isoformat(timespec="seconds")
        with open(metrics_path, "a") as f:
            f.write(json.dumps(kw) + "\n")

    data = KuaiRandSequences(args.cache, max_len=1)  # max_len unused here
    meta = json.load(open(args.cache.replace(".npz", ".meta.json")))
    num_items = meta["vocab_size"]
    vids, offsets, is_test = data.video_ids, data.offsets, data.is_test

    # Train-window co-watch pairs: event e (train) with a neighbor e-d, d in [1, W],
    # same user, neighbor also train. Sampled on the fly per batch.
    train_events = np.where(~is_test & ((np.arange(len(vids)) - data.event_start) >= 1))[0]
    print(f"items={num_items:,} | train events={len(train_events):,} | window={args.window}")

    # Fixed val pairs for the adjacency gate: (last train watch, first val watch)
    rng = np.random.RandomState(args.seed)
    val_idx = rng.choice(data.val_pool, 4096, replace=False)
    val_a = np.array([vids[max(data.event_start[e], e - 1)] for e in val_idx])
    val_b = vids[val_idx]

    key = jax.random.PRNGKey(args.seed)
    table = jax.random.normal(key, (num_items + 1, args.dim)) * 0.05
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(table)

    @jax.jit
    def step(table, opt_state, a_ids, b_ids, negs):
        def loss_fn(tbl):
            a = tbl[a_ids]
            b = tbl[b_ids]
            n = tbl[negs]
            a = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
            b = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
            n = n / (jnp.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
            pos = jnp.sum(a * b, axis=-1) / args.temp
            neg = (a @ n.T) / args.temp
            neg = jnp.where(negs[None, :] == b_ids[:, None], -1e9, neg)
            lg = jnp.concatenate([pos[:, None], neg], axis=-1)
            return -jax.nn.log_softmax(lg, axis=-1)[:, 0].mean()

        loss, grads = jax.value_and_grad(loss_fn)(table)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(table, updates), opt_state, loss

    def adjacency_gap(table):
        t = np.asarray(table)
        en = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)
        rnd = np.random.RandomState(0).randint(1, num_items + 1, size=len(val_b))
        return float((en[val_a] * en[val_b]).sum(1).mean() - (en[val_a] * en[rnd]).sum(1).mean())

    if args.smoke:
        args.pairs_per_epoch, args.epochs = 200 * args.batch_size, 1

    best_gap, best_epoch, patience = -1.0, 0, 0
    steps_per_epoch = args.pairs_per_epoch // args.batch_size
    for epoch in range(1, args.epochs + 1):
        t0, tot = time.time(), 0.0
        for s in range(steps_per_epoch):
            e = rng.choice(train_events, args.batch_size)
            d = rng.randint(1, args.window + 1, size=args.batch_size)
            nb = np.maximum(e - d, data.event_start[e])          # same-user clamp
            keep = ~is_test[nb]                                  # neighbor must be train
            a_ids, b_ids = vids[e], vids[nb]
            b_ids = np.where(keep, b_ids, a_ids)                 # degenerate pairs get masked by identity (no-op-ish)
            negs = rng.randint(1, num_items + 1, size=args.num_negatives)
            table, opt_state, loss = step(table, opt_state,
                                          jnp.array(a_ids), jnp.array(b_ids),
                                          jnp.array(negs, dtype=jnp.int32))
            tot += float(loss)
            if not np.isfinite(tot):
                log_metrics(fatal="nan_loss", epoch=epoch, step=s)
                raise SystemExit(2)
        gap = adjacency_gap(table)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d} | loss {tot/steps_per_epoch:.4f} | {args.pairs_per_epoch/dt:.0f} pairs/s | adjΔ {gap:.4f}")
        log_metrics(epoch=epoch, loss=round(tot / steps_per_epoch, 5), adjacency_gap=round(gap, 5),
                    pairs_per_sec=round(args.pairs_per_epoch / dt))
        if gap > best_gap:
            best_gap, best_epoch, patience = gap, epoch, 0
            if not args.smoke:
                with open(os.path.join(args.exp_dir, "item2vec_table.msgpack"), "wb") as f:
                    f.write(flax.serialization.to_bytes({"table": table, "epoch": epoch, "adj_gap": gap}))
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stop at {epoch} (best {best_epoch})")
                break
    log_metrics(final=True, best_epoch=best_epoch, best_adj_gap=round(best_gap, 5))
    print(f"best adjacency gap: {best_gap:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
