"""Deep-K evaluation of saved KuaiRand readout checkpoints.

Motivated by the rank gradient in the A/B (experiments/kuairand_readout_ab_20260801):
the begin arm loses badly at HR@1 but the deficit shrinks monotonically with K,
and on the validation split it CROSSES OVER to a +8.5% advantage by HR@20. If the
crossover also happens on test at larger K, the readout token is a recall-at-depth
win rather than a loss — which is the metric a retrieval stage feeding a ranker
actually cares about.

Reproduces the training script's exact test sample (same seed and draw order), then
reports HR@K / NDCG@K for K up to 1000 from a single rank computation.

Run (PYTHONPATH=src):
  python examples/eval_kuairand_deepk.py --exp_dir experiments/kuairand_readout_ab_20260801
"""

import argparse
import json
import os

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from models.readout_hstu import ReadoutHSTUModel

KS = [1, 5, 10, 20, 50, 100, 200, 500, 1000]


class KuaiRandSequences:
    """Mirror of the training script's loader (single-target windows)."""

    def __init__(self, path, max_len):
        d = np.load(path)
        self.video_ids, self.time_ms, self.is_test = d["video_ids"], d["time_ms"], d["is_test"]
        self.offsets, self.max_len = d["offsets"], max_len
        counts = np.diff(self.offsets)
        self.event_start = np.repeat(self.offsets[:-1], counts)
        has_history = (np.arange(len(self.video_ids)) - self.event_start) >= 1
        self.train_pool = np.where(~self.is_test & has_history)[0]
        test_days = np.unique(self.day(self.time_ms[self.is_test]))
        self.val_day = test_days[0]
        ev_day = self.day(self.time_ms)
        self.val_pool = np.where(self.is_test & has_history & (ev_day == self.val_day))[0]
        self.test_pool = np.where(self.is_test & has_history & (ev_day > self.val_day))[0]

    @staticmethod
    def day(ms):
        return (ms // 86_400_000).astype(np.int64)

    def gather(self, event_idx):
        e, L = np.asarray(event_idx), self.max_len
        cols = e[:, None] + np.arange(-L, 0)[None, :]
        valid = cols >= self.event_start[e][:, None]
        x = np.where(valid, self.video_ids[np.clip(cols, 0, None)], 0).astype(np.int32)
        return x, self.video_ids[e].astype(np.int32)


def make_layout(max_len, readout, num_rel_buckets=64):
    ctx = np.arange(max_len)
    anchor = ctx if readout == "item" else np.concatenate([ctx, [max_len - 1]])
    pos = ctx if readout == "item" else np.concatenate([ctx, [max_len]])
    rel_idx = np.clip(pos[:, None] - pos[None, :], 0, num_rel_buckets - 1).astype(np.int32)
    return jnp.array(anchor, dtype=jnp.int32), jnp.array(rel_idx)


def make_mask(x, anchor, max_len):
    n = anchor.shape[0]
    k_ids = jnp.arange(n)
    valid_key = jnp.concatenate([x != 0, jnp.zeros((x.shape[0], n - max_len), dtype=bool)], axis=1)
    mask = ((k_ids < max_len)[None, :] & (k_ids[None, :] <= anchor[:, None]))[None, :, :] & valid_key[:, None, :]
    return mask | jnp.eye(n, dtype=bool)[None, :, :]


def eval_arm(arm, cfg, data, test_idx, chunk=256):
    max_len, num_items = cfg["max_len"], cfg["data_hash_meta"]["vocab_size"]
    anchor, rel_idx = make_layout(max_len, arm)
    n_total = int(anchor.shape[0])
    ro_pos = max_len - 1 if arm == "item" else n_total - 1
    begin_id = num_items + 1

    model = ReadoutHSTUModel(
        num_items=num_items, embedding_dim=cfg["embedding_dim"], num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"], attention_dim=cfg["attention_dim"], linear_dim=cfg["linear_dim"],
        attn_dropout_rate=cfg["dropout_rate"], linear_dropout_rate=cfg["dropout_rate"], tie_output=False,
    )
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, n_total), dtype=jnp.int32),
                        jnp.ones((1, n_total, n_total), dtype=bool), rel_idx)["params"]
    with open(cfg["ckpt"], "rb") as f:
        state = flax.serialization.from_bytes({"params": params, "epoch": 0, "best_val_ndcg": 0.0}, f.read())
    params, best_ep = state["params"], state["epoch"]

    @jax.jit
    def scores(p, x):
        tokens = x if arm == "item" else jnp.concatenate(
            [x, jnp.full((x.shape[0], 1), begin_id, dtype=jnp.int32)], axis=1)
        h, out_table = model.apply({"params": p}, tokens, make_mask(x, anchor, max_len), rel_idx,
                                   deterministic=True)
        return h[:, ro_pos] @ out_table.T

    ranks = []
    for i in range(0, len(test_idx), chunk):
        x, y = data.gather(test_idx[i:i + chunk])
        s = np.array(scores(params, jnp.array(x)))
        s[:, 0] = -np.inf                                     # padding id is not a candidate
        ranks.append((s > s[np.arange(len(y)), y][:, None]).sum(axis=1) + 1)
        if (i // chunk) % 200 == 0:
            print(f"  [{arm}] {i:,}/{len(test_idx):,}", flush=True)
    ranks = np.concatenate(ranks)
    out = {"best_epoch": int(best_ep), "n": int(len(ranks))}
    for k in KS:
        hit = ranks <= k
        out[f"HR@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float((hit / np.log2(ranks + 1)).mean())
    out["MRR"] = float((1.0 / ranks).mean())
    out["_ranks"] = ranks
    return out


def main():
    ap = argparse.ArgumentParser(description="Deep-K eval of saved readout A/B checkpoints.")
    ap.add_argument("--exp_dir", type=str, required=True)
    ap.add_argument("--cache", type=str, default="./data/kuairand/kuairand27k_top100000.npz")
    ap.add_argument("--val_cap", type=int, default=50_000)
    ap.add_argument("--test_cap", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = json.load(open(os.path.join(args.exp_dir, "item", "config.json")))
    data = KuaiRandSequences(args.cache, base["max_len"])
    # Replicate the training script's draw order exactly (val first, then test).
    rng = np.random.RandomState(args.seed)
    rng.choice(data.val_pool, min(args.val_cap, len(data.val_pool)), replace=False)
    test_idx = rng.choice(data.test_pool, min(args.test_cap, len(data.test_pool)), replace=False)
    print(f"test targets: {len(test_idx):,}")

    results = {}
    for arm in ["item", "begin"]:
        cfg = json.load(open(os.path.join(args.exp_dir, arm, "config.json")))
        cfg["ckpt"] = os.path.join(args.exp_dir, arm, "best_checkpoint.msgpack")
        print(f"\n=== {arm} (ckpt {cfg['ckpt']}) ===", flush=True)
        results[arm] = eval_arm(arm, cfg, data, test_idx)

    it, bg = results["item"], results["begin"]
    print(f"\n{'K':>6} {'item HR@K':>11} {'begin HR@K':>12} {'rel':>8} {'hits(i/b)':>15} {'sigma':>7}")
    for k in KS:
        i, b = it[f"HR@{k}"], bg[f"HR@{k}"]
        hi, hb = i * it["n"], b * bg["n"]
        sig = (hb - hi) / np.sqrt(max(hi + hb, 1))
        print(f"{k:>6} {i:>11.5f} {b:>12.5f} {(b/i-1)*100:>7.1f}% {hi:>7.0f}/{hb:<7.0f} {sig:>7.1f}")
    print(f"\n{'K':>6} {'item NDCG@K':>13} {'begin NDCG@K':>14} {'rel':>8}")
    for k in KS:
        i, b = it[f"NDCG@{k}"], bg[f"NDCG@{k}"]
        print(f"{k:>6} {i:>13.5f} {b:>14.5f} {(b/i-1)*100:>7.1f}%")
    print(f"\nMRR: item {it['MRR']:.5f} | begin {bg['MRR']:.5f} ({(bg['MRR']/it['MRR']-1)*100:+.1f}%)")
    print(f"median rank: item {np.median(it['_ranks']):.0f} | begin {np.median(bg['_ranks']):.0f}")

    out = {a: {k: v for k, v in r.items() if not k.startswith("_")} for a, r in results.items()}
    with open(os.path.join(args.exp_dir, "deepk_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(args.exp_dir, "deepk_ranks.npz"),
                        item=it["_ranks"], begin=bg["_ranks"])
    print(f"\nwrote {args.exp_dir}/deepk_eval.json and deepk_ranks.npz")


if __name__ == "__main__":
    main()
