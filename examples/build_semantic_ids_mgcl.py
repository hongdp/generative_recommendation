"""Poor-man's UniSID MGCL: multi-granularity contrastive Semantic IDs (beauty).

Reimplements the core of UniSID's Multi-Granularity Contrastive Learning
(arXiv 2602.10445) without the MLLM backbone: three linear projection heads over
frozen sentence-t5-XXL item embeddings produce per-level logits z_i^l in R^K.
The logits double as (a) contrastive representations and (b) code logits —
the discrete code is argmax(z_i^l), exactly as in the paper.

Per-level positives (granularity from the Amazon category taxonomy):
  L1: same depth-3 category         (coarse,  ~46 classes)
  L2: same full leaf category path  (fine,    ~600 classes)
  L3: the item itself, two dropout views (instance discrimination)

A code-usage entropy bonus counteracts argmax collapse (the paper gets this
for free from InfoNCE uniformity at scale; at 12k items we make it explicit).
Output: 4-level IDs (3 learned + frequency-ordered dedup), pipeline-compatible.

Run WITH PYTHONPATH=src (repo imports only; no sentence-transformers needed).
"""

import argparse
import ast
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax


def build_category_labels(data_dir, dataset):
    sys.path.insert(0, "examples")
    from build_semantic_ids_rich import parse_rich_metadata, CATEGORY_DIRS
    from datasets import AmazonDataLoader
    import os

    loader = AmazonDataLoader(category=dataset, data_dir=data_dir, min_rating=0)
    cat_dir = CATEGORY_DIRS[dataset]
    meta = parse_rich_metadata(os.path.join(data_dir, "amazon", cat_dir, f"meta_{cat_dir}.json.gz"))
    n = loader.num_items

    coarse, fine = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)
    c_map, f_map = {}, {}
    for tok in range(1, n + 1):
        m = meta.get(loader.id_to_item[tok], {})
        cats = m.get("categories")
        if isinstance(cats, str):
            try:
                cats = ast.literal_eval(cats)
            except Exception:
                cats = None
        path = [str(c) for c in cats[0]] if (cats and isinstance(cats, list) and cats[0]) else ["Beauty"]
        ck, fk = " > ".join(path[:3]), " > ".join(path)
        coarse[tok - 1] = c_map.setdefault(ck, len(c_map))
        fine[tok - 1] = f_map.setdefault(fk, len(f_map))
    print(f"granularity classes: coarse={len(c_map)}, fine={len(f_map)}")
    return loader, coarse, fine


def supcon_loss(z, labels, tau):
    """Supervised contrastive loss over L2-normalized representations z [N,D]."""
    z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    sim = z @ z.T / tau                                   # [N,N]
    n = z.shape[0]
    eye = jnp.eye(n, dtype=bool)
    pos = (labels[:, None] == labels[None, :]) & ~eye     # positive mask
    logits = jnp.where(eye, -1e9, sim)                    # exclude self from denominator
    log_prob = logits - jax.nn.logsumexp(logits, axis=1, keepdims=True)
    pos_cnt = jnp.maximum(pos.sum(axis=1), 1)
    per_anchor = (jnp.where(pos, log_prob, 0.0).sum(axis=1)) / pos_cnt
    has_pos = pos.sum(axis=1) > 0
    return -jnp.where(has_pos, per_anchor, 0.0).sum() / jnp.maximum(has_pos.sum(), 1)


def instance_loss(z1, z2, tau):
    """NT-Xent between two views (SimCLR): positives are the paired views."""
    z1 = z1 / (jnp.linalg.norm(z1, axis=-1, keepdims=True) + 1e-8)
    z2 = z2 / (jnp.linalg.norm(z2, axis=-1, keepdims=True) + 1e-8)
    sim = z1 @ z2.T / tau                                 # [N,N]
    labels = jnp.arange(z1.shape[0])
    l12 = optax.softmax_cross_entropy_with_integer_labels(sim, labels).mean()
    l21 = optax.softmax_cross_entropy_with_integer_labels(sim.T, labels).mean()
    return 0.5 * (l12 + l21)


def usage_entropy_bonus(logits):
    """Negative entropy of the mean code distribution (maximize usage spread)."""
    p = jax.nn.softmax(logits, axis=-1).mean(axis=0)
    return jnp.sum(p * jnp.log(p + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="beauty")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--embeddings_path", type=str, default="./data/item_emb_rich_xxl_beauty_std.npy")
    ap.add_argument("--output_path", type=str, default="./data/semantic_ids_mgcl_beauty.json")
    ap.add_argument("--num_codes", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--view_dropout", type=float, default=0.2, help="Input dropout for the two L3 instance views.")
    ap.add_argument("--entropy_weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    loader, coarse, fine = build_category_labels(args.data_dir, args.dataset)
    emb = np.load(args.embeddings_path)  # (n+1, D), row 0 = padding
    x_all = jnp.array(emb[1:])
    n, D = x_all.shape
    K = args.num_codes
    print(f"items={n}, embed dim={D}, codes/level={K}")

    key = jax.random.PRNGKey(args.seed)
    k1, k2, k3, key = jax.random.split(key, 4)
    init = jax.nn.initializers.orthogonal()
    params = {
        "W1": init(k1, (D, K)), "b1": jnp.zeros(K),
        "W2": init(k2, (D, K)), "b2": jnp.zeros(K),
        "W3": init(k3, (D, K)), "b3": jnp.zeros(K),
    }
    opt = optax.adamw(args.lr, weight_decay=1e-4)
    opt_state = opt.init(params)
    coarse_j, fine_j = jnp.array(coarse), jnp.array(fine)

    def heads(params, x):
        return (x @ params["W1"] + params["b1"],
                x @ params["W2"] + params["b2"],
                x @ params["W3"] + params["b3"])

    @jax.jit
    def step(params, opt_state, idx, rng):
        def loss_fn(p):
            xb = x_all[idx]
            z1, z2, _ = heads(p, xb)
            l1 = supcon_loss(z1, coarse_j[idx], args.tau)
            l2 = supcon_loss(z2, fine_j[idx], args.tau)
            ra, rb = jax.random.split(rng)
            va = xb * jax.random.bernoulli(ra, 1 - args.view_dropout, xb.shape) / (1 - args.view_dropout)
            vb = xb * jax.random.bernoulli(rb, 1 - args.view_dropout, xb.shape) / (1 - args.view_dropout)
            z3a = va @ p["W3"] + p["b3"]
            z3b = vb @ p["W3"] + p["b3"]
            l3 = instance_loss(z3a, z3b, args.tau)
            ent = (usage_entropy_bonus(z1) + usage_entropy_bonus(z2) + usage_entropy_bonus(z3a)) / 3
            return l1 + l2 + l3 + args.entropy_weight * ent, (l1, l2, l3)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    np_rng = np.random.RandomState(args.seed)
    for s in range(1, args.steps + 1):
        idx = jnp.array(np_rng.choice(n, size=min(args.batch_size, n), replace=False))
        key, rng = jax.random.split(key)
        params, opt_state, loss, (l1, l2, l3) = step(params, opt_state, idx, rng)
        if s % 200 == 0 or s == 1:
            print(f"step {s:4d} | total {float(loss):.4f} | L1 {float(l1):.4f} L2 {float(l2):.4f} L3 {float(l3):.4f}")

    # Extract codes: argmax of each head's logits (deterministic, no dropout)
    z1, z2, z3 = heads(params, x_all)
    codes = np.stack([np.array(jnp.argmax(z, axis=-1)) for z in (z1, z2, z3)], axis=1)
    for lvl in range(3):
        u = len(np.unique(codes[:, lvl]))
        print(f"L{lvl+1} code usage: {u}/{K}")
    triples = set(map(tuple, codes.tolist()))
    print(f"3-level collisions: {1 - len(triples)/n:.2%}")

    # dedup 4th level, frequency-ordered
    from collections import Counter, defaultdict
    tr_in, tr_tar = loader.get_split("train", max_len=20, format_type="index").to_numpy()
    freq = Counter(tr_in[tr_in > 0].tolist()); freq.update(tr_tar[tr_tar > 0].tolist())
    groups = defaultdict(list)
    for tok in range(1, n + 1):
        groups[tuple(codes[tok - 1])].append(tok)
    out = {0: [0, 0, 0, 0]}
    mg = 0
    for t, items in groups.items():
        items.sort(key=lambda it: (-freq.get(it, 0), it))
        mg = max(mg, len(items))
        for i, item in enumerate(items):
            out[item] = [int(c) for c in t] + [i]
    assert mg <= K, mg
    quads = set(map(tuple, (v for k, v in out.items() if k != 0)))
    assert len(quads) == n
    print(f"dedup max group: {mg}")
    with open(args.output_path, "w") as f:
        json.dump({str(k): v for k, v in out.items()}, f)
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
