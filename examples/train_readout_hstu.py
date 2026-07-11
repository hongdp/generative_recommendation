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
def build_multi_position_data(user_history, max_len, user_timestamps=None):
    """Input x = train items[:-1] (last max_len), labels y = train items[1:].

    Train items are seq[:-2] (leave-one-out protocol: seq[-2] is val,
    seq[-1] is test). Left-padded with 0; label 0 = no loss at that position.
    With user_timestamps, also returns each label event's timestamp (§3.2:
    fork i's request time = timestamp of the event following t_i); -1 at pads.
    """
    inputs, labels, label_ts = [], [], []
    for uid, seq in user_history.items():
        train_items = seq[:-2]
        if len(train_items) < 2:
            continue
        x = train_items[:-1][-max_len:]
        y = train_items[1:][-max_len:]
        pad = max_len - len(x)
        inputs.append([0] * pad + x)
        labels.append([0] * pad + y)
        if user_timestamps is not None:
            ts = user_timestamps[uid][:-2][1:][-max_len:]
            label_ts.append([-1] * pad + ts)
    inputs = np.array(inputs, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    if user_timestamps is not None:
        return inputs, labels, np.array(label_ts, dtype=np.int64)
    return inputs, labels


def time_featurizer(train_label_ts, seconds=False):
    """Returns ts -> [.., 6] features: weekly + annual sin/cos, normalized
    linear time (train-range), and a validity flag; missing ts (< 0) -> zeros."""
    days_all = train_label_ts[train_label_ts >= 0] / (86400.0 if seconds else 1.0)
    dmin, dmax = float(days_all.min()), float(days_all.max())

    def featurize(ts):
        ts = np.asarray(ts, dtype=np.float64)
        valid = ts >= 0
        days = np.where(valid, ts / (86400.0 if seconds else 1.0), 0.0)
        dow = 2 * np.pi * ((days % 7.0) / 7.0)
        doy = 2 * np.pi * ((days % 365.25) / 365.25)
        tn = (days - dmin) / max(dmax - dmin, 1.0)
        f = np.stack([np.sin(dow), np.cos(dow), np.sin(doy), np.cos(doy), tn,
                      np.ones_like(days)], axis=-1)
        return (f * valid[..., None]).astype(np.float32)

    return featurize


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
    parser.add_argument("--time_features", action="store_true",
                        help="Condition <begin> tokens on the fork's request time (§3.2/§10): weekly+annual "
                             "cycles, linear time, validity flag through a learned projection. begin arm only.")
    parser.add_argument("--feat_zero_init", action="store_true",
                        help="Zero-init the feature projection so training starts exactly at the "
                             "unconditioned baseline (stabilizer for weak-signal datasets).")
    parser.add_argument("--late_no_residual", action="store_true",
                        help="Ablation: late-fusion head without the residual connection (q = MLP([h;t]) "
                             "instead of h + MLP([h;t])); W2 gets standard init since zero-init would "
                             "zero the whole query.")
    parser.add_argument("--late_w2_std", action="store_true",
                        help="Ablation cell: residual late fusion but with standard-init W2 (isolates "
                             "the init effect from the residual effect in the 3-cell factorization).")
    parser.add_argument("--time_mode", type=str, default="input", choices=["input", "late"],
                        help="input: features condition the <begin> token before the transformer "
                             "(time can change WHAT gets aggregated). late: features are fused into "
                             "the finished readout vector via a residual MLP (zero-init last layer) — "
                             "time can only transform a fixed summary + add a time prior.")
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
    if args.time_features and args.readout != "begin":
        raise ValueError("--time_features conditions the <begin> token; use --readout begin")
    rj = "" if args.reinject == "none" else f"_reinj{args.reinject}"
    time_late = args.time_features and args.time_mode == "late"
    if time_late and args.reinject != "none":
        raise ValueError("--time_mode late and --reinject are not combined in this experiment")
    rj += ("_timelatenr" if time_late and args.late_no_residual
           else "_timelatestd" if time_late and args.late_w2_std
           else "_timelate" if time_late else "_time") if args.time_features else ""
    rj += "z" if args.feat_zero_init else ""
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
    train_feats = val_feats = test_feats = None
    if args.time_features:
        if not hasattr(loader, "user_timestamps"):
            raise ValueError(f"loader for {dataset} does not expose user_timestamps")
        seconds = dataset in ["beauty", "sports", "toys"]  # steam stores days
        train_inputs, train_labels, train_label_ts = build_multi_position_data(
            loader.user_history, max_len, loader.user_timestamps)
        featurize = time_featurizer(train_label_ts, seconds=seconds)
        train_feats = featurize(train_label_ts)                      # [N_users, L, 6]
        from datasets.base import build_sequence_user_ids
        val_uids = build_sequence_user_ids(loader.user_history, "val")
        test_uids = build_sequence_user_ids(loader.user_history, "test")
        val_feats = featurize(np.array([loader.user_timestamps[u][-2] for u in val_uids]))
        test_feats = featurize(np.array([loader.user_timestamps[u][-1] for u in test_uids]))
    else:
        train_inputs, train_labels = build_multi_position_data(loader.user_history, max_len)
    n_positions = int((train_labels > 0).sum())
    print(f"Train: {len(train_inputs)} users, {n_positions} supervised positions")

    val_inputs, val_targets = loader.get_split("val", max_len=max_len, format_type="index").to_numpy()
    test_inputs, test_targets = loader.get_split("test", max_len=max_len, format_type="index").to_numpy()
    print(f"Val: {len(val_targets)} | Test: {len(test_targets)}")
    if args.time_features:
        assert len(val_feats) == len(val_targets) and len(test_feats) == len(test_targets)

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
        feat_zero_init=args.feat_zero_init,
    )

    def to_tokens(x):
        if args.readout == "item":
            return x
        branches = jnp.full((x.shape[0], max_len), begin_id, dtype=jnp.int32)
        return jnp.concatenate([x, branches], axis=1)

    feat_mask_full = jnp.array((np.arange(n_total) >= max_len).astype(np.float32)[None, :, None])

    def expand_feats(f):
        """[B, L, 6] per-fork or [B, 6] per-sample -> [B, N, 6], zeros on context rows."""
        z = np.zeros((f.shape[0], n_total, 6), dtype=np.float32)
        z[:, max_len:, :] = f if f.ndim == 3 else f[:, None, :]
        return z

    key = jax.random.PRNGKey(args.seed)
    dummy = jnp.zeros((1, n_total), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, n_total, n_total), dtype=bool)
    init_kw = {}
    if args.time_features and not time_late:
        init_kw = {"feats": jnp.zeros((1, n_total, 6)), "feat_mask": feat_mask_full}
    params = model.init(key, dummy, dummy_mask, rel_idx, **init_kw)["params"]
    # Wrapped param trees: re-injection adds a gate alpha (masked out of weight
    # decay so its fitted value reflects the data, not the regularizer); late
    # fusion adds a residual MLP head with a zero-init output layer (starts at
    # the unconditioned baseline, same principle as --feat_zero_init).
    wrapped = args.reinject != "none" or time_late
    if args.reinject != "none":
        alpha0 = jnp.full(() if args.reinject == "scalar" else (args.embedding_dim,), args.alpha_init)
        params = {"model": params, "alpha": alpha0}
    elif time_late:
        key, lk = jax.random.split(key)
        d = args.embedding_dim
        lk1, lk2 = jax.random.split(lk)
        w2 = (jax.random.normal(lk2, (d, d)) / np.sqrt(d)
              if (args.late_no_residual or args.late_w2_std) else jnp.zeros((d, d)))
        params = {"model": params, "late": {
            "W1": jax.random.normal(lk1, (d + 6, d)) / np.sqrt(d + 6), "b1": jnp.zeros(d),
            "W2": w2, "b2": jnp.zeros(d),
        }}
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {n_params / 1e6:.2f}M")

    if wrapped:
        wd_mask = {k: (False if k == "alpha" else jax.tree_util.tree_map(lambda _: True, v))
                   for k, v in params.items()}
        optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay, mask=wd_mask)
    else:
        optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    def model_tree(p):
        return p["model"] if wrapped else p

    def late_apply(lp, h, f):
        """Late-fusion head: h + MLP(concat(h, time)) (W2 zero-init), or the
        non-residual ablation q = MLP(concat(h, time)) (W2 standard init)."""
        z = jax.nn.gelu(jnp.concatenate([h, f], axis=-1) @ lp["W1"] + lp["b1"])
        out = z @ lp["W2"] + lp["b2"]
        return out if args.late_no_residual else h + out

    @jax.jit
    def train_step(params, opt_state, x, labels, negs, dropout_key, feats=None):
        def loss_fn(p):
            mp = model_tree(p)
            tokens = to_tokens(x)
            mask = make_mask(x, anchor, max_len)
            fkw = {} if (feats is None or time_late) else {"feats": feats, "feat_mask": feat_mask_full}
            h, out_table = model.apply(
                {"params": mp}, tokens, mask, rel_idx,
                rngs={"dropout": dropout_key}, deterministic=False, **fkw,
            )
            # Readout states aligned with labels [batch, L, d]
            h_ro = h[:, :max_len] if args.readout == "item" else h[:, max_len:]
            if time_late:
                h_ro = late_apply(p["late"], h_ro, feats[:, max_len:, :])
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
    def predict_scores(params, x, feats=None):
        """Full-catalog scores from the serving readout (last real position)."""
        mp = model_tree(params)
        tokens = to_tokens(x)
        mask = make_mask(x, anchor, max_len)
        fkw = {} if (feats is None or time_late) else {"feats": feats, "feat_mask": feat_mask_full}
        h, out_table = model.apply({"params": mp}, tokens, mask, rel_idx, deterministic=True, **fkw)
        # item arm: stream of the most recent item; begin arm: b_{L-1}
        h_q = h[:, max_len - 1] if args.readout == "item" else h[:, -1]
        if time_late:
            h_q = late_apply(params["late"], h_q, feats[:, -1, :])
        if args.reinject != "none":
            h_q = h_q + params["alpha"] * mp["item_embedding"]["embedding"][x[:, -1]]
        return h_q @ out_table.T                                   # [B, num_items + 1]

    from evaluation.metrics import compute_ranks, calculate_metrics_from_ranks

    def run_eval(params, inputs, targets, feats=None):
        ranks = []
        for i in range(0, len(inputs), 512):
            fb = jnp.array(expand_feats(feats[i : i + 512])) if feats is not None else None
            s = predict_scores(params, jnp.array(inputs[i : i + 512]), fb)
            ranks.append(np.array(compute_ranks(s, targets[i : i + 512])))
        return calculate_metrics_from_ranks(np.concatenate(ranks), [1, 5, 10, 20])

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
                fb = jnp.array(expand_feats(train_feats[idx])) if args.time_features else None
                params, opt_state, loss = train_step(
                    params, opt_state, jnp.array(train_inputs[idx]), jnp.array(train_labels[idx]), negs, step_rng, fb
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

            val_results = run_eval(params, val_inputs, val_targets, val_feats)
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
    test_results = run_eval(best_params, test_inputs, test_targets, test_feats)
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
    diag_kw = {}
    if args.time_features and not time_late:
        diag_kw = {"feats": jnp.array(expand_feats(val_feats[: sample.shape[0]])), "feat_mask": feat_mask_full}
    h, out_table = model.apply({"params": best_model_params}, tokens, mask, rel_idx, deterministic=True, **diag_kw)
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
