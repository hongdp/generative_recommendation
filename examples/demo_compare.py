"""Demo: Compare HSTU vs TIGER Flow predictions side-by-side.

Shows the same test users with predictions from both models.
"""

import argparse
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import flax.serialization

from datasets import SteamDataLoader
from models.hstu import HSTUModel
from models.tiger_flow import TIGERFlowModel


def get_title(item_id, token_to_title):
    """Convert internal item ID to human-readable title."""
    if item_id == 0:
        return "(padding)"
    title = token_to_title.get(item_id, None)
    if title:
        return title.strip()
    return f"[Unknown ID={item_id}]"


def main():
    parser = argparse.ArgumentParser(description="Compare HSTU vs TIGER Flow")
    parser.add_argument("--hstu_ckpt", type=str,
        default="./data/hstu_steam_checkpoints/best_checkpoint.msgpack")
    parser.add_argument("--flow_ckpt", type=str,
        default="./data/tiger_flow_steam_checkpoints/best_checkpoint.msgpack")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--denoise_steps", type=int, default=3)
    args = parser.parse_args()

    # =========================================================================
    # 1. Load dataset
    # =========================================================================
    loader = SteamDataLoader(data_dir="./data")
    token_to_title = loader.token_to_title
    num_items = loader.num_items
    max_len = 20

    test_dataset = loader.get_split("test", max_len=max_len, format_type="index")
    test_in, test_tar = test_dataset.to_numpy()
    print(f"Test set: {len(test_tar)} samples, Items: {num_items}")

    # =========================================================================
    # 2. Load HSTU model
    # =========================================================================
    print("Loading HSTU model...")
    hstu = HSTUModel(
        num_items=num_items,
        embedding_dim=256, num_blocks=4, num_heads=4,
        attention_dim=128, linear_dim=512,
        max_sequence_len=max_len,
    )
    key = jax.random.PRNGKey(0)
    dummy_seq = jnp.zeros((1, max_len), dtype=jnp.int32)
    hstu_vars = hstu.init(key, dummy_seq)

    with open(args.hstu_ckpt, "rb") as f:
        hstu_ckpt = flax.serialization.from_bytes(hstu_vars, f.read())
    hstu_params = hstu_ckpt.get("params", hstu_ckpt)
    if "params" in hstu_params:
        hstu_params = hstu_params["params"]
    print(f"  Loaded: {args.hstu_ckpt}")

    @jax.jit
    def hstu_predict(params, item_seq):
        logits = hstu.apply({"params": params}, item_seq, deterministic=True)
        return logits[:, -1, :]  # [batch, num_items+1] — last position

    # =========================================================================
    # 3. Load TIGER Flow model
    # =========================================================================
    print("Loading TIGER Flow model...")
    from sklearn.decomposition import PCA as PCAModel

    latent_dim = 256
    Z_frozen = np.load("./data/steam_t5_embeddings.npy").astype(np.float32)
    Z_frozen[0] = 0.0

    pca = PCAModel(n_components=latent_dim)
    pca.fit(Z_frozen[1:])
    Z_pca = np.zeros((Z_frozen.shape[0], latent_dim), dtype=np.float32)
    Z_pca[1:] = pca.transform(Z_frozen[1:])
    mu = Z_pca[1:].mean(axis=0, keepdims=True)
    std = Z_pca[1:].std(axis=0, keepdims=True) + 1e-8
    Z_latent = np.zeros_like(Z_pca)
    Z_latent[1:] = (Z_pca[1:] - mu) / std

    norms = np.linalg.norm(Z_latent[1:], axis=-1, keepdims=True) + 1e-8
    Z_normed = np.zeros_like(Z_latent)
    Z_normed[1:] = Z_latent[1:] / norms
    Z_latent_jnp = jnp.array(Z_latent)
    Z_normed_jnp = jnp.array(Z_normed)

    flow_model = TIGERFlowModel(
        embedding_dim=384, latent_dim=latent_dim,
        num_blocks=4, num_heads=6, attention_dim=384, linear_dim=1024,
        max_encoder_len=max_len,
    )
    dummy_enc = jnp.zeros((1, max_len, latent_dim))
    dummy_mask = jnp.ones((1, max_len))
    dummy_z = jnp.zeros((1, latent_dim))
    dummy_t = jnp.zeros((1,))
    flow_vars = flow_model.init(key, dummy_enc, dummy_mask, dummy_z, dummy_t)

    with open(args.flow_ckpt, "rb") as f:
        flow_ckpt = flax.serialization.from_bytes(flow_vars, f.read())
    flow_params = flow_ckpt.get("params", flow_ckpt)
    if "params" in flow_params:
        flow_params = flow_params["params"]
    print(f"  Loaded: {args.flow_ckpt}")

    @jax.jit
    def flow_encode(params, enc_lat, enc_mask):
        return flow_model.apply(
            {"params": params}, enc_lat, enc_mask,
            method=flow_model.encode, deterministic=True)

    @jax.jit
    def flow_velocity(params, enc_out, enc_mask, z, t):
        return flow_model.apply(
            {"params": params}, enc_out, enc_mask, z, t,
            method=flow_model.predict_velocity, deterministic=True)

    def flow_denoise(params, enc_out, enc_mask, z_T, steps):
        dt = 1.0 / steps
        z = z_T
        for s in range(steps):
            t = jnp.full((z.shape[0],), 1.0 - s * dt)
            v = flow_velocity(params, enc_out, enc_mask, z, t)
            z = z - dt * v
        return z

    # =========================================================================
    # 4. Run comparison demo
    # =========================================================================
    np.random.seed(42)
    sample_indices = np.random.choice(len(test_tar), args.num_samples, replace=False)

    print("\n" + "=" * 90)
    print("  HSTU vs TIGER Flow — Side-by-Side Prediction Comparison")
    print("=" * 90)

    for idx_num, sample_idx in enumerate(sample_indices):
        history_ids = test_in[sample_idx]
        target_id = int(test_tar[sample_idx])
        history_items = [int(h) for h in history_ids if h != 0]

        print(f"\n{'━' * 90}")
        print(f"  Sample {idx_num + 1}/{args.num_samples}")
        print(f"{'━' * 90}")

        # --- History ---
        print(f"\n  📜 User History ({len(history_items)} games):")
        for i, hid in enumerate(history_items):
            marker = "  " if i < len(history_items) - 1 else "→ "
            print(f"    {marker}{i+1:2d}. {get_title(hid, token_to_title)}")

        # --- Ground truth ---
        target_title = get_title(target_id, token_to_title)
        print(f"\n  🎯 Ground Truth: {target_title}")

        # --- HSTU predictions ---
        seq = jnp.array(history_ids)[None]  # [1, max_len]
        logits = hstu_predict(hstu_params, seq)  # [1, num_items+1]
        logits_np = np.array(logits[0])
        logits_np[0] = -1e9  # mask padding
        # Mask history items
        for hid in history_items:
            logits_np[hid] = -1e9
        top_k_hstu = np.argsort(-logits_np)[:args.top_k]
        top_k_scores = logits_np[top_k_hstu]

        print(f"\n  🧠 HSTU Predictions:")
        hstu_hit = False
        for rank, (pred_id, score) in enumerate(zip(top_k_hstu, top_k_scores)):
            pred_title = get_title(int(pred_id), token_to_title)
            is_hit = int(pred_id) == target_id
            marker = "✅" if is_hit else "  "
            print(f"    {marker} {rank+1:2d}. [{score:7.2f}] {pred_title}")
            if is_hit:
                hstu_hit = True
        if not hstu_hit:
            # Find actual rank
            rank_of_target = int(np.where(np.argsort(-logits_np) == target_id)[0][0]) + 1
            print(f"    ❌ Target at rank {rank_of_target}")

        # --- TIGER Flow predictions ---
        enc_lat = Z_latent_jnp[history_ids][None]
        enc_mask = jnp.array((history_ids != 0).astype(np.float32))[None]
        enc_out = flow_encode(flow_params, enc_lat, enc_mask)

        enc_out_rep = jnp.repeat(enc_out, args.num_seeds, axis=0)
        enc_mask_rep = jnp.repeat(enc_mask, args.num_seeds, axis=0)
        z_T = jax.random.normal(jax.random.PRNGKey(sample_idx), (args.num_seeds, latent_dim))
        z_hat = flow_denoise(flow_params, enc_out_rep, enc_mask_rep, z_T, args.denoise_steps)

        z_hat_norm = z_hat / (jnp.linalg.norm(z_hat, axis=-1, keepdims=True) + 1e-8)
        scores = z_hat_norm @ Z_normed_jnp[1:].T
        top1_items = np.array(jnp.argmax(scores, axis=-1)) + 1
        top1_scores = np.array(jnp.max(scores, axis=-1))

        sorted_idx = np.argsort(-top1_scores)
        seen = set()
        flow_preds = []
        for si in sorted_idx:
            item = int(top1_items[si])
            conf = float(top1_scores[si])
            if item not in seen:
                seen.add(item)
                flow_preds.append((item, conf))
            if len(flow_preds) >= args.top_k:
                break

        print(f"\n  🔮 TIGER Flow Predictions:")
        flow_hit = False
        for rank, (pred_id, conf) in enumerate(flow_preds):
            pred_title = get_title(pred_id, token_to_title)
            is_hit = pred_id == target_id
            marker = "✅" if is_hit else "  "
            print(f"    {marker} {rank+1:2d}. [{conf:.4f}] {pred_title}")
            if is_hit:
                flow_hit = True
        if not flow_hit:
            print(f"    ❌ Target not in top-{args.top_k}")

    print(f"\n{'=' * 90}")


if __name__ == "__main__":
    main()
