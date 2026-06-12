"""Demo: Show model predictions vs ground truth for sampled users.

Loads a trained TIGER Flow checkpoint, samples users from the test set,
and displays their interaction history, model predictions, and ground truth
with human-readable game titles.
"""

import argparse
import pickle
import jax
import jax.numpy as jnp
import numpy as np

from datasets import SteamDataLoader
from models.tiger_flow import TIGERFlowModel


def get_title(item_id, id_to_item, token_to_title):
    """Convert internal item ID to human-readable title."""
    if item_id == 0:
        return "(padding)"
    # token_to_title keys are internal item IDs (int)
    title = token_to_title.get(item_id, None)
    if title:
        return title.strip()
    return f"[Unknown ID={item_id}]"


def main():
    parser = argparse.ArgumentParser(description="Demo: TIGER Flow predictions")
    parser.add_argument("--checkpoint", type=str,
        default="./data/tiger_flow_steam_checkpoints/best_checkpoint.msgpack")
    parser.add_argument("--dataset", type=str, default="steam")
    parser.add_argument("--num_samples", type=int, default=5,
        help="Number of test users to show.")
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--denoise_steps", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=10,
        help="Number of top predictions to display.")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--attention_dim", type=int, default=384)
    parser.add_argument("--linear_dim", type=int, default=1024)
    args = parser.parse_args()

    # =========================================================================
    # 1. Load dataset & metadata
    # =========================================================================
    loader = SteamDataLoader(data_dir="./data")
    id_to_item = loader.id_to_item
    token_to_title = loader.token_to_title

    max_len = 20
    test_dataset = loader.get_split("test", max_len=max_len, format_type="index")
    test_in, test_tar = test_dataset.to_numpy()
    print(f"Test set: {len(test_tar)} samples")

    # =========================================================================
    # 2. Load frozen embeddings + PCA (matching training)
    # =========================================================================
    from sklearn.decomposition import PCA

    Z_frozen_raw = np.load("./data/steam_t5_embeddings.npy").astype(np.float32)
    Z_frozen_raw[0] = 0.0

    # PCA + standardization (same as train_tiger_flow.py)
    pca = PCA(n_components=args.latent_dim)
    pca.fit(Z_frozen_raw[1:])
    Z_pca = np.zeros((Z_frozen_raw.shape[0], args.latent_dim), dtype=np.float32)
    Z_pca[1:] = pca.transform(Z_frozen_raw[1:])
    mu = Z_pca[1:].mean(axis=0, keepdims=True)
    std = Z_pca[1:].std(axis=0, keepdims=True) + 1e-8
    Z_latent = np.zeros_like(Z_pca)
    Z_latent[1:] = (Z_pca[1:] - mu) / std

    # Normalized for ANN
    norms = np.linalg.norm(Z_latent[1:], axis=-1, keepdims=True) + 1e-8
    Z_normed = np.zeros_like(Z_latent)
    Z_normed[1:] = Z_latent[1:] / norms

    Z_latent_jnp = jnp.array(Z_latent)
    Z_normed_jnp = jnp.array(Z_normed)

    # =========================================================================
    # 3. Initialize & load model
    # =========================================================================
    model = TIGERFlowModel(
        embedding_dim=args.embedding_dim,
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        max_encoder_len=max_len,
    )

    key = jax.random.PRNGKey(0)
    dummy_enc = jnp.zeros((1, max_len, args.latent_dim))
    dummy_mask = jnp.ones((1, max_len))
    dummy_z = jnp.zeros((1, args.latent_dim))
    dummy_t = jnp.zeros((1,))
    variables = model.init(key, dummy_enc, dummy_mask, dummy_z, dummy_t)

    # Load checkpoint
    import flax.serialization
    with open(args.checkpoint, "rb") as f:
        ckpt = flax.serialization.from_bytes(variables, f.read())
    params = ckpt.get("params", ckpt)
    if "params" in params:
        params = params["params"]
    print(f"Loaded checkpoint: {args.checkpoint}")

    # =========================================================================
    # 4. JIT-compiled inference
    # =========================================================================
    @jax.jit
    def encode(params, enc_latents, enc_mask):
        return model.apply(
            {"params": params}, enc_latents, enc_mask,
            method=model.encode, deterministic=True,
        )

    @jax.jit
    def predict_v(params, enc_out, enc_mask, z, t):
        return model.apply(
            {"params": params}, enc_out, enc_mask, z, t,
            method=model.predict_velocity, deterministic=True,
        )

    def denoise(params, enc_out, enc_mask, z_T, steps):
        dt = 1.0 / steps
        z = z_T
        for s in range(steps):
            t = jnp.full((z.shape[0],), 1.0 - s * dt)
            v = predict_v(params, enc_out, enc_mask, z, t)
            z = z - dt * v
        return z

    # =========================================================================
    # 5. Run demo
    # =========================================================================
    np.random.seed(42)
    sample_indices = np.random.choice(len(test_tar), args.num_samples, replace=False)

    print("\n" + "=" * 80)
    print("  TIGER Flow — Prediction Demo (Steam Dataset)")
    print("=" * 80)

    for idx_num, sample_idx in enumerate(sample_indices):
        history_ids = test_in[sample_idx]  # [max_len]
        target_id = test_tar[sample_idx]

        # Get non-padding history
        history_items = [int(h) for h in history_ids if h != 0]

        print(f"\n{'─' * 80}")
        print(f"  Sample {idx_num + 1}/{args.num_samples}  (test index: {sample_idx})")
        print(f"{'─' * 80}")

        # Show history
        print(f"\n  📜 User History ({len(history_items)} games, most recent last):")
        for i, hid in enumerate(history_items):
            title = get_title(hid, id_to_item, token_to_title)
            marker = "  " if i < len(history_items) - 1 else "→ "
            print(f"    {marker}{i+1:2d}. {title}")

        # Ground truth
        target_title = get_title(int(target_id), id_to_item, token_to_title)
        print(f"\n  🎯 Ground Truth Next Game:")
        print(f"    ★  {target_title}")

        # Model prediction
        enc_lat = Z_latent_jnp[history_ids][None]  # [1, max_len, dim]
        enc_mask = jnp.array((history_ids != 0).astype(np.float32))[None]
        enc_out = encode(params, enc_lat, enc_mask)

        # Multi-seed
        enc_out_rep = jnp.repeat(enc_out, args.num_seeds, axis=0)
        enc_mask_rep = jnp.repeat(enc_mask, args.num_seeds, axis=0)
        noise_rng = jax.random.PRNGKey(sample_idx)
        z_T = jax.random.normal(noise_rng, (args.num_seeds, args.latent_dim))

        z_hat = denoise(params, enc_out_rep, enc_mask_rep, z_T, args.denoise_steps)

        # ANN retrieval
        z_hat_norm = z_hat / (jnp.linalg.norm(z_hat, axis=-1, keepdims=True) + 1e-8)
        scores = z_hat_norm @ Z_normed_jnp[1:].T  # [seeds, num_items]
        top1_items = np.array(jnp.argmax(scores, axis=-1)) + 1
        top1_scores = np.array(jnp.max(scores, axis=-1))

        # Sort by confidence, deduplicate
        sorted_idx = np.argsort(-top1_scores)
        seen = set()
        predictions = []
        for si in sorted_idx:
            item = int(top1_items[si])
            conf = float(top1_scores[si])
            if item not in seen:
                seen.add(item)
                predictions.append((item, conf))
            if len(predictions) >= args.top_k:
                break

        # Display predictions
        print(f"\n  🔮 Model Predictions (top-{args.top_k}, sorted by confidence):")
        hit_found = False
        for rank, (pred_id, conf) in enumerate(predictions):
            pred_title = get_title(pred_id, id_to_item, token_to_title)
            is_hit = pred_id == int(target_id)
            marker = "✅" if is_hit else "  "
            print(f"    {marker} {rank+1:2d}. [{conf:.4f}] {pred_title}")
            if is_hit:
                hit_found = True

        if not hit_found:
            print(f"    ❌ Ground truth not in top-{args.top_k}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
