"""Training script for TIGER Flow with VAE (joint training).

Architecture (complete DiT-style three-component system):
  1. VAE Encoder: Sentence-T5 embedding → z₀ ~ N(0, I)  (learned, KL-regularized)
  2. Transformer:  z_T → z₀ via N-step denoising         (conditioned on user history)
  3. VAE Decoder:  z₀ → reconstructed embedding           (preserves item identity)

Loss = L_flow(θ,φ) + λ_kl · L_KL(φ) + λ_rec · L_rec(φ,ψ)

The KL regularization ensures the latent space matches N(0,I), solving the
scale mismatch that prevented pure flow matching from working with frozen
Sentence-T5 embeddings (which are L2-normalized and clustered at cos_sim ~0.84).
"""

import argparse
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.serialization

from datasets import MovieLensDataLoader, AmazonDataLoader, SteamDataLoader
from models.tiger_flow import TIGERFlowModel, ItemVAEEncoder, ItemVAEDecoder, sinusoidal_embedding
from evaluation.metrics import compute_ranks_from_predictions, calculate_metrics_from_ranks


def main():
    parser = argparse.ArgumentParser(
        description="TIGER Flow + VAE: DiT-style joint training for recommendation."
    )
    parser.add_argument("--checkpoint_dir", type=str, default="./data/tiger_flow_vae_checkpoints")
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tb_log_dir", type=str, default="./data/tensorboard/tiger_flow_vae_steam")
    parser.add_argument(
        "--z_anchor_path", type=str, default="./data/steam_t5_embeddings.npy",
        help="Path to frozen item content embeddings (Sentence-T5).",
    )
    parser.add_argument(
        "--dataset", type=str, default="steam",
        choices=["ml-1m", "beauty", "sports", "toys", "steam"],
    )
    parser.add_argument("--patience", type=int, default=5)
    # Transformer dims
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--attention_dim", type=int, default=384)
    parser.add_argument("--linear_dim", type=int, default=1024)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    # Training
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--denoise_steps", type=int, default=3)
    # VAE loss weights
    parser.add_argument("--lambda_kl", type=float, default=0.1,
        help="Weight for KL divergence regularization.")
    parser.add_argument("--lambda_rec", type=float, default=1.0,
        help="Weight for reconstruction loss.")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    if args.checkpoint_dir == "./data/tiger_flow_vae_checkpoints":
        args.checkpoint_dir = f"./data/tiger_flow_vae_{dataset}_checkpoints"
    if args.tb_log_dir == "./data/tensorboard/tiger_flow_vae_steam" and dataset != "steam":
        args.tb_log_dir = f"./data/tensorboard/tiger_flow_vae_{dataset}"
    if args.z_anchor_path == "./data/steam_t5_embeddings.npy" and dataset != "steam":
        args.z_anchor_path = f"./data/{dataset}_t5_embeddings.npy"

    print(f"--- Training TIGER Flow + VAE on {dataset.upper()} ---")
    print("Device list:", jax.devices())

    # =========================================================================
    # 1. Load Dataset
    # =========================================================================
    data_dir = "./data"
    if dataset == "ml-1m":
        loader = MovieLensDataLoader(dataset_name="ml-1m", data_dir=data_dir, min_rating=0)
    elif dataset in ["beauty", "sports", "toys"]:
        loader = AmazonDataLoader(category=dataset, data_dir=data_dir, min_rating=0)
    elif dataset == "steam":
        loader = SteamDataLoader(data_dir=data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    print(f"Dataset stats: Users = {loader.num_users}, Items = {loader.num_items}")

    max_len = 20 if dataset in ["beauty", "sports", "toys", "steam"] else 50

    # =========================================================================
    # 2. Load Frozen Embeddings (NO PCA — VAE learns the projection)
    # =========================================================================
    print(f"Loading frozen item embeddings from: {args.z_anchor_path}")
    Z_frozen = np.load(args.z_anchor_path).astype(np.float32)  # [num_items+1, 768]
    Z_frozen[0] = 0.0  # Padding item
    embed_dim_raw = Z_frozen.shape[1]
    print(f"Frozen embeddings: {Z_frozen.shape} (dim={embed_dim_raw})")

    Z_frozen_jnp = jnp.array(Z_frozen)

    # =========================================================================
    # 3. Prepare Train / Val / Test Splits
    # =========================================================================
    print("Preparing data splits...")
    train_dataset = loader.get_split("train", max_len=max_len, format_type="index")
    train_in, train_tar = train_dataset.to_numpy()
    print(f"Train: {len(train_tar)} samples")

    val_dataset = loader.get_split("val", max_len=max_len, format_type="index")
    val_in, val_tar = val_dataset.to_numpy()

    test_dataset = loader.get_split("test", max_len=max_len, format_type="index")
    test_in, test_tar = test_dataset.to_numpy()

    # =========================================================================
    # 4. Initialize Models (VAE Encoder + Transformer + VAE Decoder)
    # =========================================================================
    print("Initializing models...")

    vae_enc = ItemVAEEncoder(latent_dim=args.latent_dim)
    vae_dec = ItemVAEDecoder(output_dim=embed_dim_raw)
    transformer = TIGERFlowModel(
        embedding_dim=args.embedding_dim,
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        max_encoder_len=max_len,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
    )

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # Init VAE encoder
    dummy_e = jnp.zeros((1, embed_dim_raw))
    vae_enc_vars = vae_enc.init(k1, dummy_e)
    vae_enc_params = vae_enc_vars["params"]

    # Init VAE decoder
    dummy_z = jnp.zeros((1, args.latent_dim))
    vae_dec_vars = vae_dec.init(k2, dummy_z)
    vae_dec_params = vae_dec_vars["params"]

    # Init Transformer (encoder input is raw 768-d, NOT latent_dim)
    dummy_enc_lat = jnp.zeros((1, max_len, embed_dim_raw))
    dummy_enc_mask = jnp.ones((1, max_len))
    dummy_z_t = jnp.zeros((1, args.latent_dim))
    dummy_t = jnp.zeros((1,))
    transformer_vars = transformer.init(k3, dummy_enc_lat, dummy_enc_mask, dummy_z_t, dummy_t)
    transformer_params = transformer_vars["params"]

    # Combine all params
    params = {
        "vae_enc": vae_enc_params,
        "vae_dec": vae_dec_params,
        "transformer": transformer_params,
    }

    num_params = sum(p.size for p in jax.tree.leaves(params))
    num_vae = sum(p.size for p in jax.tree.leaves(vae_enc_params)) + \
              sum(p.size for p in jax.tree.leaves(vae_dec_params))
    num_tf = sum(p.size for p in jax.tree.leaves(transformer_params))
    print(f"Total parameters: {num_params:,} (VAE: {num_vae:,}, Transformer: {num_tf:,})")

    # =========================================================================
    # 5. Setup Optimizer
    # =========================================================================
    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    # =========================================================================
    # 6. Define JIT-compiled Training Step
    # =========================================================================
    lambda_kl = args.lambda_kl
    lambda_rec = args.lambda_rec

    @jax.jit
    def train_step(params, opt_state, batch_frozen_in, batch_mask, batch_frozen_tar, rng_key):
        """Joint training step: VAE + Flow Matching.

        Loss = L_flow + λ_kl·L_KL + λ_rec·L_rec

        Args:
            batch_frozen_in: [bs, max_len, 768] frozen T5 embeddings for history items
            batch_mask: [bs, max_len] attention mask
            batch_frozen_tar: [bs, 768] frozen T5 embedding of target item
        """
        def loss_fn(p):
            noise_rng, t_rng, reparam_rng, dropout_rng = jax.random.split(rng_key, 4)
            batch_size = batch_frozen_tar.shape[0]

            # --- VAE Encode target item (with reparameterization) ---
            mu_tar, log_var_tar = vae_enc.apply({"params": p["vae_enc"]}, batch_frozen_tar)
            std_tar = jnp.exp(0.5 * log_var_tar)
            eps_reparam = jax.random.normal(reparam_rng, mu_tar.shape)
            z_target = mu_tar + std_tar * eps_reparam  # [bs, latent_dim]

            # --- History items stay in raw 768-d (Transformer input_projection handles it) ---
            enc_latents = batch_frozen_in  # [bs, max_len, 768]

            # --- Flow matching on z_target ---
            epsilon = jax.random.normal(noise_rng, z_target.shape)
            t = jax.random.uniform(t_rng, (batch_size,))
            t_expand = t[:, None]
            z_t = (1 - t_expand) * z_target + t_expand * epsilon
            v_target = epsilon - z_target

            v_hat = transformer.apply(
                {"params": p["transformer"]},
                enc_latents, batch_mask, z_t, t,
                rngs={"dropout": dropout_rng},
                deterministic=False,
            )

            loss_flow = jnp.mean(jnp.sum((v_hat - v_target) ** 2, axis=-1))

            # --- KL divergence: D_KL(q(z|e) || N(0,I)) ---
            loss_kl = -0.5 * jnp.mean(jnp.sum(
                1 + log_var_tar - mu_tar ** 2 - jnp.exp(log_var_tar), axis=-1
            ))

            # --- Reconstruction: decode z_target → ê, compare with original ---
            e_hat = vae_dec.apply({"params": p["vae_dec"]}, z_target)
            loss_rec = jnp.mean(jnp.sum((e_hat - batch_frozen_tar) ** 2, axis=-1))

            total_loss = loss_flow + lambda_kl * loss_kl + lambda_rec * loss_rec
            return total_loss, (loss_flow, loss_kl, loss_rec)

        (total_loss, (loss_flow, loss_kl, loss_rec)), grads = \
            jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, total_loss, loss_flow, loss_kl, loss_rec

    # =========================================================================
    # 7. Define JIT-compiled Inference Functions
    # =========================================================================
    @jax.jit
    def vae_encode_mu(params, embeddings):
        """Encode items through VAE, return μ only (deterministic)."""
        mu, _ = vae_enc.apply({"params": params["vae_enc"]}, embeddings)
        return mu

    @jax.jit
    def predict_enc(params, encoder_latents, encoder_mask):
        """Encode user history through Transformer."""
        return transformer.apply(
            {"params": params["transformer"]},
            encoder_latents, encoder_mask,
            method=transformer.encode,
            deterministic=True,
        )

    @jax.jit
    def predict_velocity(params, enc_out, encoder_mask, z, t):
        """One denoising step."""
        return transformer.apply(
            {"params": params["transformer"]},
            enc_out, encoder_mask, z, t,
            method=transformer.predict_velocity,
            deterministic=True,
        )

    def denoise_n_steps(params, enc_out, encoder_mask, z_T, num_steps):
        """N-step Euler integration from noise z_T to clean latent z_0."""
        dt = 1.0 / num_steps
        z = z_T
        for step in range(num_steps):
            t_val = 1.0 - step * dt
            t = jnp.full((z.shape[0],), t_val)
            v_hat = predict_velocity(params, enc_out, encoder_mask, z, t)
            z = z - dt * v_hat
        return z

    # =========================================================================
    # 8. Evaluation Function
    # =========================================================================
    def evaluate_flow(params, eval_in, eval_tar, batch_size=None):
        """Evaluate: VAE encode all items → multi-seed denoise → ANN retrieval."""
        if batch_size is None:
            batch_size = args.batch_size
        num_seeds = args.num_seeds
        num_steps = args.denoise_steps
        num_samples = len(eval_in)

        # Precompute all item latents via VAE encoder (μ only)
        all_item_latents = []
        for i in range(0, Z_frozen.shape[0], 1024):
            batch_e = Z_frozen_jnp[i : i + 1024]
            mu = vae_encode_mu(params, batch_e)
            all_item_latents.append(mu)
        Z_vae_latent = jnp.concatenate(all_item_latents, axis=0)  # [num_items+1, latent_dim]

        # Precompute ||z_item||² for efficient L2 distance
        Z_items = Z_vae_latent[1:]  # [num_items, latent_dim]
        Z_items_sq = jnp.sum(Z_items ** 2, axis=-1)  # [num_items]

        all_ranks = []
        total_unique = 0
        total_predictions = 0

        for i in range(0, num_samples, batch_size):
            batch_in = eval_in[i : i + batch_size]
            batch_tar = eval_tar[i : i + batch_size]
            actual_bs = len(batch_in)

            # History items stay in raw 768-d (no VAE)
            enc_latents = Z_frozen_jnp[batch_in]  # [bs, max_len, 768]
            enc_mask = jnp.array((batch_in != 0).astype(np.float32))

            # Transformer encode
            enc_out = predict_enc(params, enc_latents, enc_mask)

            # Multi-seed denoising
            enc_out_rep = jnp.repeat(enc_out, num_seeds, axis=0)
            enc_mask_rep = jnp.repeat(enc_mask, num_seeds, axis=0)
            noise_rng = jax.random.PRNGKey(i)
            z_T = jax.random.normal(noise_rng, (actual_bs * num_seeds, args.latent_dim))

            z_hat = denoise_n_steps(params, enc_out_rep, enc_mask_rep, z_T, num_steps)

            # ANN retrieval using L2 distance (aligned with MSE training)
            # ||z_hat - z_item||² = ||z_hat||² + ||z_item||² - 2·z_hat·z_item
            z_hat_sq = jnp.sum(z_hat ** 2, axis=-1, keepdims=True)  # [bs*seeds, 1]
            dot = z_hat @ Z_items.T  # [bs*seeds, num_items]
            dists = z_hat_sq + Z_items_sq[None, :] - 2 * dot  # [bs*seeds, num_items]

            top1_items = jnp.argmin(dists, axis=-1) + 1  # +1 for 1-based indexing
            top1_dists = jnp.min(dists, axis=-1)  # lower = more confident
            top1_items = np.array(top1_items).reshape(actual_bs, num_seeds)
            top1_dists = np.array(top1_dists).reshape(actual_bs, num_seeds)

            # Sort by confidence (ascending distance = descending confidence)
            batch_predictions = []
            for j in range(actual_bs):
                sorted_idx = np.argsort(top1_dists[j])  # ascending: closest first
                preds = [int(top1_items[j, idx]) for idx in sorted_idx]
                batch_predictions.append(preds)
                total_unique += len(set(preds))
                total_predictions += len(preds)

            batch_ranks = compute_ranks_from_predictions(batch_predictions, batch_tar)
            all_ranks.extend(batch_ranks)

        ranks = np.array(all_ranks)
        results = calculate_metrics_from_ranks(ranks, k_list=[1, 5, 10, 20])
        results["Diversity"] = total_unique / max(total_predictions, 1)
        return results

    # =========================================================================
    # 9. Setup
    # =========================================================================
    writer = None
    if args.tb_log_dir and not args.eval_only:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        print(f"TensorBoard logging: {args.tb_log_dir}")

    start_epoch = 1
    best_val_ndcg = -1.0
    best_params = None
    best_val_metrics = {}
    patience_counter = 0

    if args.eval_only:
        if not args.resume_path:
            raise ValueError("Must specify --resume_path for --eval_only.")
        print("\nRunning test evaluation only...")
        test_results = evaluate_flow(params, test_in, test_tar)
        print("\n--- Test Results ---")
        for metric, score in test_results.items():
            print(f"{metric}: {score:.5f}")
        return

    # =========================================================================
    # 10. Training Loop
    # =========================================================================
    epochs = args.epochs
    batch_size = args.batch_size
    num_samples = len(train_tar)
    epoch_rng = jax.random.PRNGKey(777)
    num_batches = num_samples // batch_size
    global_step = 0

    print(f"\nTraining for {epochs} epochs...")
    print(f"Denoise steps: {args.denoise_steps}, Seeds: {args.num_seeds}")
    print(f"Loss weights: λ_kl={args.lambda_kl}, λ_rec={args.lambda_rec}")

    for epoch in range(start_epoch, epochs + 1):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_in = train_in[indices]
        shuffled_tar = train_tar[indices]

        epoch_total = 0.0
        epoch_flow = 0.0
        epoch_kl = 0.0
        epoch_rec = 0.0
        num_batches_processed = 0
        start_time = time.time()

        for i in range(0, num_samples, batch_size):
            if i + batch_size > num_samples:
                break

            batch_in_ids = shuffled_in[i : i + batch_size]
            batch_tar_ids = shuffled_tar[i : i + batch_size]

            # Look up frozen embeddings
            batch_frozen_in = jnp.array(Z_frozen[batch_in_ids])    # [bs, max_len, 768]
            batch_mask = jnp.array((batch_in_ids != 0).astype(np.float32))
            batch_frozen_tar = jnp.array(Z_frozen[batch_tar_ids])  # [bs, 768]

            epoch_rng, step_rng = jax.random.split(epoch_rng)
            params, opt_state, total_loss, loss_flow, loss_kl, loss_rec = train_step(
                params, opt_state,
                batch_frozen_in, batch_mask, batch_frozen_tar,
                step_rng,
            )

            epoch_total += total_loss
            epoch_flow += loss_flow
            epoch_kl += loss_kl
            epoch_rec += loss_rec
            num_batches_processed += 1
            global_step += 1

            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("Loss/total", float(total_loss), global_step)
                writer.add_scalar("Loss/flow", float(loss_flow), global_step)
                writer.add_scalar("Loss/kl", float(loss_kl), global_step)
                writer.add_scalar("Loss/rec", float(loss_rec), global_step)

        elapsed = time.time() - start_time
        n = max(num_batches_processed, 1)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Flow: {float(epoch_flow)/n:.2f}  KL: {float(epoch_kl)/n:.2f}  "
            f"Rec: {float(epoch_rec)/n:.2f}  Total: {float(epoch_total)/n:.2f} | "
            f"Time: {elapsed:.2f}s"
        )

        if writer is not None:
            writer.add_scalar("Loss/epoch_total", float(epoch_total) / n, global_step)

        # --- Validation ---
        print(f"Evaluating validation split at epoch {epoch}...")
        val_results = evaluate_flow(params, val_in, val_tar)
        val_ndcg = val_results["NDCG@10"]
        val_hr = val_results["HR@10"]
        val_mrr = val_results["MRR"]
        val_div = val_results["Diversity"]
        print(
            f"--- Val @ Epoch {epoch} | NDCG@10: {val_ndcg:.5f} | "
            f"HR@10: {val_hr:.5f} | MRR: {val_mrr:.5f} | Diversity: {val_div:.3f}"
        )

        if writer is not None:
            for metric, score in val_results.items():
                writer.add_scalar(f"Val/{metric}", score, global_step)

        # Check improvement
        improved = False
        for metric, score in val_results.items():
            if metric == "Diversity":
                continue
            if metric not in best_val_metrics or score > best_val_metrics[metric]:
                best_val_metrics[metric] = score
                improved = True

        if improved:
            best_val_ndcg = best_val_metrics.get("NDCG@10", best_val_ndcg)
            best_params = params
            patience_counter = 0
            print(">>> New best! Saving checkpoint...")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(args.checkpoint_dir, "best_checkpoint.msgpack")
            ckpt_state = {
                "params": params,
                "opt_state": opt_state,
                "epoch": epoch,
                "best_val_ndcg": best_val_ndcg,
            }
            with open(ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(ckpt_state))
            print(f"Checkpoint saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

        # Save latest
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        latest_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.msgpack")
        ckpt_state = {"params": params, "opt_state": opt_state, "epoch": epoch, "best_val_ndcg": best_val_ndcg}
        with open(latest_path, "wb") as f:
            f.write(flax.serialization.to_bytes(ckpt_state))

    # =========================================================================
    # 11. Final Test Evaluation
    # =========================================================================
    if best_params is None:
        best_params = params

    print("\nRunning final test evaluation...")
    test_results = evaluate_flow(best_params, test_in, test_tar)

    print("\n--- Final Test Results ---")
    for metric, score in test_results.items():
        print(f"{metric}: {score:.5f}")

    if writer is not None:
        for metric, score in test_results.items():
            writer.add_scalar(f"Test/{metric}", score, global_step)
        writer.close()

    # Log results
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    results_row = (
        f"| {date_str} | TIGER Flow+VAE ({args.denoise_steps}-step, "
        f"seeds={args.num_seeds}, kl={args.lambda_kl}, rec={args.lambda_rec}) on "
        f"{dataset.upper()} | Local (GeForce RTX 4080) | "
        f"{test_results['HR@5']:.5f} | {test_results['NDCG@5']:.5f} | "
        f"{test_results['HR@10']:.5f} | {test_results['NDCG@10']:.5f} | "
        f"{test_results['HR@20']:.5f} | {test_results['NDCG@20']:.5f} | "
        f"{test_results['MRR']:.5f} | "
        f"VAE+Flow (Best Val NDCG@10={best_val_ndcg:.5f}) |"
    )
    with open("experiment_results.md", "a") as f:
        f.write(results_row + "\n")
    print(f"\nResults written to experiment_results.md")


if __name__ == "__main__":
    main()
