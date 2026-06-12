"""Training and evaluation script for TIGER Flow (3-step flow matching) recommendation model.

Architecture overview (DiT-style two-stage design):
  Stage 1 — N-step denoising in latent space:
    Transformer denoises z_T (Gaussian noise) → z_0 (item semantic embedding),
    conditioned on user history via cross-attention, in N=3 Euler steps.
  Stage 2 — One-shot retrieval:
    ANN lookup maps denoised z_0 to nearest item ID (cosine similarity).

No discrete tokens, codebooks, or Semantic IDs are needed.
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
from models.tiger_flow import TIGERFlowModel
from evaluation.metrics import compute_ranks_from_predictions, calculate_metrics_from_ranks


def main():
    parser = argparse.ArgumentParser(
        description="TIGER Flow: 3-step flow matching for recommendation."
    )
    parser.add_argument("--checkpoint_dir", type=str, default="./data/tiger_flow_checkpoints")
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tb_log_dir", type=str, default="./data/tensorboard/tiger_flow_steam")
    parser.add_argument(
        "--z_anchor_path", type=str, default="./data/steam_t5_embeddings.npy",
        help="Path to frozen item content embeddings (e.g., Sentence-T5).",
    )
    parser.add_argument(
        "--dataset", type=str, default="steam",
        choices=["ml-1m", "beauty", "sports", "toys", "steam"],
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--attention_dim", type=int, default=384)
    parser.add_argument("--linear_dim", type=int, default=1024)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--num_seeds", type=int, default=20,
        help="Number of noise seeds for multi-sample ANN retrieval (analogous to beam size).",
    )
    parser.add_argument(
        "--denoise_steps", type=int, default=3,
        help="Number of Euler denoising steps (matching 3 tokens in TIGER).",
    )
    parser.add_argument(
        "--lambda_ce", type=float, default=10.0,
        help="Weight for contrastive CE loss relative to flow matching MSE loss.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.07,
        help="Temperature for cosine similarity logits in contrastive CE loss.",
    )
    args = parser.parse_args()

    dataset = args.dataset.lower()
    if args.checkpoint_dir == "./data/tiger_flow_checkpoints":
        args.checkpoint_dir = f"./data/tiger_flow_{dataset}_checkpoints"
    if args.tb_log_dir == "./data/tensorboard/tiger_flow_steam" and dataset != "steam":
        args.tb_log_dir = f"./data/tensorboard/tiger_flow_{dataset}"
    if args.z_anchor_path == "./data/steam_t5_embeddings.npy" and dataset != "steam":
        args.z_anchor_path = f"./data/{dataset}_t5_embeddings.npy"

    print(f"--- Training TIGER Flow on {dataset.upper()} ---")
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
    # 2. Load Frozen Embeddings & Compute PCA Projection
    # =========================================================================
    print(f"Loading frozen item embeddings from: {args.z_anchor_path}")
    Z_frozen = np.load(args.z_anchor_path).astype(np.float32)  # [num_items+1, 768]
    Z_frozen[0] = 0.0  # Ensure padding item has zero embedding

    print(f"Computing PCA projection (768 → {args.latent_dim})...")
    U, S, Vt = np.linalg.svd(Z_frozen, full_matrices=False)
    W_proj = Vt[:args.latent_dim, :].T  # [768, latent_dim]

    # Precompute all item latents in PCA space
    Z_latent_raw = Z_frozen @ W_proj  # [num_items+1, latent_dim]

    # Standardize to N(0,1) scale per dimension.
    # Sentence-T5 outputs L2-normalized embeddings (||z||=1), so PCA projections
    # have std ~0.02/dim — 50x smaller than ε ~ N(0,I). Without standardization,
    # v_target ≈ ε and the model only learns to predict noise, not item embeddings.
    # This is analogous to VAE's KL regularization in image diffusion models.
    z_mean = Z_latent_raw[1:].mean(axis=0)  # skip padding
    z_std = Z_latent_raw[1:].std(axis=0) + 1e-8
    Z_latent = np.zeros_like(Z_latent_raw)
    Z_latent[1:] = (Z_latent_raw[1:] - z_mean) / z_std  # standardized items
    Z_latent[0] = 0.0  # Padding stays zero
    print(f"Standardized Z_latent: mean≈{Z_latent[1:].mean():.4f}, std≈{Z_latent[1:].std():.4f}")

    # Precompute normalized latents for cosine similarity ANN retrieval
    # ANN is done in standardized space — cosine similarity is scale-invariant
    Z_norms = np.linalg.norm(Z_latent[1:], axis=-1, keepdims=True) + 1e-8
    Z_latent_normed = np.zeros_like(Z_latent)
    Z_latent_normed[1:] = Z_latent[1:] / Z_norms  # [num_items+1, latent_dim], skip pad

    variance_explained = np.sum(S[:args.latent_dim] ** 2) / np.sum(S ** 2)
    print(f"PCA variance explained: {variance_explained:.4f}")

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
    # 4. Initialize Model
    # =========================================================================
    print("Initializing TIGERFlowModel...")
    model = TIGERFlowModel(
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
    dummy_enc_lat = jnp.zeros((1, max_len, args.latent_dim))
    dummy_enc_mask = jnp.ones((1, max_len))
    dummy_z_t = jnp.zeros((1, args.latent_dim))
    dummy_t = jnp.zeros((1,))
    variables = model.init(key, dummy_enc_lat, dummy_enc_mask, dummy_z_t, dummy_t)
    params = variables["params"]

    num_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Model parameters: {num_params:,}")

    # =========================================================================
    # 5. Setup Optimizer
    # =========================================================================
    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    # =========================================================================
    # 6. Define JIT-compiled Training Step
    # =========================================================================
    lambda_ce = args.lambda_ce
    temperature = args.temperature

    @jax.jit
    def train_step(params, opt_state, encoder_latents, encoder_mask, z_target, target_ids, rng_key):
        """One training step with combined flow matching + contrastive CE loss.

        Loss = L_flow + λ_CE · L_CE where:
          L_flow = MSE on velocity prediction (denoising signal)
          L_CE   = softmax cross-entropy on predicted z_0 vs all items (ranking signal)

        The predicted z_0 is derived from single-step: z_0_hat = z_t - t · v_hat.
        This directly optimizes retrieval ranking (like HSTU's full-catalog softmax).
        """
        def loss_fn(p):
            noise_rng, t_rng, dropout_rng = jax.random.split(rng_key, 3)
            batch_size = z_target.shape[0]

            # Sample noise and timestep
            epsilon = jax.random.normal(noise_rng, z_target.shape)
            t = jax.random.uniform(t_rng, (batch_size,))  # [batch] in [0, 1]

            # Flow matching interpolation: z_t = (1-t) * z_0 + t * ε
            t_expand = t[:, None]  # [batch, 1]
            z_t = (1 - t_expand) * z_target + t_expand * epsilon

            # Target velocity: v = ε - z_0
            v_target = epsilon - z_target

            # Model predicts velocity
            v_hat = model.apply(
                {"params": p},
                encoder_latents, encoder_mask, z_t, t,
                rngs={"dropout": dropout_rng},
                deterministic=False,
            )

            # Loss 1: Flow matching MSE on velocity
            loss_flow = jnp.mean(jnp.sum((v_hat - v_target) ** 2, axis=-1))

            # Loss 2: Contrastive CE on predicted z_0 (same as HSTU/TIGER CE loss)
            # Derive z_0 from single-step: z_0 = z_t - t * v  (since v = (z_t - z_0)/t)
            z_0_hat = z_t - t_expand * v_hat
            z_0_hat_norm = z_0_hat / (jnp.linalg.norm(z_0_hat, axis=-1, keepdims=True) + 1e-8)
            # Cosine similarity logits against ALL items
            logits = z_0_hat_norm @ Z_items_normed_jnp.T / temperature  # [batch, num_items]
            labels = target_ids - 1  # items are 1-indexed, shift to 0-based
            loss_ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))

            total_loss = loss_flow + lambda_ce * loss_ce
            return total_loss, (loss_flow, loss_ce)

        (total_loss, (loss_flow, loss_ce)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, total_loss, loss_flow, loss_ce

    # =========================================================================
    # 7. Define JIT-compiled Inference Functions
    # =========================================================================
    @jax.jit
    def predict_enc(params, encoder_latents, encoder_mask):
        """Encode user history (called once per user)."""
        return model.apply(
            {"params": params},
            encoder_latents, encoder_mask,
            method=model.encode,
            deterministic=True,
        )

    @jax.jit
    def predict_velocity(params, enc_out, encoder_mask, z, t):
        """One denoising step: predict velocity at timestep t."""
        return model.apply(
            {"params": params},
            enc_out, encoder_mask, z, t,
            method=model.predict_velocity,
            deterministic=True,
        )

    def denoise_n_steps(params, enc_out, encoder_mask, z_T, num_steps):
        """N-step Euler integration from noise z_T to clean latent z_0.

        z_{t-dt} = z_t - dt * v_hat(z_t, t)  for t = 1, 1-dt, ..., dt
        """
        dt = 1.0 / num_steps
        z = z_T
        for step in range(num_steps):
            t_val = 1.0 - step * dt
            t = jnp.full((z.shape[0],), t_val)
            v_hat = predict_velocity(params, enc_out, encoder_mask, z, t)
            z = z - dt * v_hat
        return z

    # =========================================================================
    # 8. Evaluation Function (Multi-seed ANN Retrieval)
    # =========================================================================
    Z_latent_normed_jnp = jnp.array(Z_latent_normed)
    Z_items_normed_jnp = Z_latent_normed_jnp[1:]  # [num_items, latent_dim] exclude padding

    def evaluate_flow(params, eval_in, eval_tar, batch_size=None):
        """Evaluate with multi-seed 3-step denoising + ANN top-1 retrieval.

        For each user, sample num_seeds different noise vectors, denoise each
        in N steps, retrieve top-1 nearest item for each seed via cosine
        similarity. The num_seeds items form the prediction list.
        """
        if batch_size is None:
            batch_size = args.batch_size
        num_seeds = args.num_seeds
        num_steps = args.denoise_steps
        num_samples = len(eval_in)
        all_ranks = []
        total_unique = 0
        total_predictions = 0

        for i in range(0, num_samples, batch_size):
            batch_in = eval_in[i : i + batch_size]
            batch_tar = eval_tar[i : i + batch_size]
            actual_bs = len(batch_in)

            # Prepare encoder input: look up item latents
            enc_latents = jnp.array(Z_latent[batch_in])  # [bs, max_len, latent_dim]
            enc_mask = jnp.array((batch_in != 0).astype(np.float32))

            # Encode user history (once)
            enc_out = predict_enc(params, enc_latents, enc_mask)

            # Replicate encoder outputs for num_seeds parallel denoising runs
            enc_out_rep = jnp.repeat(enc_out, num_seeds, axis=0)  # [bs*seeds, ...]
            enc_mask_rep = jnp.repeat(enc_mask, num_seeds, axis=0)

            # Sample different noise for each (user, seed) pair
            noise_rng = jax.random.PRNGKey(i)
            z_T = jax.random.normal(noise_rng, (actual_bs * num_seeds, args.latent_dim))

            # N-step Euler denoising
            z_hat = denoise_n_steps(params, enc_out_rep, enc_mask_rep, z_T, num_steps)

            # ANN retrieval: cosine similarity → top-1 per seed
            z_hat_norm = z_hat / (jnp.linalg.norm(z_hat, axis=-1, keepdims=True) + 1e-8)
            # [bs*seeds, num_items+1]
            scores = z_hat_norm @ Z_latent_normed_jnp.T
            # Skip pad at index 0, get top-1 item and its confidence score
            top1_items = jnp.argmax(scores[:, 1:], axis=-1) + 1  # [bs*seeds]
            top1_scores = jnp.max(scores[:, 1:], axis=-1)  # [bs*seeds] confidence
            top1_items = np.array(top1_items).reshape(actual_bs, num_seeds)
            top1_scores = np.array(top1_scores).reshape(actual_bs, num_seeds)

            # Build prediction lists SORTED BY CONFIDENCE (highest cosine sim first)
            # This makes HR@1 = "did the most confident seed hit the target?"
            # Analogous to TIGER beam search sorting by cumulative log-probability.
            batch_predictions = []
            for j in range(actual_bs):
                # Sort seeds by descending confidence score
                sorted_indices = np.argsort(-top1_scores[j])
                preds = list(int(top1_items[j, idx]) for idx in sorted_indices)
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
    # 9. Resume / Eval-only Setup
    # =========================================================================
    writer = None
    if args.tb_log_dir and not args.eval_only:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        print(f"TensorBoard logging enabled: {args.tb_log_dir}")

    start_epoch = 1
    best_val_ndcg = -1.0
    best_params = None
    best_val_metrics = {}
    patience_counter = 0

    if args.resume_path:
        print(f"Loading checkpoint from {args.resume_path}...")
        state_template = {
            "params": params,
            "opt_state": opt_state,
            "epoch": 0,
            "best_val_ndcg": 0.0,
        }
        with open(args.resume_path, "rb") as f:
            ckpt = flax.serialization.from_bytes(state_template, f.read())
        params = ckpt["params"]
        opt_state = ckpt["opt_state"]
        start_epoch = ckpt["epoch"] + 1
        best_val_ndcg = float(ckpt["best_val_ndcg"])
        best_params = params
        best_val_metrics["NDCG@10"] = best_val_ndcg
        print(f"Resumed from epoch {ckpt['epoch']}, best NDCG@10 = {best_val_ndcg:.5f}")

    if args.eval_only:
        if not args.resume_path:
            raise ValueError("Must specify --resume_path for --eval_only.")
        print("\nRunning test evaluation only...")
        test_results = evaluate_flow(best_params, test_in, test_tar)
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
    global_step = (start_epoch - 1) * num_batches

    print(f"\nTraining for {epochs} epochs starting from epoch {start_epoch}...")
    print(f"Denoise steps: {args.denoise_steps}, Noise seeds: {args.num_seeds}")
    for epoch in range(start_epoch, epochs + 1):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_in = train_in[indices]
        shuffled_tar = train_tar[indices]

        epoch_loss = 0.0
        epoch_loss_flow = 0.0
        epoch_loss_ce = 0.0
        num_batches_processed = 0
        start_time = time.time()

        for i in range(0, num_samples, batch_size):
            if i + batch_size > num_samples:
                break

            batch_in_ids = shuffled_in[i : i + batch_size]  # [bs, max_len]
            batch_tar_ids = shuffled_tar[i : i + batch_size]  # [bs]

            # Look up precomputed item latents (on-the-fly, avoids 4GB precomputation)
            batch_enc_latents = jnp.array(Z_latent[batch_in_ids])  # [bs, max_len, latent_dim]
            batch_enc_mask = jnp.array((batch_in_ids != 0).astype(np.float32))
            batch_z_target = jnp.array(Z_latent[batch_tar_ids])  # [bs, latent_dim]
            batch_tar_ids_jnp = jnp.array(batch_tar_ids)  # [bs] item IDs for CE loss

            epoch_rng, step_rng = jax.random.split(epoch_rng)
            params, opt_state, total_loss, loss_flow, loss_ce = train_step(
                params, opt_state,
                batch_enc_latents, batch_enc_mask, batch_z_target, batch_tar_ids_jnp,
                step_rng,
            )

            epoch_loss += total_loss
            epoch_loss_flow += loss_flow
            epoch_loss_ce += loss_ce
            num_batches_processed += 1
            global_step += 1

            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("Loss/total", float(total_loss), global_step)
                writer.add_scalar("Loss/flow_matching", float(loss_flow), global_step)
                writer.add_scalar("Loss/contrastive_ce", float(loss_ce), global_step)

        elapsed = time.time() - start_time
        n = max(num_batches_processed, 1)
        avg_total = float(epoch_loss) / n
        avg_flow = float(epoch_loss_flow) / n
        avg_ce = float(epoch_loss_ce) / n
        print(
            f"Epoch {epoch:02d}/{epochs} | Total: {avg_total:.2f} "
            f"(Flow: {avg_flow:.2f} + {args.lambda_ce}×CE: {avg_ce:.4f}) | Time: {elapsed:.2f}s"
        )

        if writer is not None:
            writer.add_scalar("Loss/train_epoch", avg_total, global_step)
            writer.add_scalar("Loss/train_flow_epoch", avg_flow, global_step)
            writer.add_scalar("Loss/train_ce_epoch", avg_ce, global_step)

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

        # Check improvement on any metric (excluding diversity)
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
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)."
                )
                break

        # Save latest checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        latest_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.msgpack")
        ckpt_state = {
            "params": params,
            "opt_state": opt_state,
            "epoch": epoch,
            "best_val_ndcg": best_val_ndcg,
        }
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
        print("TensorBoard writer closed.")

    # =========================================================================
    # 12. Log Results
    # =========================================================================
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = "experiment_results.md"
    results_row = (
        f"| {date_str} | TIGER Flow ({args.denoise_steps}-step, "
        f"seeds={args.num_seeds}, embed={args.embedding_dim}) on "
        f"{dataset.upper()} | Local (GeForce RTX 4080) | "
        f"{test_results['HR@5']:.5f} | {test_results['NDCG@5']:.5f} | "
        f"{test_results['HR@10']:.5f} | {test_results['NDCG@10']:.5f} | "
        f"{test_results['HR@20']:.5f} | {test_results['NDCG@20']:.5f} | "
        f"{test_results['MRR']:.5f} | "
        f"Flow matching + ANN retrieval "
        f"(Best Val NDCG@10={best_val_ndcg:.5f}) |"
    )

    with open(log_path, "a") as f:
        f.write(results_row + "\n")
    print(f"\nResults written to {log_path}")


if __name__ == "__main__":
    main()
