"""HSTU + Flow + CE: End-to-end joint training with shared output.

Architecture:
  - Embedding table: TRAINABLE, updated by CE loss on flow output
  - HSTU backbone: TRAINABLE, updated by Flow loss
  - FlowHead: TRAINABLE, updated by both Flow + CE loss

Shared output design:
  v_hat = FlowHead(h_user, z_t, t)      → velocity prediction
  z_0_hat = z_t - t * v_hat             → one-step denoised embedding

  L_flow = MSE(v_hat, v_target)         → teaches denoising dynamics
  L_ce = CE(z_0_hat @ emb.T, target)    → supervises output quality

Both losses share the same output z_0_hat.
"""

import argparse
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
import flax.serialization

from datasets import MovieLensDataLoader, AmazonDataLoader, SteamDataLoader
from models.hstu import HSTUBlock
from models.tiger_flow import FlowHead, sinusoidal_embedding
from evaluation.metrics import compute_ranks_from_predictions, calculate_metrics_from_ranks
from models.hstu_flow import HSTUFlowCEModel, VAEEncoder, VAEDecoder


def main():
    parser = argparse.ArgumentParser(description="HSTU + Flow + CE: E2E Training")
    parser.add_argument("--checkpoint_dir", type=str, default="./data/hstu_flow_ce_steam_checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tb_log_dir", type=str, default="./data/tensorboard/hstu_flow_ce_steam")
    parser.add_argument("--dataset", type=str, default="steam",
        choices=["ml-1m", "beauty", "sports", "toys", "steam"])
    parser.add_argument("--patience", type=int, default=5)
    # Architecture
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--linear_dim", type=int, default=512)
    parser.add_argument("--flow_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--lambda_ce", type=float, default=1.0,
        help="Weight for CE loss.")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    max_len = 20 if dataset in ["beauty", "sports", "toys", "steam"] else 50

    print(f"--- HSTU + Flow + CE (E2E) on {dataset.upper()} ---")
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
    num_items = loader.num_items
    print(f"Dataset: Users={loader.num_users}, Items={num_items}")

    # =========================================================================
    # 2. Prepare Data
    # =========================================================================
    train_dataset = loader.get_split("train", max_len=max_len, format_type="index")
    train_in, train_tar = train_dataset.to_numpy()
    print(f"Train: {len(train_tar)} samples")

    val_dataset = loader.get_split("val", max_len=max_len, format_type="index")
    val_in, val_tar = val_dataset.to_numpy()

    test_dataset = loader.get_split("test", max_len=max_len, format_type="index")
    test_in, test_tar = test_dataset.to_numpy()

    # =========================================================================
    # 3. Initialize Model
    # =========================================================================
    model = HSTUFlowCEModel(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        flow_hidden_dim=args.flow_hidden_dim,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
        max_sequence_len=max_len,
    )

    key, enc_key, dec_key = jax.random.split(jax.random.PRNGKey(42), 3)
    dummy_seq = jnp.zeros((1, max_len), dtype=jnp.int32)
    dummy_z = jnp.zeros((1, args.embedding_dim))
    dummy_t = jnp.zeros((1,))
    model_vars = model.init(key, dummy_seq, dummy_z, dummy_t)
    params = model_vars["params"]

    vae_enc = VAEEncoder(latent_dim=args.embedding_dim)
    vae_dec = VAEDecoder(output_dim=args.embedding_dim)
    enc_vars = vae_enc.init(enc_key, jnp.zeros((1, args.embedding_dim)))
    dec_vars = vae_dec.init(dec_key, jnp.zeros((1, args.embedding_dim)))

    # Merge VAE params into the main params dict
    params = flax.core.unfreeze(params)
    params["vae_enc"] = enc_vars["params"]
    params["vae_dec"] = dec_vars["params"]
    params = flax.core.freeze(params)

    num_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Total parameters: {num_params:,}")

    # =========================================================================
    # 4. Optimizer
    # =========================================================================
    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    # =========================================================================
    # 5. Training Step
    # =========================================================================
    @jax.jit
    def train_step(params, opt_state, batch_in, batch_tar, rng_key):
        """Joint training: Flow in VAE latent space + CE in orig space."""
        def loss_fn(p):
            noise_rng, t_rng, dropout_rng, vae_rng = jax.random.split(rng_key, 4)
            batch_size = batch_tar.shape[0]

            # 1. Get embedding table
            v_hat_dummy, emb_table = model.apply(
                {"params": p}, batch_in,
                jnp.zeros((batch_size, args.embedding_dim)),
                jnp.zeros((batch_size,)),
                rngs={"dropout": dropout_rng},
                deterministic=False,
            )

            # 2. VAE Encoding for target items
            emb_target = emb_table[batch_tar]
            mu, logvar = vae_enc.apply({"params": p["vae_enc"]}, emb_target)
            
            # Reparameterization trick
            std = jnp.exp(0.5 * logvar)
            eps_vae = jax.random.normal(vae_rng, std.shape)
            z_latent_target = mu + eps_vae * std
            
            # VAE KL Loss (target N(0, I))
            kl_loss = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1)
            kl_loss = jnp.mean(kl_loss)

            # 3. Flow setup (in latent space)
            # Stop gradient on target for flow matching.
            z_latent_target_sg = jax.lax.stop_gradient(z_latent_target)
            
            epsilon = jax.random.normal(noise_rng, z_latent_target_sg.shape)
            t = jax.random.uniform(t_rng, (batch_size,), minval=0.01, maxval=1.0)
            t_expand = t[:, None]
            
            z_t = (1 - t_expand) * z_latent_target_sg + t_expand * epsilon
            v_target = epsilon - z_latent_target_sg

            # 4. Predict velocity
            v_hat, _ = model.apply(
                {"params": p}, batch_in, z_t, t,
                rngs={"dropout": dropout_rng},
                deterministic=False,
            )
            
            # Flow Loss
            flow_loss = jnp.mean(jnp.sum((v_hat - v_target) ** 2, axis=-1))

            # 5. CE Loss (VAE reconstruction -> Decode -> CE)
            e_pred_vae = vae_dec.apply({"params": p["vae_dec"]}, z_latent_target)
            ce_logits_vae = jnp.dot(e_pred_vae, emb_table.T)
            ce_logits_vae = ce_logits_vae.at[:, 0].set(-1e9)
            ce_loss_vae = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(ce_logits_vae, batch_tar)
            )

            # Weighting: VAE KL loss needs a small weight to avoid collapsing the embeddings.
            total_loss = flow_loss + args.lambda_ce * ce_loss_vae + 0.1 * kl_loss
            return total_loss, (flow_loss, ce_loss_vae, kl_loss)

        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        flow_loss, ce_loss_vae, kl_loss = aux
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, total_loss, flow_loss, ce_loss_vae, kl_loss

    # =========================================================================
    # 6. Inference
    # =========================================================================
    # For evaluation, we can use Flow denoising

    # Create a standalone HSTU encoder for extracting h_user
    class _HSTUEnc(nn.Module):
        num_items: int
        embedding_dim: int
        num_blocks: int
        num_heads: int
        attention_dim: int
        linear_dim: int
        attn_dropout_rate: float
        linear_dropout_rate: float
        max_sequence_len: int

        @nn.compact
        def __call__(self, item_seq, deterministic=True):
            embed_layer = nn.Embed(
                num_embeddings=self.num_items + 1,
                features=self.embedding_dim,
                name="item_embedding",
            )
            x = embed_layer(item_seq)
            for i in range(self.num_blocks):
                x = HSTUBlock(
                    attention_dim=self.attention_dim,
                    linear_dim=self.linear_dim,
                    num_heads=self.num_heads,
                    attn_dropout_rate=self.attn_dropout_rate,
                    linear_dropout_rate=self.linear_dropout_rate,
                    enable_relative_attention_bias=True,
                    max_sequence_len=self.max_sequence_len,
                    name=f"hstu_block_{i}",
                )(x, deterministic=deterministic)
            return x[:, -1, :]  # h_user

    hstu_enc = _HSTUEnc(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
        max_sequence_len=max_len,
    )
    _enc_vars = hstu_enc.init(key, dummy_seq)

    @jax.jit
    def get_h_user(params, batch_in):
        # Extract embedding + HSTU block params (same names as in HSTUFlowCEModel)
        enc_params = {k: v for k, v in params.items()
                      if k.startswith("hstu_block_") or k == "item_embedding"}
        return hstu_enc.apply({"params": enc_params}, batch_in, deterministic=True)

    flow_head_module = FlowHead(hidden_dim=args.flow_hidden_dim, output_dim=args.embedding_dim)

    @jax.jit
    def predict_velocity(params, h_user, z, t):
        return flow_head_module.apply({"params": params["flow_head"]}, h_user, z, t)

    def denoise_n_steps(params, h_user, z_T, num_steps):
        dt = 1.0 / num_steps
        z = z_T
        for step in range(num_steps):
            t_val = 1.0 - step * dt
            t = jnp.full((z.shape[0],), t_val)
            v_hat = predict_velocity(params, h_user, z, t)
            z = z - dt * v_hat
        return z

    # =========================================================================
    # 7. Evaluation
    # =========================================================================
    def evaluate_flow(params, eval_in, eval_tar, num_steps, batch_size=None):
        """Evaluate using Flow denoising."""
        if batch_size is None:
            batch_size = args.batch_size
        num_seeds = args.num_seeds
        num_samples = len(eval_in)
        emb_table = params["item_embedding"]["embedding"]
        item_sq = jnp.sum(emb_table[1:] ** 2, axis=-1)

        all_ranks = []
        total_unique = 0
        total_predictions = 0

        for i in range(0, num_samples, batch_size):
            batch_in = eval_in[i:i+batch_size]
            batch_tar = eval_tar[i:i+batch_size]
            actual_bs = len(batch_tar)

            h_user = get_h_user(params, jnp.array(batch_in))

            h_user_rep = jnp.repeat(h_user, num_seeds, axis=0)
            noise_rng = jax.random.PRNGKey(i)
            z_T = jax.random.normal(noise_rng, (actual_bs * num_seeds, args.embedding_dim))
            z_hat_latent = denoise_n_steps(params, h_user_rep, z_T, num_steps)
            
            # Decode to original embedding space
            z_hat = vae_dec.apply({"params": params["vae_dec"]}, z_hat_latent)

            # L2 ANN
            z_hat_sq = jnp.sum(z_hat ** 2, axis=-1, keepdims=True)
            dot = z_hat @ emb_table[1:].T
            dists = z_hat_sq + item_sq[None, :] - 2 * dot

            top1_items = jnp.argmin(dists, axis=-1) + 1
            top1_dists = jnp.min(dists, axis=-1)
            top1_items = np.array(top1_items).reshape(actual_bs, num_seeds)
            top1_dists = np.array(top1_dists).reshape(actual_bs, num_seeds)

            batch_predictions = []
            for j in range(actual_bs):
                sorted_idx = np.argsort(top1_dists[j])
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
    # 8. Training Loop
    # =========================================================================
    writer = None
    if args.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        print(f"TensorBoard: {args.tb_log_dir}")

    best_params = None
    best_val_metrics = {}
    patience_counter = 0

    batch_size = args.batch_size
    num_samples = len(train_tar)
    epoch_rng = jax.random.PRNGKey(777)
    global_step = 0

    print(f"\nE2E training for {args.epochs} epochs...")
    print(f"Loss: L_flow + {args.lambda_ce} * L_ce")
    print(f"Seeds: {args.num_seeds}")

    for epoch in range(1, args.epochs + 1):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_in = train_in[indices]
        shuffled_tar = train_tar[indices]

        epoch_flow_loss = 0.0
        epoch_ce_loss_vae = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for i in range(0, num_samples, batch_size):
            if i + batch_size > num_samples:
                break

            batch_in = jnp.array(shuffled_in[i:i+batch_size])
            batch_tar = jnp.array(shuffled_tar[i:i+batch_size])

            epoch_rng, step_rng = jax.random.split(epoch_rng)
            params, opt_state, total_loss, flow_loss, ce_loss_vae, kl_loss = train_step(
                params, opt_state, batch_in, batch_tar, step_rng
            )

            epoch_flow_loss += flow_loss
            epoch_ce_loss_vae += ce_loss_vae
            epoch_kl_loss += kl_loss
            num_batches += 1
            global_step += 1

            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("Loss/flow", float(flow_loss), global_step)
                writer.add_scalar("Loss/ce_vae", float(ce_loss_vae), global_step)
                writer.add_scalar("Loss/kl", float(kl_loss), global_step)
                writer.add_scalar("Loss/total", float(total_loss), global_step)

        elapsed = time.time() - start_time
        n = max(num_batches, 1)
        avg_flow = float(epoch_flow_loss) / n
        avg_ce_v = float(epoch_ce_loss_vae) / n
        avg_kl = float(epoch_kl_loss) / n
        print(f"Epoch {epoch:02d}/{args.epochs} | Flow: {avg_flow:.2f}  CE(v): {avg_ce_v:.4f}  KL: {avg_kl:.4f} | Time: {elapsed:.1f}s")

        if writer is not None:
            writer.add_scalar("Loss/epoch_flow", avg_flow, global_step)
            writer.add_scalar("Loss/epoch_ce_vae", avg_ce_v, global_step)
            writer.add_scalar("Loss/epoch_kl", avg_kl, global_step)

        # --- Validation (multiple denoise steps) ---
        print(f"Evaluating...")
        val_step_counts = [1, 3, 5, 10]
        best_step_hr = 0.0
        best_step_results = None

        for n_steps in val_step_counts:
            val_results = evaluate_flow(params, val_in, val_tar, num_steps=n_steps)
            hr = val_results["HR@10"]
            ndcg = val_results["NDCG@10"]
            print(
                f"  steps={n_steps:2d} | NDCG@10: {ndcg:.5f} | "
                f"HR@10: {hr:.5f} | MRR: {val_results['MRR']:.5f} | "
                f"Diversity: {val_results['Diversity']:.3f}"
            )
            if writer is not None:
                for metric, score in val_results.items():
                    writer.add_scalar(f"Val_s{n_steps}/{metric}", score, global_step)

            if hr > best_step_hr:
                best_step_hr = hr
                best_step_results = val_results

        # Check improvement (based on best step count)
        improved = False
        for metric, score in best_step_results.items():
            if metric == "Diversity":
                continue
            if metric not in best_val_metrics or score > best_val_metrics[metric]:
                best_val_metrics[metric] = score
                improved = True

        if improved:
            best_params = params
            patience_counter = 0
            print(">>> New best! Saving...")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt = {"params": params, "epoch": epoch}
            with open(os.path.join(args.checkpoint_dir, "best_checkpoint.msgpack"), "wb") as f:
                f.write(flax.serialization.to_bytes(ckpt))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # =========================================================================
    # 9. Final Test
    # =========================================================================
    if best_params is None:
        best_params = params

    # =========================================================================
    # 10. Final Test — compare different denoising steps
    # =========================================================================
    print("\n" + "=" * 60)
    print("Final Test: comparing denoising steps")
    print("=" * 60)

    step_counts = [1, 3, 5, 10, 20]
    for n_steps in step_counts:
        print(f"\n--- Test with {n_steps} denoise steps ---")

        def evaluate_flow_nsteps(params, eval_in, eval_tar, num_steps, batch_size=None):
            if batch_size is None:
                batch_size = args.batch_size
            num_seeds = args.num_seeds
            num_samples = len(eval_in)
            emb_table = params["item_embedding"]["embedding"]
            item_sq_local = jnp.sum(emb_table[1:] ** 2, axis=-1)

            all_ranks = []
            total_unique = 0
            total_predictions = 0

            for i in range(0, num_samples, batch_size):
                batch_in = eval_in[i:i+batch_size]
                batch_tar = eval_tar[i:i+batch_size]
                actual_bs = len(batch_tar)

                h_user = get_h_user(params, jnp.array(batch_in))
                h_user_rep = jnp.repeat(h_user, num_seeds, axis=0)
                noise_rng = jax.random.PRNGKey(i)
                z_T = jax.random.normal(noise_rng, (actual_bs * num_seeds, args.embedding_dim))
                z_hat = denoise_n_steps(params, h_user_rep, z_T, num_steps)

                z_hat_sq = jnp.sum(z_hat ** 2, axis=-1, keepdims=True)
                dot = z_hat @ emb_table[1:].T
                dists = z_hat_sq + item_sq_local[None, :] - 2 * dot

                top1_items = jnp.argmin(dists, axis=-1) + 1
                top1_dists = jnp.min(dists, axis=-1)
                top1_items = np.array(top1_items).reshape(actual_bs, num_seeds)
                top1_dists = np.array(top1_dists).reshape(actual_bs, num_seeds)

                batch_predictions = []
                for j in range(actual_bs):
                    sorted_idx = np.argsort(top1_dists[j])
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

        results = evaluate_flow_nsteps(best_params, test_in, test_tar, n_steps)
        for metric, score in results.items():
            print(f"  {metric}: {score:.5f}")
        if writer is not None:
            for metric, score in results.items():
                writer.add_scalar(f"Test_steps{n_steps}/{metric}", score, global_step)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
