"""HSTU + Flow Head: Joint training with frozen embedding table.

Architecture:
  1. Item embedding table (frozen, from pretrained HSTU): provides the target space
  2. HSTU backbone (trainable, random init): encodes user history → h_user
  3. FlowHead (trainable): h_user + z_t + t → v_hat (denoises in embedding space)

Key design:
  - Embedding table is frozen → stable target space with CF structure
  - HSTU backbone learns from scratch → optimized for flow matching, not softmax
  - stop_gradient on target → prevents flow loss from moving embeddings
  - Flow loss gradients flow back through h_user → HSTU learns user encoding
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
from models.hstu import HSTUModel, HSTUBlock
from models.tiger_flow import FlowHead, sinusoidal_embedding
from evaluation.metrics import compute_ranks_from_predictions, calculate_metrics_from_ranks
from models.hstu_flow import HSTUFlowModel


def main():
    parser = argparse.ArgumentParser(description="HSTU + Flow: Joint Training")
    parser.add_argument("--hstu_ckpt", type=str,
        default="./data/hstu_steam_checkpoints/best_checkpoint.msgpack",
        help="Pretrained HSTU checkpoint (only embedding table is loaded).")
    parser.add_argument("--checkpoint_dir", type=str, default="./data/hstu_flow_joint_steam_checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tb_log_dir", type=str, default="./data/tensorboard/hstu_flow_joint_steam")
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
    parser.add_argument("--denoise_steps", type=int, default=3)
    args = parser.parse_args()

    dataset = args.dataset.lower()
    max_len = 20 if dataset in ["beauty", "sports", "toys", "steam"] else 50

    print(f"--- HSTU + Flow Joint Training on {dataset.upper()} ---")
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
    # 3. Load Frozen Embedding Table from HSTU
    # =========================================================================
    print(f"Loading embedding table from: {args.hstu_ckpt}")
    hstu_dummy = HSTUModel(num_items=num_items, embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks, num_heads=args.num_heads,
        attention_dim=args.attention_dim, linear_dim=args.linear_dim,
        max_sequence_len=max_len)
    key = jax.random.PRNGKey(42)
    dummy_seq = jnp.zeros((1, max_len), dtype=jnp.int32)
    hstu_vars = hstu_dummy.init(key, dummy_seq)

    with open(args.hstu_ckpt, "rb") as f:
        hstu_ckpt = flax.serialization.from_bytes(hstu_vars, f.read())
    hstu_ckpt_params = hstu_ckpt.get("params", hstu_ckpt)
    if "params" in hstu_ckpt_params:
        hstu_ckpt_params = hstu_ckpt_params["params"]

    # Extract frozen embedding table
    emb_table = jnp.array(hstu_ckpt_params["item_embedding"]["embedding"])  # [num_items+1, emb_dim]
    print(f"Frozen embedding table: {emb_table.shape}")

    # Precompute for L2 ANN
    item_sq = jnp.sum(emb_table[1:] ** 2, axis=-1)  # [num_items]

    # =========================================================================
    # 4. Initialize Model (random HSTU blocks + FlowHead)
    # =========================================================================
    model = HSTUFlowModel(
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

    dummy_x = jnp.zeros((1, max_len, args.embedding_dim))
    dummy_z = jnp.zeros((1, args.embedding_dim))
    dummy_t = jnp.zeros((1,))
    model_vars = model.init(key, dummy_x, dummy_z, dummy_t)
    params = model_vars["params"]

    num_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Trainable parameters: {num_params:,} (HSTU blocks + FlowHead, random init)")
    print(f"Frozen parameters: {emb_table.size:,} (embedding table)")

    # =========================================================================
    # 5. Optimizer
    # =========================================================================
    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    # =========================================================================
    # 6. Training Step
    # =========================================================================
    @jax.jit
    def train_step(params, opt_state, batch_in_ids, batch_tar_ids, rng_key):
        """Joint training: HSTU encode → FlowHead denoise.

        Embedding table is frozen (not in params).
        Target uses stop_gradient.
        """
        def loss_fn(p):
            noise_rng, t_rng, dropout_rng = jax.random.split(rng_key, 3)
            batch_size = batch_tar_ids.shape[0]

            # Look up from frozen embedding table
            x_embedded = emb_table[batch_in_ids]  # [bs, max_len, emb_dim]
            z_target = jax.lax.stop_gradient(emb_table[batch_tar_ids])  # [bs, emb_dim]

            # Flow matching
            epsilon = jax.random.normal(noise_rng, z_target.shape)
            t = jax.random.uniform(t_rng, (batch_size,))
            t_expand = t[:, None]
            z_t = (1 - t_expand) * z_target + t_expand * epsilon
            v_target = epsilon - z_target

            v_hat = model.apply(
                {"params": p}, x_embedded, z_t, t,
                rngs={"dropout": dropout_rng},
                deterministic=False,
            )

            loss = jnp.mean(jnp.sum((v_hat - v_target) ** 2, axis=-1))
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # =========================================================================
    # 7. Inference
    # =========================================================================
    # Create a standalone encoder that mirrors HSTUFlowModel's HSTU blocks
    class _HSTUEncoder(nn.Module):
        """Mirrors HSTUFlowModel's encoder blocks for inference."""
        embedding_dim: int
        num_blocks: int
        num_heads: int
        attention_dim: int
        linear_dim: int
        attn_dropout_rate: float
        linear_dropout_rate: float
        max_sequence_len: int

        @nn.compact
        def __call__(self, x_embedded, deterministic=True):
            x = x_embedded
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
            return x[:, -1, :]

    hstu_encoder = _HSTUEncoder(
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
        max_sequence_len=max_len,
    )
    # Initialize so we know the param structure
    _enc_vars = hstu_encoder.init(key, dummy_x)

    @jax.jit
    def get_h_user(params, batch_in_ids):
        x_embedded = emb_table[batch_in_ids]
        # Extract only the HSTU block params (same names as in HSTUFlowModel)
        enc_params = {k: v for k, v in params.items() if k.startswith("hstu_block_")}
        return hstu_encoder.apply({"params": enc_params}, x_embedded, deterministic=True)

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
    # 8. Evaluation
    # =========================================================================
    def evaluate(params, eval_in, eval_tar, batch_size=None):
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

            # HSTU encode
            h_user = get_h_user(params, jnp.array(batch_in))  # [bs, emb_dim]

            # Multi-seed denoise
            h_user_rep = jnp.repeat(h_user, num_seeds, axis=0)
            noise_rng = jax.random.PRNGKey(i)
            z_T = jax.random.normal(noise_rng, (actual_bs * num_seeds, args.embedding_dim))

            z_hat = denoise_n_steps(params, h_user_rep, z_T, num_steps)

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
    # 9. Training Loop
    # =========================================================================
    writer = None
    if args.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        print(f"TensorBoard: {args.tb_log_dir}")

    best_val_ndcg = -1.0
    best_params = None
    best_val_metrics = {}
    patience_counter = 0

    batch_size = args.batch_size
    num_samples = len(train_tar)
    epoch_rng = jax.random.PRNGKey(777)
    global_step = 0

    print(f"\nJoint training for {args.epochs} epochs...")
    print(f"Frozen: embedding table | Trainable: HSTU blocks + FlowHead")
    print(f"Denoise steps: {args.denoise_steps}, Seeds: {args.num_seeds}")

    for epoch in range(1, args.epochs + 1):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_in = train_in[indices]
        shuffled_tar = train_tar[indices]

        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for i in range(0, num_samples, batch_size):
            if i + batch_size > num_samples:
                break

            batch_in = jnp.array(shuffled_in[i : i + batch_size])
            batch_tar = jnp.array(shuffled_tar[i : i + batch_size])

            epoch_rng, step_rng = jax.random.split(epoch_rng)
            params, opt_state, loss = train_step(params, opt_state, batch_in, batch_tar, step_rng)

            epoch_loss += loss
            num_batches += 1
            global_step += 1

            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("Loss/flow", float(loss), global_step)

        elapsed = time.time() - start_time
        n = max(num_batches, 1)
        print(f"Epoch {epoch:02d}/{args.epochs} | Flow Loss: {float(epoch_loss)/n:.4f} | Time: {elapsed:.1f}s")

        if writer is not None:
            writer.add_scalar("Loss/epoch", float(epoch_loss) / n, global_step)

        # Validation
        print(f"Evaluating...")
        val_results = evaluate(params, val_in, val_tar)
        val_ndcg = val_results["NDCG@10"]
        val_hr = val_results["HR@10"]
        print(
            f"--- Val @ Epoch {epoch} | NDCG@10: {val_ndcg:.5f} | "
            f"HR@10: {val_hr:.5f} | MRR: {val_results['MRR']:.5f} | "
            f"Diversity: {val_results['Diversity']:.3f}"
        )

        if writer is not None:
            for metric, score in val_results.items():
                writer.add_scalar(f"Val/{metric}", score, global_step)

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
            print(">>> New best! Saving...")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt = {"params": params, "epoch": epoch, "best_val_ndcg": best_val_ndcg}
            with open(os.path.join(args.checkpoint_dir, "best_checkpoint.msgpack"), "wb") as f:
                f.write(flax.serialization.to_bytes(ckpt))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # =========================================================================
    # 10. Final Test
    # =========================================================================
    if best_params is None:
        best_params = params

    print("\nFinal test evaluation...")
    test_results = evaluate(best_params, test_in, test_tar)
    print("\n--- Final Test Results ---")
    for metric, score in test_results.items():
        print(f"{metric}: {score:.5f}")

    if writer is not None:
        for metric, score in test_results.items():
            writer.add_scalar(f"Test/{metric}", score, global_step)
        writer.close()


if __name__ == "__main__":
    main()
