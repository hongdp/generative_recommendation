"""HSTU + Flow Head: Flow matching in HSTU's learned embedding space.

Architecture:
  1. HSTU backbone (frozen): item_ids → HSTU blocks → h_user
  2. FlowHead (trainable): h_user + z_t + t → v_hat (velocity prediction)
  3. ANN retrieval: z_0_hat vs HSTU item_embeddings

The HSTU backbone is loaded from a pretrained checkpoint and frozen.
Only the FlowHead is trained. This tests whether flow matching can
retrieve items when operating in a well-structured embedding space
that already encodes collaborative filtering + popularity signals.
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
from models.hstu import HSTUModel
from models.tiger_flow import FlowHead, sinusoidal_embedding
from evaluation.metrics import compute_ranks_from_predictions, calculate_metrics_from_ranks


def main():
    parser = argparse.ArgumentParser(description="HSTU + Flow Head training")
    parser.add_argument("--hstu_ckpt", type=str,
        default="./data/hstu_steam_checkpoints/best_checkpoint.msgpack")
    parser.add_argument("--checkpoint_dir", type=str, default="./data/hstu_flow_steam_checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tb_log_dir", type=str, default="./data/tensorboard/hstu_flow_steam")
    parser.add_argument("--dataset", type=str, default="steam",
        choices=["ml-1m", "beauty", "sports", "toys", "steam"])
    parser.add_argument("--patience", type=int, default=5)
    # HSTU architecture (must match checkpoint)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--linear_dim", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    # Flow head
    parser.add_argument("--flow_hidden_dim", type=int, default=512)
    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--denoise_steps", type=int, default=3)
    args = parser.parse_args()

    dataset = args.dataset.lower()
    max_len = 20 if dataset in ["beauty", "sports", "toys", "steam"] else 50

    print(f"--- HSTU + Flow Head on {dataset.upper()} ---")
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
    print(f"Dataset stats: Users = {loader.num_users}, Items = {num_items}")

    # =========================================================================
    # 2. Prepare Data
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
    # 3. Load Pretrained HSTU
    # =========================================================================
    print(f"Loading HSTU checkpoint: {args.hstu_ckpt}")
    hstu = HSTUModel(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        max_sequence_len=max_len,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
    )

    key = jax.random.PRNGKey(42)
    dummy_seq = jnp.zeros((1, max_len), dtype=jnp.int32)
    hstu_vars = hstu.init(key, dummy_seq)

    with open(args.hstu_ckpt, "rb") as f:
        hstu_ckpt = flax.serialization.from_bytes(hstu_vars, f.read())
    hstu_params = hstu_ckpt.get("params", hstu_ckpt)
    if "params" in hstu_params:
        hstu_params = hstu_params["params"]
    print("HSTU loaded successfully (frozen)")

    # Extract item embeddings for ANN retrieval
    item_embeddings = jnp.array(hstu_params["item_embedding"]["embedding"])  # [num_items+1, emb_dim]
    print(f"Item embeddings: {item_embeddings.shape}")

    # Precompute norms for cosine ANN
    item_norms = jnp.linalg.norm(item_embeddings[1:], axis=-1, keepdims=True) + 1e-8
    item_normed = item_embeddings[1:] / item_norms

    # Also precompute for L2 ANN
    item_sq = jnp.sum(item_embeddings[1:] ** 2, axis=-1)

    # =========================================================================
    # 4. Setup HSTU Encoder (returns hidden states, no logit projection)
    # =========================================================================

    # We need a clean way to get hidden states from HSTU.
    # Create a wrapper model that only returns hidden states.
    class HSTUEncoder(nn.Module):
        """HSTU that returns hidden states instead of logits."""
        num_items: int
        embedding_dim: int = 256
        num_blocks: int = 4
        num_heads: int = 4
        attention_dim: int = 128
        linear_dim: int = 512
        attn_dropout_rate: float = 0.1
        linear_dropout_rate: float = 0.1
        max_sequence_len: int = 50

        @nn.compact
        def __call__(self, item_seq, deterministic=True):
            from models.hstu import HSTUBlock
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
            return x  # [batch, seq_len, emb_dim] — hidden states, NO logit projection

    hstu_enc = HSTUEncoder(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        max_sequence_len=max_len,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
    )

    # Initialize with dummy and load HSTU params (same param names)
    hstu_enc_vars = hstu_enc.init(key, dummy_seq)
    # The params structure matches HSTUModel exactly (same names), minus the logit projection
    # (which is weight-tied and doesn't have separate params).
    # So we can directly reuse hstu_params!
    hstu_enc_params = hstu_params  # Same structure!

    # Verify by running a forward pass
    test_out = hstu_enc.apply({"params": hstu_enc_params}, dummy_seq, deterministic=True)
    print(f"HSTU encoder output shape: {test_out.shape}")  # Should be [1, max_len, emb_dim]

    @jax.jit
    def get_user_repr(hstu_params, item_seq):
        """Get user representation from frozen HSTU."""
        h = hstu_enc.apply({"params": hstu_params}, item_seq, deterministic=True)
        return h[:, -1, :]  # [batch, emb_dim] — last position

    # =========================================================================
    # 5. Initialize Flow Head
    # =========================================================================
    flow_head = FlowHead(
        hidden_dim=args.flow_hidden_dim,
        output_dim=args.embedding_dim,
    )

    k1 = jax.random.PRNGKey(123)
    dummy_h = jnp.zeros((1, args.embedding_dim))
    dummy_z = jnp.zeros((1, args.embedding_dim))
    dummy_t = jnp.zeros((1,))
    flow_vars = flow_head.init(k1, dummy_h, dummy_z, dummy_t)
    flow_params = flow_vars["params"]

    num_flow_params = sum(p.size for p in jax.tree.leaves(flow_params))
    print(f"FlowHead parameters: {num_flow_params:,}")

    # =========================================================================
    # 6. Optimizer (only for FlowHead)
    # =========================================================================
    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(flow_params)

    # =========================================================================
    # 7. Training Step (HSTU frozen, only FlowHead trained)
    # =========================================================================
    @jax.jit
    def train_step(flow_params, opt_state, h_user, z_target, rng_key):
        """Train FlowHead with flow matching loss.

        Args:
            h_user: User representation from frozen HSTU [bs, emb_dim]
            z_target: Target item embedding from HSTU [bs, emb_dim]
        """
        def loss_fn(p):
            noise_rng, t_rng = jax.random.split(rng_key)
            batch_size = z_target.shape[0]

            epsilon = jax.random.normal(noise_rng, z_target.shape)
            t = jax.random.uniform(t_rng, (batch_size,))
            t_expand = t[:, None]
            z_t = (1 - t_expand) * z_target + t_expand * epsilon
            v_target = epsilon - z_target

            v_hat = flow_head.apply({"params": p}, h_user, z_t, t)
            loss = jnp.mean(jnp.sum((v_hat - v_target) ** 2, axis=-1))
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(flow_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, flow_params)
        new_params = optax.apply_updates(flow_params, updates)
        return new_params, new_opt_state, loss

    # =========================================================================
    # 8. Inference Functions
    # =========================================================================
    @jax.jit
    def predict_velocity(flow_params, h_user, z, t):
        return flow_head.apply({"params": flow_params}, h_user, z, t)

    def denoise_n_steps(flow_params, h_user, z_T, num_steps):
        dt = 1.0 / num_steps
        z = z_T
        for step in range(num_steps):
            t_val = 1.0 - step * dt
            t = jnp.full((z.shape[0],), t_val)
            v_hat = predict_velocity(flow_params, h_user, z, t)
            z = z - dt * v_hat
        return z

    # =========================================================================
    # 9. Evaluation
    # =========================================================================
    def evaluate(flow_params, hstu_params, eval_in, eval_tar, batch_size=None):
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

            # Get user representation from frozen HSTU
            item_seq = jnp.array(batch_in)
            h_user = get_user_repr(hstu_params, item_seq)  # [bs, emb_dim]

            # Multi-seed denoising
            h_user_rep = jnp.repeat(h_user, num_seeds, axis=0)  # [bs*seeds, emb_dim]
            noise_rng = jax.random.PRNGKey(i)
            z_T = jax.random.normal(noise_rng, (actual_bs * num_seeds, args.embedding_dim))

            z_hat = denoise_n_steps(flow_params, h_user_rep, z_T, num_steps)

            # L2 ANN retrieval in HSTU embedding space
            z_hat_sq = jnp.sum(z_hat ** 2, axis=-1, keepdims=True)
            dot = z_hat @ item_embeddings[1:].T
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
    # 10. Setup
    # =========================================================================
    writer = None
    if args.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        print(f"TensorBoard logging: {args.tb_log_dir}")

    best_val_ndcg = -1.0
    best_flow_params = None
    best_val_metrics = {}
    patience_counter = 0

    # =========================================================================
    # 11. Training Loop
    # =========================================================================
    batch_size = args.batch_size
    num_samples = len(train_tar)
    epoch_rng = jax.random.PRNGKey(777)
    global_step = 0

    print(f"\nTraining FlowHead for {args.epochs} epochs (HSTU frozen)...")
    print(f"Denoise steps: {args.denoise_steps}, Seeds: {args.num_seeds}")

    # Precompute: HSTU user representations for training data
    # This is expensive but only done once since HSTU is frozen.
    print("Precomputing HSTU user representations for training data...")
    all_h_user = []
    precomp_bs = 1024
    for i in range(0, num_samples, precomp_bs):
        batch_in = train_in[i : i + precomp_bs]
        h = get_user_repr(hstu_params, jnp.array(batch_in))
        all_h_user.append(np.array(h))
        if (i // precomp_bs) % 100 == 0:
            print(f"  {i}/{num_samples}...")
    train_h_user = np.concatenate(all_h_user, axis=0)
    print(f"Precomputed user representations: {train_h_user.shape}")

    # Precompute target embeddings
    train_z_target = np.array(item_embeddings[train_tar])
    print(f"Target embeddings: {train_z_target.shape}")

    for epoch in range(1, args.epochs + 1):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for i in range(0, num_samples, batch_size):
            if i + batch_size > num_samples:
                break

            batch_idx = indices[i : i + batch_size]
            batch_h = jnp.array(train_h_user[batch_idx])
            batch_z = jnp.array(train_z_target[batch_idx])

            epoch_rng, step_rng = jax.random.split(epoch_rng)
            flow_params, opt_state, loss = train_step(
                flow_params, opt_state, batch_h, batch_z, step_rng
            )

            epoch_loss += loss
            num_batches += 1
            global_step += 1

            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("Loss/flow", float(loss), global_step)

        elapsed = time.time() - start_time
        n = max(num_batches, 1)
        print(f"Epoch {epoch:02d}/{args.epochs} | Flow Loss: {float(epoch_loss)/n:.4f} | Time: {elapsed:.1f}s")

        # --- Validation ---
        print(f"Evaluating validation split...")
        val_results = evaluate(flow_params, hstu_params, val_in, val_tar)
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
            best_flow_params = flow_params
            patience_counter = 0
            print(">>> New best! Saving checkpoint...")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt = {"flow_params": flow_params, "epoch": epoch, "best_val_ndcg": best_val_ndcg}
            with open(os.path.join(args.checkpoint_dir, "best_checkpoint.msgpack"), "wb") as f:
                f.write(flax.serialization.to_bytes(ckpt))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # =========================================================================
    # 12. Final Test
    # =========================================================================
    if best_flow_params is None:
        best_flow_params = flow_params

    print("\nFinal test evaluation...")
    test_results = evaluate(best_flow_params, hstu_params, test_in, test_tar)
    print("\n--- Final Test Results ---")
    for metric, score in test_results.items():
        print(f"{metric}: {score:.5f}")

    if writer is not None:
        for metric, score in test_results.items():
            writer.add_scalar(f"Test/{metric}", score, global_step)
        writer.close()


if __name__ == "__main__":
    main()
