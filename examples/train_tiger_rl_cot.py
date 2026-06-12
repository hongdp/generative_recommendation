"""Full-scale training and evaluation script for the TIGER Seq2Seq (Encoder-Decoder) recommendation model."""

import argparse
import os
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.serialization

from datasets import MovieLensDataLoader, AmazonDataLoader, SteamDataLoader
from models.tiger_rl_cot import TIGERRLCoTModel
from evaluation.metrics import compute_ranks_from_predictions, calculate_metrics_from_ranks


def noop():
    pass


def main():
    parser = argparse.ArgumentParser(description="TIGER Seq2Seq training and evaluation on recommendation datasets.")
    parser.add_argument("--checkpoint_dir", type=str, default="./data/tiger_seq2seq_checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--resume_path", type=str, default="", help="Path to checkpoint to resume training or evaluate.")
    parser.add_argument("--eval_only", action="store_true", help="Only run test set evaluation using --resume_path.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--tb_log_dir", type=str, default="./data/tensorboard/tiger_seq2seq_ml1m", help="TensorBoard log directory.")
    parser.add_argument("--semantic_ids_path", type=str, default="./data/semantic_ids.json", help="Path to Semantic IDs JSON file.")
    parser.add_argument("--z_anchor_path", type=str, default="./data/steam_t5_embeddings.npy", help="Path to frozen continuous text embeddings (Z).")
    parser.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "beauty", "sports", "toys", "steam"], help="Dataset name.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--embedding_dim", type=int, default=384, help="Embedding dimension.")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of model blocks.")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads.")
    parser.add_argument("--attention_dim", type=int, default=384, help="Attention projection dimension.")
    parser.add_argument("--linear_dim", type=int, default=1024, help="Linear layer projection dimension.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and evaluation.")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    if args.checkpoint_dir == "./data/tiger_seq2seq_checkpoints":
        args.checkpoint_dir = f"./data/tiger_rl_cot_{dataset}_checkpoints"
    
    if args.tb_log_dir == "./data/tensorboard/tiger_cot_steam":
        args.tb_log_dir = f"./data/tensorboard/tiger_rl_cot_{dataset}"

    if args.semantic_ids_path == "./data/semantic_ids.json" and dataset != "ml-1m":
        args.semantic_ids_path = f"./data/semantic_ids_{dataset}.json"

    print(f"--- Training TIGER RL-CoT Model on {args.dataset.upper()} ---")
    print("Device list:", jax.devices())

    # 1. Load data
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

    # Load Semantic IDs
    ids_path = args.semantic_ids_path
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"Semantic IDs not found at {ids_path}. Generate them first.")
    
    with open(ids_path, "r") as f:
        semantic_ids = {int(k): v for k, v in json.load(f).items()}

    # Constants
    K = 256  # Codebook size
    vocab_size = 3 * K + 2  # c1: [1, 256], c2: [257, 512], c3: [513, 768], start: 769, pad: 0
    start_token = vocab_size - 1
    max_len = 20 if dataset in ["beauty", "sports", "toys", "steam"] else 50
    beam_size = 20

    print(f"Loading continuous text anchors (Z) from: {args.z_anchor_path}")
    Z_frozen = np.load(args.z_anchor_path)

    # 2. Get splits (Encoder inputs are raw item IDs now!)
    print("Extracting dataset splits...")
    train_dataset = loader.get_split("train", max_len=max_len, format_type="index")
    train_enc_in, train_tar = train_dataset.to_numpy()
    print(f"Train split: {len(train_tar)} samples")

    val_dataset = loader.get_split("val", max_len=max_len, format_type="index")
    val_enc_in, val_tar = val_dataset.to_numpy()

    test_dataset = loader.get_split("test", max_len=max_len, format_type="index")
    test_enc_in, test_tar = test_dataset.to_numpy()

    # 3. Setup Model
    print("Initializing TIGER RL-CoT Model...")
    model = TIGERRLCoTModel(
        num_items=loader.num_items,
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        max_encoder_len=max_len,
        max_decoder_len=4,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
    )

    key = jax.random.PRNGKey(42)
    dummy_enc = jnp.zeros((1, max_len), dtype=jnp.int32)
    variables = model.init(key, dummy_enc)
    params = variables["params"]

    # 4. Setup Optimizer
    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    # 5. Define training step
    @jax.jit
    def train_step(params, opt_state, batch_enc, batch_z_tar, gumbel_key, dropout_key, temp):
        def loss_fn(p):
            e_out, _ = model.apply(
                {"params": p},
                batch_enc,
                deterministic=False,
                temperature=temp,
                hard=False,
                rngs={"dropout": dropout_key, "gumbel": gumbel_key},
            )
            # MSE loss on the continuous embedding
            loss_embed = jnp.mean(jnp.sum((e_out - batch_z_tar)**2, axis=-1))
            return loss_embed

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # 6. Define batched single-step prediction functions for beam search decoding
    @jax.jit
    def predict_enc(params, encoder_tokens):
        return model.apply(
            {"params": params},
            encoder_tokens,
            method=model.encode,
            deterministic=True,
        )

    @jax.jit
    def predict_dec_step(params, decoder_tokens, encoder_outputs, encoder_tokens):
        return model.apply(
            {"params": params},
            decoder_tokens,
            encoder_outputs,
            encoder_tokens,
            method=model.decode_step,
            deterministic=True,
        )

    # 7. Batched Beam Search decoder
    def beam_search_decode(params, batch_enc_in, B=10):
        batch_size = len(batch_enc_in)
        
        # Step 0: Encode the user sequences to get encoder outputs
        encoder_outputs = predict_enc(params, batch_enc_in)
        
        # Step 1: Decode Level 1 token
        dec_in1 = jnp.ones((batch_size, 1), dtype=jnp.int32) * start_token
        logits1, _ = predict_dec_step(params, dec_in1, encoder_outputs, batch_enc_in)
        
        # Log-probs for Level 1 tokens (indices 1 to 256)
        log_probs1 = jax.nn.log_softmax(logits1[:, 1 : 257], axis=-1)
        top_probs, top_indices = jax.lax.top_k(log_probs1, k=B)  # [batch_size, B]
        top_tokens1 = top_indices + 1

        # Step 2: Decode Level 2 token (Batched across beams)
        # Replicate context
        enc_out2 = np.repeat(encoder_outputs, B, axis=0)
        enc_in2 = np.repeat(batch_enc_in, B, axis=0)
        
        # Replicate and append decoder inputs
        dec_in2 = np.ones((batch_size * B, 1), dtype=np.int32) * start_token
        dec_in2 = np.concatenate([dec_in2, top_tokens1.reshape(-1, 1)], axis=-1) # [batch_size * B, 2]
        
        logits2, _ = predict_dec_step(params, jnp.array(dec_in2), enc_out2, enc_in2)
        log_probs2 = jax.nn.log_softmax(logits2[:, 257 : 513], axis=-1)  # [batch_size * B, 256]
        log_probs2 = log_probs2.reshape(batch_size, B, 256)
        
        # Cumulative probability
        cum_probs2 = top_probs[:, :, None] + log_probs2  # [batch_size, B, 256]
        cum_probs2 = cum_probs2.reshape(batch_size, -1)  # [batch_size, B * 256]
        top_probs2, top_flat_indices2 = jax.lax.top_k(cum_probs2, k=B)  # [batch_size, B]
        
        beam_idx2 = top_flat_indices2 // 256
        c2 = top_flat_indices2 % 256
        c1 = top_indices[np.arange(batch_size)[:, None], beam_idx2]
        
        top_tokens1_expanded = c1 + 1
        top_tokens2 = c2 + 257

        # Step 3: Decode Level 3 token
        enc_out3 = np.repeat(encoder_outputs, B, axis=0)
        enc_in3 = np.repeat(batch_enc_in, B, axis=0)
        
        dec_in3 = np.ones((batch_size * B, 1), dtype=np.int32) * start_token
        dec_in3 = np.concatenate([
            dec_in3,
            top_tokens1_expanded.reshape(-1, 1),
            top_tokens2.reshape(-1, 1)
        ], axis=-1) # [batch_size * B, 3]
        
        logits3, _ = predict_dec_step(params, jnp.array(dec_in3), enc_out3, enc_in3)
        log_probs3 = jax.nn.log_softmax(logits3[:, 513 : 769], axis=-1)  # [batch_size * B, 256]
        log_probs3 = log_probs3.reshape(batch_size, B, 256)
        
        cum_probs3 = top_probs2[:, :, None] + log_probs3  # [batch_size, B, 256]
        cum_probs3 = cum_probs3.reshape(batch_size, -1)  # [batch_size, B * 256]
        top_probs3, top_flat_indices3 = jax.lax.top_k(cum_probs3, k=B)  # [batch_size, B]
        
        beam_idx3 = top_flat_indices3 // 256
        c3 = top_flat_indices3 % 256
        c2_final = c2[np.arange(batch_size)[:, None], beam_idx3]
        c1_final = c1[np.arange(batch_size)[:, None], beam_idx3]

        # Step 4: Decode e_out
        enc_out4 = np.repeat(encoder_outputs, B, axis=0)
        enc_in4 = np.repeat(batch_enc_in, B, axis=0)
        
        dec_in4 = np.ones((batch_size * B, 1), dtype=jnp.int32) * start_token
        dec_in4 = np.concatenate([
            dec_in4,
            top_tokens1_expanded.reshape(-1, 1),
            top_tokens2.reshape(-1, 1),
            c3.reshape(-1, 1) + 513
        ], axis=-1) # [batch_size * B, 4]
        
        _, e_outs4 = predict_dec_step(params, jnp.array(dec_in4), enc_out4, enc_in4)
        e_outs_final = e_outs4.reshape(batch_size, B, 768)

        return e_outs_final

    # 8. Evaluation function
    Z_jax = jnp.array(Z_frozen)

    @jax.jit
    def batch_nearest_neighbor(e_outs_batch, Z):
        # e_outs_batch: [batch_size, B, 768]
        # Z: [num_items+1, 768]
        z_norm_sq = jnp.sum(Z**2, axis=-1)
        # e_outs_batch @ Z.T: [batch_size, B, num_items+1]
        scores = jnp.einsum('bvf,nf->bvn', e_outs_batch, Z) - 0.5 * z_norm_sq
        # Mask out padding item (index 0)
        scores = jnp.where(jnp.arange(Z.shape[0]) == 0, -1e9, scores)
        best_items = jnp.argmax(scores, axis=-1) # [batch_size, B]
        return best_items

    def evaluate_tiger(params, tokens_in, targets, batch_size=None):
        if batch_size is None:
            batch_size = args.batch_size
        num_samples = len(tokens_in)
        ranks = []
        
        total_paths = 0
        valid_paths = 0
        total_top1_paths = 0
        valid_top1_paths = 0
        
        for i in range(0, num_samples, batch_size):
            batch_in = tokens_in[i : i + batch_size]
            batch_tar = targets[i : i + batch_size]
            
            e_outs_final = beam_search_decode(params, batch_in, B=beam_size)
            best_items = batch_nearest_neighbor(e_outs_final, Z_jax) # [batch, B]
            
            # Map predictions
            batch_predictions = best_items.tolist()
            
            total_paths += len(batch_in) * beam_size
            valid_paths += len(batch_in) * beam_size  # Always valid in Soft Retrieval!
            total_top1_paths += len(batch_in)
            valid_top1_paths += len(batch_in)
                
            batch_ranks = compute_ranks_from_predictions(batch_predictions, batch_tar)
            ranks.extend(batch_ranks)

        ranks = np.array(ranks)
        results = calculate_metrics_from_ranks(ranks, k_list=[1, 5, 10, 20])
        results["Valid@1"] = float(valid_top1_paths) / total_top1_paths if total_top1_paths > 0 else 0.0
        results["Valid@Beam"] = float(valid_paths) / total_paths if total_paths > 0 else 0.0
        return results

    # 9. Resume training setup
    writer = None
    if args.tb_log_dir and not args.eval_only:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        print(f"TensorBoard logging enabled. Logs saved to {args.tb_log_dir}")

    start_epoch = 1
    best_val_ndcg = -1.0
    best_params = None
    best_val_metrics = {}
    patience = args.patience
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
            checkpoint_state = flax.serialization.from_bytes(state_template, f.read())
        
        params = checkpoint_state["params"]
        opt_state = checkpoint_state["opt_state"]
        start_epoch = checkpoint_state["epoch"] + 1
        best_val_ndcg = float(checkpoint_state["best_val_ndcg"])
        best_params = params
        best_val_metrics["NDCG@10"] = best_val_ndcg
        print(f"Resumed from epoch {checkpoint_state['epoch']} with best validation NDCG@10 = {best_val_ndcg:.5f}")

    if args.eval_only:
        if not args.resume_path:
            raise ValueError("Must specify --resume_path when using --eval_only.")
        print("\nRunning test evaluation only...")
        test_results = evaluate_tiger(best_params, test_enc_in, test_tar, batch_size=args.batch_size)
        print("\n--- Test Evaluation Results ---")
        for metric, score in test_results.items():
            print(f"{metric}: {score:.5f}")
        return

    # 10. Training Loop
    epochs = args.epochs
    batch_size = args.batch_size
    num_samples = len(train_tar)
    epoch_rng = jax.random.PRNGKey(777)

    num_batches = num_samples // batch_size
    global_step = (start_epoch - 1) * num_batches
    
    init_temp = 1.0
    min_temp = 0.1
    temp_decay = (init_temp - min_temp) / max(1, epochs)

    print(f"\nTraining TIGER Seq2Seq model for {epochs} epochs starting from epoch {start_epoch}...")
    for epoch in range(start_epoch, epochs + 1):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_enc_in = train_enc_in[indices]
        shuffled_tar = train_tar[indices]
        
        current_temp = max(min_temp, init_temp - temp_decay * (epoch - 1))
        print(f"--- Epoch {epoch}/{epochs} | Gumbel Temperature: {current_temp:.4f} ---")

        epoch_loss = 0.0
        num_batches_processed = 0
        start_time = time.time()
        
        for i in range(0, num_samples, batch_size):
            if i + batch_size > num_samples:
                break
                
            batch_enc = shuffled_enc_in[i : i + batch_size]
            batch_tar_items = shuffled_tar[i : i + batch_size]
            batch_z_tar = Z_frozen[batch_tar_items]
            
            epoch_rng, gumbel_rng, step_rng = jax.random.split(epoch_rng, 3)
            
            params, opt_state, loss_val = train_step(
                params,
                opt_state,
                jnp.array(batch_enc),
                jnp.array(batch_z_tar),
                gumbel_rng,
                step_rng,
                current_temp
            )
            epoch_loss += loss_val
            num_batches_processed += 1
            global_step += 1

            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("Loss/train_step", float(loss_val), global_step)

        elapsed = time.time() - start_time
        avg_loss = float(epoch_loss) / num_batches_processed
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        
        if writer is not None:
            writer.add_scalar("Loss/train_epoch", avg_loss, global_step)

        # Evaluate on validation split every epoch
        if True:
            print(f"Evaluating validation split at epoch {epoch}...")
            val_results = evaluate_tiger(params, val_enc_in, val_tar, batch_size=args.batch_size)
            val_ndcg = val_results["NDCG@10"]
            val_hr = val_results["HR@10"]
            val_mrr = val_results["MRR"]
            print(f"--- Validation @ Epoch {epoch} | NDCG@10: {val_ndcg:.5f} | HR@10: {val_hr:.5f} | MRR: {val_mrr:.5f}")

            if writer is not None:
                for metric, score in val_results.items():
                    writer.add_scalar(f"Val/{metric}", score, global_step)

            # Check for improvement in ANY metric (excluding validity metrics)
            improved = False
            for metric, score in val_results.items():
                if metric in ["Valid@1", "Valid@Beam"]:
                    continue
                if metric not in best_val_metrics or score > best_val_metrics[metric]:
                    best_val_metrics[metric] = score
                    improved = True

            if improved:
                best_val_ndcg = best_val_metrics.get("NDCG@10", best_val_ndcg)
                best_params = params
                patience_counter = 0
                print(">>> New best validation score on at least one metric! Saving checkpoint...")
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(args.checkpoint_dir, "best_checkpoint.msgpack")
                checkpoint_state = {
                    "params": params,
                    "opt_state": opt_state,
                    "epoch": epoch,
                    "best_val_ndcg": best_val_ndcg,
                }
                with open(checkpoint_path, "wb") as f:
                    f.write(flax.serialization.to_bytes(checkpoint_state))
                print(f"Checkpoint saved to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch} (no validation improvement on any metric for {patience} epochs).")
                    break

        # Save latest checkpoint
        latest_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.msgpack")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_state = {
            "params": params,
            "opt_state": opt_state,
            "epoch": epoch,
            "best_val_ndcg": best_val_ndcg,
        }
        with open(latest_path, "wb") as f:
            f.write(flax.serialization.to_bytes(checkpoint_state))

    # 11. Final Test evaluation using best checkpoint
    if best_params is None:
        best_params = params

    print("\nRunning final test evaluation...")
    test_results = evaluate_tiger(best_params, test_enc_in, test_tar, batch_size=args.batch_size)

    print("\n--- Final Test Evaluation Results ---")
    for metric, score in test_results.items():
        print(f"{metric}: {score:.5f}")

    if writer is not None:
        for metric, score in test_results.items():
            writer.add_scalar(f"Test/{metric}", score, global_step)
        writer.close()
        print("TensorBoard writer closed.")

    # 12. Document results in experiment_results.md
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = "experiment_results.md"
    results_row = (
        f"| {date_str} | TIGER RL-CoT (End-to-End Latent Routing) on {args.dataset.upper()} | Local (GeForce RTX 4080) | "
        f"{test_results['HR@5']:.5f} | {test_results['NDCG@5']:.5f} | {test_results['HR@10']:.5f} | {test_results['NDCG@10']:.5f} | {test_results['HR@20']:.5f} | {test_results['NDCG@20']:.5f} | {test_results['MRR']:.5f} | "
        f"Replication on {args.dataset} matching TIGER paper evaluation (Best Val NDCG@10={best_val_ndcg:.5f}) |"
    )

    with open(log_path, "a") as f:
        f.write(results_row + "\n")
    print(f"\nResults successfully written to {log_path}!")


if __name__ == "__main__":
    main()
