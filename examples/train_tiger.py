"""Full-scale training and evaluation script for the TIGER generative recommendation model on MovieLens-1M."""

import argparse
import os
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax

from datasets import MovieLensDataLoader, AmazonDataLoader, SteamDataLoader
from models.tiger_model import TIGERModel
from models.tiger_tokenization import (
    load_semantic_ids,
    semantic_ids_hash,
    build_semantic_id_to_item,
    sequence_to_decoder_only_tokens,
    preprocess_decoder_only_training_data,
)
from evaluation.evaluator import Evaluator
from evaluation.tiger_decode import make_decoder_only_predictor, beam_search_decode_decoder_only
from evaluation.training_utils import (
    EarlyStopper,
    save_checkpoint,
    load_checkpoint,
    verify_semantic_ids_hash,
    assert_decode_validity,
)


def main():
    parser = argparse.ArgumentParser(description="TIGER training and evaluation on sequential recommendation datasets.")
    parser.add_argument("--checkpoint_dir", type=str, default="./data/tiger_checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--resume_path", type=str, default="", help="Path to checkpoint to resume training or evaluate.")
    parser.add_argument("--eval_only", action="store_true", help="Only run test set evaluation using --resume_path.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--tb_log_dir", type=str, default="./data/tensorboard/tiger_ml1m", help="TensorBoard log directory.")
    parser.add_argument("--semantic_ids_path", type=str, default="./data/semantic_ids.json", help="Path to Semantic IDs JSON file.")
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
    if args.checkpoint_dir == "./data/tiger_checkpoints" and dataset != "ml-1m":
        args.checkpoint_dir = f"./data/tiger_{dataset}_checkpoints"
    if args.tb_log_dir == "./data/tensorboard/tiger_ml1m" and dataset != "ml-1m":
        args.tb_log_dir = f"./data/tensorboard/tiger_{dataset}"
    if args.semantic_ids_path == "./data/semantic_ids.json" and dataset != "ml-1m":
        args.semantic_ids_path = f"./data/semantic_ids_{dataset}.json"

    print(f"--- Replicating TIGER Results on {args.dataset.upper()} ---")
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

    semantic_ids = load_semantic_ids(ids_path)
    ids_hash = semantic_ids_hash(semantic_ids)
    print(f"Loaded Semantic IDs from {ids_path} (hash={ids_hash})")

    # Constants
    K = 256  # Codebook size
    vocab_size = 3 * K + 2  # c1: [1, 256], c2: [257, 512], c3: [513, 768], start: 769, pad: 0
    start_token = vocab_size - 1
    max_len = 20 if dataset in ["beauty", "sports", "toys", "steam"] else 50
    beam_size = 20

    # 2. Get splits and format to JAX/TIGER tokens
    print("Preprocessing datasets into TIGER tokens...")
    train_dataset = loader.get_split("train", max_len=max_len, format_type="index")
    train_in, train_tar = train_dataset.to_numpy()
    train_tokens_in, train_tokens_tar = preprocess_decoder_only_training_data(train_in, train_tar, semantic_ids, K, start_token)
    print(f"Train split: {len(train_tokens_tar)} samples")

    val_dataset = loader.get_split("val", max_len=max_len, format_type="index")
    val_in, val_tar = val_dataset.to_numpy()
    val_tokens_in = sequence_to_decoder_only_tokens(val_in, semantic_ids, K, start_token)

    test_dataset = loader.get_split("test", max_len=max_len, format_type="index")
    test_in, test_tar = test_dataset.to_numpy()
    test_tokens_in = sequence_to_decoder_only_tokens(test_in, semantic_ids, K, start_token)

    # 3. Setup Model
    print("Initializing TIGER Model...")
    model = TIGERModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        max_sequence_len=3 * max_len + 4,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
    )

    key = jax.random.PRNGKey(42)
    dummy_seq = jnp.zeros((1, 3 * max_len + 3), dtype=jnp.int32)
    variables = model.init(key, dummy_seq)
    params = variables["params"]

    # 4. Setup Optimizer
    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    # 5. Define training step
    @jax.jit
    def train_step(params, opt_state, batch_inputs, batch_targets, dropout_key):
        def loss_fn(p):
            logits = model.apply(
                {"params": p},
                batch_inputs,
                rngs={"dropout": dropout_key},
                deterministic=False,
            )
            # Mask out padding tokens (token ID 0) from loss computation
            mask = (batch_targets != 0).astype(jnp.float32)
            # Cross entropy loss
            loss_vals = optax.softmax_cross_entropy_with_integer_labels(logits, batch_targets)
            return jnp.sum(loss_vals * mask) / jnp.maximum(jnp.sum(mask), 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # 6. Beam search decoding (shared implementation, parametrized by codebook K)
    predict_next_token = make_decoder_only_predictor(model)

    def beam_search_decode(params, batch_inputs, B=10):
        return beam_search_decode_decoder_only(params, batch_inputs, predict_next_token, K, B=B)

    # 8. Evaluation function
    semantic_id_to_item = build_semantic_id_to_item(semantic_ids)
    evaluator = Evaluator(k_list=[1, 5, 10, 20])

    def evaluate_tiger(params, tokens_in, targets, batch_size=None):
        if batch_size is None:
            batch_size = args.batch_size

        def predict(batch_inputs):
            return beam_search_decode(params, batch_inputs, B=beam_size)

        return evaluator.evaluate_generative_discrete(
            predict, semantic_id_to_item, tokens_in, targets,
            beam_size=beam_size, batch_size=batch_size,
        )

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

    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_checkpoint.msgpack")

    if args.eval_only:
        # Evaluate the best-val checkpoint (explicit --resume_path overrides).
        eval_ckpt = args.resume_path or best_ckpt_path
        if not os.path.exists(eval_ckpt):
            raise ValueError(f"No checkpoint to evaluate at {eval_ckpt}.")
        print(f"Loading checkpoint for evaluation from {eval_ckpt}...")
        verify_semantic_ids_hash(eval_ckpt, ids_hash)
        checkpoint_state = load_checkpoint(eval_ckpt, params, opt_state)
        best_params = checkpoint_state["params"]
        print(f"Loaded checkpoint (epoch {checkpoint_state['epoch']}, "
              f"best val NDCG@10={checkpoint_state['best_val_ndcg']:.5f}).")
        test_results = evaluate_tiger(best_params, test_tokens_in, test_tar, batch_size=args.batch_size)
        assert_decode_validity(test_results)
        print("\n--- Test Evaluation Results ---")
        for metric, score in test_results.items():
            print(f"{metric}: {score:.5f}")
        return

    if args.resume_path:
        print(f"Loading checkpoint from {args.resume_path}...")
        verify_semantic_ids_hash(args.resume_path, ids_hash)
        checkpoint_state = load_checkpoint(args.resume_path, params, opt_state)
        params = checkpoint_state["params"]
        opt_state = checkpoint_state["opt_state"]
        start_epoch = checkpoint_state["epoch"] + 1
        best_val_ndcg = float(checkpoint_state["best_val_ndcg"])
        best_val_metrics["NDCG@10"] = best_val_ndcg
        # Restore the true best-val params from the best checkpoint so a resumed
        # run's final test eval uses best-val weights, not the latest epoch.
        if os.path.exists(best_ckpt_path):
            best_params = load_checkpoint(best_ckpt_path, params, opt_state)["params"]
        else:
            best_params = params
        print(f"Resumed from epoch {checkpoint_state['epoch']} with best validation NDCG@10 = {best_val_ndcg:.5f}")

    # 10. Training Loop
    epochs = args.epochs
    batch_size = args.batch_size
    num_samples = len(train_tokens_tar)
    epoch_rng = jax.random.PRNGKey(777)

    num_batches = num_samples // batch_size
    global_step = (start_epoch - 1) * num_batches

    print(f"\nTraining TIGER model for {epochs} epochs starting from epoch {start_epoch}...")
    for epoch in range(start_epoch, epochs + 1):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_in = train_tokens_in[indices]
        shuffled_tar = train_tokens_tar[indices]

        epoch_loss = 0.0
        num_batches_processed = 0
        start_time = time.time()
        
        for i in range(0, num_samples, batch_size):
            if i + batch_size > num_samples:
                break
                
            batch_in = shuffled_in[i : i + batch_size]
            batch_tar = shuffled_tar[i : i + batch_size]
            epoch_rng, step_rng = jax.random.split(epoch_rng)
            
            params, opt_state, loss_val = train_step(
                params, opt_state, jnp.array(batch_in), jnp.array(batch_tar), step_rng
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
            val_results = evaluate_tiger(params, val_tokens_in, val_tar, batch_size=args.batch_size)
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
                checkpoint_path = save_checkpoint(
                    params, opt_state, epoch, best_val_ndcg,
                    args.checkpoint_dir, semantic_ids_hash=ids_hash,
                )
                print(f"Checkpoint saved to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch} (no validation improvement on any metric for {patience} epochs).")
                    break

        # Save latest checkpoint
        save_checkpoint(
            params, opt_state, epoch, best_val_ndcg,
            args.checkpoint_dir, filename="latest_checkpoint.msgpack",
            semantic_ids_hash=ids_hash,
        )

    # 11. Final Test evaluation using best checkpoint
    if best_params is None:
        best_params = params

    print("\nRunning final test evaluation...")
    test_results = evaluate_tiger(best_params, test_tokens_in, test_tar, batch_size=args.batch_size)
    assert_decode_validity(test_results)

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
    model_desc = "TIGER (K-Means)" if "kmeans" in args.semantic_ids_path.lower() else "TIGER (VAE)"
    results_row = (
        f"| {date_str} | Full {model_desc} (blocks={args.num_blocks}, embed={args.embedding_dim}) on {args.dataset.upper()} | Local (GeForce RTX 4080) | "
        f"{test_results['HR@5']:.5f} | {test_results['NDCG@5']:.5f} | {test_results['HR@10']:.5f} | {test_results['NDCG@10']:.5f} | {test_results['HR@20']:.5f} | {test_results['NDCG@20']:.5f} | {test_results['MRR']:.5f} | "
        f"Replication on {args.dataset} matching TIGER paper evaluation (Best Val NDCG@10={best_val_ndcg:.5f}) |"
    )

    with open(log_path, "a") as f:
        f.write(results_row + "\n")
    print(f"\nResults successfully written to {log_path}!")


if __name__ == "__main__":
    main()
