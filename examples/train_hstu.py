"""Full-scale training and evaluation script to replicate HSTU results on MovieLens-1M."""

import argparse
import os
import time
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from datasets import MovieLensDataLoader, AmazonDataLoader, SteamDataLoader
from evaluation.evaluator import Evaluator
from models.hstu import HSTUModel


def create_model(model_name: str, num_items: int, max_len: int, args):
    if model_name.lower() == "hstu":
        return HSTUModel(
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
    elif model_name.lower() == "transformer":
        try:
            from models.transformer import TransformerModel
            return TransformerModel(
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
        except ImportError:
            raise NotImplementedError(
                f"Model architecture '{model_name}' is not implemented yet. "
                "Please implement TransformerModel under src/models/transformer.py first."
            )
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Pluggable sequential model training and evaluation on recommendation datasets.")
    parser.add_argument("--model", type=str, default="hstu", choices=["hstu", "transformer"], help="Model architecture.")
    parser.add_argument("--checkpoint_dir", type=str, default="./data/checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--resume_path", type=str, default="", help="Path to checkpoint to resume training or evaluate.")
    parser.add_argument("--eval_only", action="store_true", help="Only run test set evaluation using --resume_path.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--tb_log_dir", type=str, default="./data/tensorboard/hstu_ml1m", help="TensorBoard log directory.")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of model blocks.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--attention_dim", type=int, default=128, help="Attention projection dimension.")
    parser.add_argument("--linear_dim", type=int, default=512, help="Linear layer projection dimension.")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for attention and feed-forward layers.")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum sequence length.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay rate.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "beauty", "sports", "toys", "steam"], help="Dataset name.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    if args.checkpoint_dir == "./data/checkpoints" and dataset != "ml-1m":
        args.checkpoint_dir = f"./data/{args.model}_{dataset}_checkpoints"
    if args.tb_log_dir == "./data/tensorboard/hstu_ml1m" and dataset != "ml-1m":
        args.tb_log_dir = f"./data/tensorboard/{args.model}_{dataset}"

    writer = None
    if args.tb_log_dir and not args.eval_only:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        print(f"TensorBoard logging enabled. Logs saved to {args.tb_log_dir}")

    print(f"--- Replicating {args.model.upper()} Results on MovieLens-1M ---")
    print("Device list:", jax.devices())

    # 1. Initialize data loader
    data_dir = "./data"
    if dataset == "ml-1m":
        print(f"Loading MovieLens-1M dataset from {data_dir}...")
        loader = MovieLensDataLoader(dataset_name="ml-1m", data_dir=data_dir, min_rating=0)
    elif dataset in ["beauty", "sports", "toys"]:
        print(f"Loading Amazon {dataset} dataset from {data_dir}...")
        loader = AmazonDataLoader(category=dataset, data_dir=data_dir, min_rating=0)
    elif dataset == "steam":
        print(f"Loading Steam dataset from {data_dir}...")
        loader = SteamDataLoader(data_dir=data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    print(f"Dataset stats: Users = {loader.num_users}, Items = {loader.num_items}")

    # 2. Get chronological splits
    max_len = 20 if dataset in ["beauty", "sports", "toys", "steam"] else args.max_len
    print(f"Generating train, validation, and test splits (max_len={max_len})...")
    
    train_dataset = loader.get_split("train", max_len=max_len, format_type="index")
    train_inputs, train_targets = train_dataset.to_numpy()
    print(f"Train split: {len(train_targets)} samples")

    val_dataset = loader.get_split("val", max_len=max_len, format_type="index")
    val_inputs, val_targets = val_dataset.to_numpy()
    print(f"Validation split: {len(val_targets)} samples")

    test_dataset = loader.get_split("test", max_len=max_len, format_type="index")
    test_inputs, test_targets = test_dataset.to_numpy()
    print(f"Test split: {len(test_targets)} samples")

    # 3. Instantiate pluggable Model with parsed configurations
    print(f"Initializing full {args.model.upper()} Model...")
    model = create_model(args.model, loader.num_items, max_len, args)

    # Initialize variables
    key = jax.random.PRNGKey(42)
    dummy_seq = jnp.zeros((1, max_len), dtype=jnp.int32)
    variables = model.init(key, dummy_seq)
    params = variables["params"]

    # 4. Set up Optimizer (AdamW with weight decay)
    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    # 5. Define JIT training and evaluation functions
    @jax.jit
    def train_step(params, opt_state, batch_inputs, batch_targets, dropout_key):
        def loss_fn(p):
            # Pass dropout key to enable dropout
            logits = model.apply(
                {"params": p},
                batch_inputs,
                rngs={"dropout": dropout_key},
                deterministic=False,
            )
            # Predict next item on the last sequence position
            logits_last = logits[:, -1, :]
            # Cross entropy loss
            loss_vals = optax.softmax_cross_entropy_with_integer_labels(logits_last, batch_targets)
            return jnp.mean(loss_vals)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def predict_batch(params, batch_inputs):
        logits = model.apply({"params": params}, batch_inputs, deterministic=True)
        return logits[:, -1, :]

    # 6. Training loop with early stopping
    epochs = args.epochs
    batch_size = args.batch_size
    num_samples = len(train_targets)
    best_val_ndcg = -1.0
    best_params = None
    best_val_metrics = {}
    patience = args.patience  # Stop if validation doesn't improve
    patience_counter = 0

    evaluator = Evaluator(k_list=[1, 5, 10, 20])

    start_epoch = 1
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

    num_batches_per_epoch = num_samples // batch_size
    global_step = (start_epoch - 1) * num_batches_per_epoch

    if args.eval_only:
        if not args.resume_path:
            raise ValueError("Must specify --resume_path when using --eval_only.")
        def test_predict(batch_inputs):
            return predict_batch(best_params, batch_inputs)
        print("\nRunning test evaluation only...")
        test_results = evaluator.evaluate_index_based(
            test_predict, test_inputs, test_targets, batch_size=batch_size
        )
        print("\n--- Test Evaluation Results ---")
        for metric, score in test_results.items():
            print(f"{metric}: {score:.5f}")
        return

    print(f"\nTraining full {args.model.upper()} model for {epochs} epochs (batch_size={batch_size}), starting from epoch {start_epoch}...")
    epoch_rng = jax.random.PRNGKey(777)

    for epoch in range(start_epoch, epochs + 1):
        # Shuffle training data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_inputs = train_inputs[indices]
        shuffled_targets = train_targets[indices]

        epoch_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        for i in range(0, num_samples, batch_size):
            # Drop remainder to keep batch size static (prevents compilation overhead)
            if i + batch_size > num_samples:
                break
                
            batch_in = shuffled_inputs[i : i + batch_size]
            batch_tar = shuffled_targets[i : i + batch_size]
            
            # Split key for dropout
            epoch_rng, step_rng = jax.random.split(epoch_rng)
            
            params, opt_state, loss_val = train_step(params, opt_state, batch_in, batch_tar, step_rng)
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1
            
            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("Loss/train_step", float(loss_val), global_step)
            
        elapsed = time.time() - start_time
        avg_loss = float(epoch_loss) / num_batches
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

        if writer is not None:
            writer.add_scalar("Loss/train", avg_loss, global_step)

        # Evaluate on validation set every epoch
        if True:
            def val_predict(batch_inputs):
                return predict_batch(params, batch_inputs)
                
            val_results = evaluator.evaluate_index_based(
                val_predict, val_inputs, val_targets, batch_size=batch_size
            )
            val_ndcg = val_results["NDCG@10"]
            val_hr = val_results["HR@10"]
            val_mrr = val_results["MRR"]
            print(f"--- Validation @ Epoch {epoch} | NDCG@10: {val_ndcg:.5f} | HR@10: {val_hr:.5f} | MRR: {val_mrr:.5f}")

            if writer is not None:
                for metric, score in val_results.items():
                    writer.add_scalar(f"Val/{metric}", score, global_step)

            # Check for improvement in ANY metric
            improved = False
            for metric, score in val_results.items():
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
                    print(f"\nEarly stopping triggered after {epoch} epochs (no validation improvement on any metric for {patience} epochs).")
                    break

        # Save latest checkpoint at the end of each epoch
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

    # 7. Final Test Evaluation using the best checkpoints
    if best_params is None:
        best_params = params

    def test_predict(batch_inputs):
        return predict_batch(best_params, batch_inputs)

    print("\nRunning final test evaluation...")
    test_results = evaluator.evaluate_index_based(
        test_predict, test_inputs, test_targets, batch_size=batch_size
    )

    print("\n--- Final Test Evaluation Results ---")
    for metric, score in test_results.items():
        print(f"{metric}: {score:.5f}")

    # 8. Document results in experiment_results.md
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = "experiment_results.md"
    results_row = (
        f"| {date_str} | Full {args.model.upper()} (blocks={args.num_blocks}, embed={args.embedding_dim}) on {args.dataset.upper()} | Local (GeForce RTX 4080) | "
        f"{test_results['HR@5']:.5f} | {test_results['NDCG@5']:.5f} | {test_results['HR@10']:.5f} | {test_results['NDCG@10']:.5f} | {test_results['HR@20']:.5f} | {test_results['NDCG@20']:.5f} | {test_results['MRR']:.5f} | "
        f"Replication on {args.dataset} matching LIGER paper evaluation (Best Val NDCG@10={best_val_ndcg:.5f}) |"
    )

    with open(log_path, "a") as f:
        f.write(results_row + "\n")
    print(f"\nResults successfully written to {log_path}!")

    if writer is not None:
        for metric, score in test_results.items():
            writer.add_scalar(f"Test/{metric}", score, global_step)
        writer.close()
        print("TensorBoard writer closed.")


if __name__ == "__main__":
    main()
