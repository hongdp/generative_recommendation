"""TIGER Encoder + Direct Embedding CE: Ablation comparing TIGER's encoder
representation (using Semantic ID tokens) against a standard softmax CE
prediction head instead of the 3-level autoregressive decoder.

This tests whether TIGER's power comes from the encoder (understanding history
via Semantic ID tokenization) or the decoder (autoregressive ID generation).
"""

import argparse
import os
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from datasets import MovieLensDataLoader, AmazonDataLoader, SteamDataLoader
from models.tiger_seq2seq import TIGERSeq2SeqModel
from models.tiger_encoder_ce import TIGEREncoderCEModel
from evaluation.evaluator import Evaluator
from evaluation.training_utils import EarlyStopper, save_checkpoint, load_checkpoint, log_results_to_markdown


def sequence_to_tiger_tokens(item_seq, semantic_ids, K):
    """Converts a batch of item sequences into flat, level-shifted TIGER encoder tokens."""
    batch_size = len(item_seq)
    max_len = item_seq.shape[1]

    encoder_inputs = np.zeros((batch_size, 3 * max_len), dtype=np.int32)

    for i in range(batch_size):
        seq = item_seq[i]
        non_pad_indices = np.where(seq != 0)[0]
        num_pad = max_len - len(non_pad_indices)

        for idx, pos in enumerate(non_pad_indices):
            item = seq[pos]
            c1, c2, c3 = semantic_ids[item]
            write_pos = 3 * num_pad + 3 * idx
            encoder_inputs[i, write_pos] = c1 + 1
            encoder_inputs[i, write_pos + 1] = c2 + K + 1
            encoder_inputs[i, write_pos + 2] = c3 + 2 * K + 1

    return encoder_inputs




def main():
    parser = argparse.ArgumentParser(description="TIGER Encoder + CE: Ablation study.")
    parser.add_argument("--checkpoint_dir", type=str, default="./data/tiger_encoder_ce_checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tb_log_dir", type=str, default="./data/tensorboard/tiger_encoder_ce_steam")
    parser.add_argument("--semantic_ids_path", type=str, default="./data/semantic_ids.json")
    parser.add_argument("--dataset", type=str, default="steam",
        choices=["ml-1m", "beauty", "sports", "toys", "steam"])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--attention_dim", type=int, default=384)
    parser.add_argument("--linear_dim", type=int, default=1024)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    dataset = args.dataset.lower()
    if args.checkpoint_dir == "./data/tiger_encoder_ce_checkpoints" and dataset != "ml-1m":
        args.checkpoint_dir = f"./data/tiger_encoder_ce_{dataset}_checkpoints"
    if args.tb_log_dir == "./data/tensorboard/tiger_encoder_ce_steam" and dataset != "steam":
        args.tb_log_dir = f"./data/tensorboard/tiger_encoder_ce_{dataset}"
    if args.semantic_ids_path == "./data/semantic_ids.json" and dataset != "ml-1m":
        args.semantic_ids_path = f"./data/semantic_ids_{dataset}.json"

    print(f"--- TIGER Encoder + CE Ablation on {dataset.upper()} ---")
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
    num_items = loader.num_items
    print(f"Dataset stats: Users = {loader.num_users}, Items = {num_items}")

    # Load Semantic IDs
    ids_path = args.semantic_ids_path
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"Semantic IDs not found at {ids_path}.")
    with open(ids_path, "r") as f:
        semantic_ids = {int(k): v for k, v in json.load(f).items()}

    K = 256
    vocab_size = 3 * K + 2
    max_len = 20 if dataset in ["beauty", "sports", "toys", "steam"] else 50

    # 2. Prepare data
    print("Preprocessing datasets into TIGER tokens...")
    train_dataset = loader.get_split("train", max_len=max_len, format_type="index")
    train_in, train_tar = train_dataset.to_numpy()
    train_enc_in = sequence_to_tiger_tokens(train_in, semantic_ids, K)
    print(f"Train split: {len(train_tar)} samples")

    val_dataset = loader.get_split("val", max_len=max_len, format_type="index")
    val_in, val_tar = val_dataset.to_numpy()
    val_enc_in = sequence_to_tiger_tokens(val_in, semantic_ids, K)

    test_dataset = loader.get_split("test", max_len=max_len, format_type="index")
    test_in, test_tar = test_dataset.to_numpy()
    test_enc_in = sequence_to_tiger_tokens(test_in, semantic_ids, K)

    # 3. Setup Model
    print("Initializing TIGER Encoder + CE Model...")
    model = TIGEREncoderCEModel(
        num_items=num_items,
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        max_encoder_len=3 * max_len + 4,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
    )

    key = jax.random.PRNGKey(42)
    dummy_enc = jnp.zeros((1, 3 * max_len), dtype=jnp.int32)
    variables = model.init(key, dummy_enc)
    params = variables["params"]

    # 4. Setup Optimizer
    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    # 5. Training step
    @jax.jit
    def train_step(params, opt_state, batch_enc, batch_tar, dropout_key):
        def loss_fn(p):
            logits = model.apply(
                {"params": p}, batch_enc,
                rngs={"dropout": dropout_key},
                deterministic=False,
            )
            # Cross entropy over all items
            loss_vals = optax.softmax_cross_entropy_with_integer_labels(logits, batch_tar)
            return jnp.mean(loss_vals)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # 6. Prediction function for evaluation
    @jax.jit
    def predict_batch(params, batch_enc):
        logits = model.apply({"params": params}, batch_enc, deterministic=True)
        return logits

    # 7. Setup evaluator
    evaluator = Evaluator(k_list=[1, 5, 10, 20])

    # 8. Training loop
    writer = None
    if args.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        print(f"TensorBoard: {args.tb_log_dir}")

    early_stopper = EarlyStopper(patience=args.patience)
    batch_size = args.batch_size
    num_samples = len(train_tar)
    epoch_rng = jax.random.PRNGKey(777)
    global_step = 0

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_enc_in = train_enc_in[indices]
        shuffled_tar = train_tar[indices]

        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for i in range(0, num_samples, batch_size):
            if i + batch_size > num_samples:
                break

            batch_enc = jnp.array(shuffled_enc_in[i:i+batch_size])
            batch_tar = jnp.array(shuffled_tar[i:i+batch_size])
            epoch_rng, step_rng = jax.random.split(epoch_rng)

            params, opt_state, loss_val = train_step(params, opt_state, batch_enc, batch_tar, step_rng)
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1

            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("Loss/train_step", float(loss_val), global_step)

        elapsed = time.time() - start_time
        avg_loss = float(epoch_loss) / num_batches
        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

        if writer is not None:
            writer.add_scalar("Loss/train_epoch", avg_loss, global_step)

        # Evaluate on validation split
        print(f"Evaluating validation split at epoch {epoch}...")
        def val_predict(batch_inputs):
            return predict_batch(params, jnp.array(batch_inputs))

        val_results = evaluator.evaluate_index_based(
            val_predict, val_enc_in, val_tar, batch_size=batch_size,
        )
        val_ndcg = val_results["NDCG@10"]
        val_hr = val_results["HR@10"]
        val_mrr = val_results["MRR"]
        print(f"--- Validation @ Epoch {epoch} | NDCG@10: {val_ndcg:.5f} | HR@10: {val_hr:.5f} | MRR: {val_mrr:.5f}")

        if writer is not None:
            for metric, score in val_results.items():
                writer.add_scalar(f"Val/{metric}", score, global_step)

        improved = early_stopper.check(val_results, params)
        if improved:
            print(">>> New best! Saving checkpoint...")
            ckpt_path = save_checkpoint(
                params, opt_state, epoch,
                early_stopper.get_best("NDCG@10"),
                args.checkpoint_dir,
            )
            print(f"Checkpoint saved to {ckpt_path}")
        elif early_stopper.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

        save_checkpoint(
            params, opt_state, epoch,
            early_stopper.get_best("NDCG@10"),
            args.checkpoint_dir, filename="latest_checkpoint.msgpack",
        )

    # 9. Final test evaluation
    best_params = early_stopper.best_params if early_stopper.best_params is not None else params

    def test_predict(batch_inputs):
        return predict_batch(best_params, jnp.array(batch_inputs))

    print("\nRunning final test evaluation...")
    test_results = evaluator.evaluate_index_based(
        test_predict, test_enc_in, test_tar, batch_size=batch_size,
    )

    print("\n--- Final Test Evaluation Results ---")
    for metric, score in test_results.items():
        print(f"{metric}: {score:.5f}")

    if writer is not None:
        for metric, score in test_results.items():
            writer.add_scalar(f"Test/{metric}", score, global_step)
        writer.close()

    log_results_to_markdown(
        f"TIGER Encoder+CE (blocks={args.num_blocks}, embed={args.embedding_dim})",
        args.dataset, test_results,
        early_stopper.get_best("NDCG@10"),
    )


if __name__ == "__main__":
    main()
