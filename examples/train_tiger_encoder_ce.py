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
import functools
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import optax
import flax.linen as nn

from datasets import MovieLensDataLoader, AmazonDataLoader, SteamDataLoader
from models.tiger_seq2seq import TIGERSeq2SeqModel
from models.tiger_encoder_ce import TIGEREncoderCEModel
from models.tiger_tokenization import (
    load_semantic_ids,
    semantic_ids_hash,
    sequence_to_encoder_tokens as sequence_to_tiger_tokens,
)
from evaluation.evaluator import Evaluator
from evaluation.training_utils import (
    EarlyStopper,
    save_checkpoint,
    load_checkpoint,
    log_results_to_markdown,
    verify_semantic_ids_hash,
)
from flax.jax_utils import replicate, unreplicate
from flax.training.common_utils import shard
import grain.python as grain
import sys
from absl import flags
try:
    flags.FLAGS(sys.argv)
except Exception:
    pass




def main():
    parser = argparse.ArgumentParser(description="TIGER Encoder + CE: Ablation study.")
    parser.add_argument("--checkpoint_dir", type=str, default="./data/tiger_encoder_ce_checkpoints")
    parser.add_argument("--resume_path", type=str, default="", help="Path to checkpoint to resume training or evaluate.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tb_log_dir", type=str, default="./data/tensorboard/tiger_encoder_ce_steam")
    parser.add_argument("--semantic_ids_path", type=str, default="./data/semantic_ids.json")
    parser.add_argument("--dataset", type=str, default="steam",
        choices=["ml-1m", "beauty", "sports", "toys", "steam"])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--embedding_dim", type=int, default=384, help="Encoder dim (typically 384).")
    parser.add_argument("--item_embedding_dim", type=int, default=384, help="KNN space dim.")
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
    semantic_ids = load_semantic_ids(ids_path)
    ids_hash = semantic_ids_hash(semantic_ids)
    print(f"Loaded Semantic IDs from {ids_path} (hash={ids_hash})")

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
        encoder_dim=args.embedding_dim,
        item_embedding_dim=args.item_embedding_dim,
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

    # Replicate for pmap
    params = replicate(params)
    opt_state = replicate(opt_state)

    # 5. Define training step
    @functools.partial(jax.pmap, axis_name="batch")
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
        loss = lax.pmean(loss, axis_name="batch")
        grads = lax.pmean(grads, axis_name="batch")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # 6. Evaluation predict function
    @jax.pmap
    def predict_step(params, batch_enc):
        logits = model.apply({"params": params}, batch_enc, deterministic=True)
        return logits

    # 7. Setup evaluator
    evaluator = Evaluator(k_list=[1, 5, 10, 20])

    # 8. Training loop
    writer = None
    if args.tb_log_dir:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        print(f"TensorBoard: {args.tb_log_dir}")

    early_stopper = EarlyStopper(patience=args.patience)
    
    start_epoch = 1
    
    # Auto-resume from latest checkpoint if resume_path not set
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_checkpoint.msgpack")
    if not args.resume_path:
        latest_ckpt = os.path.join(args.checkpoint_dir, "latest_checkpoint.msgpack")
        if os.path.exists(latest_ckpt):
            args.resume_path = latest_ckpt

    if args.resume_path:
        print(f"Loading checkpoint from {args.resume_path}...")
        try:
            verify_semantic_ids_hash(args.resume_path, ids_hash)
            checkpoint_state = load_checkpoint(args.resume_path, unreplicate(params), unreplicate(opt_state))
            params = replicate(checkpoint_state["params"])
            opt_state = replicate(checkpoint_state["opt_state"])
            start_epoch = checkpoint_state["epoch"] + 1
            early_stopper.best_metrics["NDCG@10"] = float(checkpoint_state["best_val_ndcg"])
            # Restore true best-val params from the best checkpoint (not the latest
            # epoch) so a resumed run's final eval uses best-val weights.
            if os.path.exists(best_ckpt_path):
                early_stopper.best_params = load_checkpoint(best_ckpt_path, unreplicate(params), unreplicate(opt_state))["params"]
            else:
                early_stopper.best_params = unreplicate(params)
            print(f"Resumed from epoch {checkpoint_state['epoch']} with best validation NDCG@10 = {checkpoint_state['best_val_ndcg']:.5f}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")

    batch_size = args.batch_size
    num_samples = len(train_tar)
    epoch_rng = jax.random.PRNGKey(777)
    
    num_batches_total = num_samples // batch_size
    global_step = (start_epoch - 1) * num_batches_total

    class InMemoryDataSource:
        def __init__(self, enc, tar):
            self.enc = enc
            self.tar = tar
        def __len__(self):
            return len(self.enc)
        def __getitem__(self, idx):
            return self.enc[idx], self.tar[idx]

    source = InMemoryDataSource(train_enc_in, train_tar)
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=args.epochs - start_epoch + 1,
        shard_options=grain.NoSharding(),
        shuffle=True,
        seed=777,
    )
    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        # worker_count=0: in-memory numpy source; multiprocess workers add no
        # throughput and crash on grain/absl flag parsing in child processes.
        worker_count=0,
        worker_buffer_size=2,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=True)
        ]
    )

    print(f"\nTraining for {args.epochs} epochs...")
    
    iterator = iter(dataloader)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for _ in range(num_batches_total):
            batch_enc_np, batch_tar_np = next(iterator)
            epoch_rng, step_rng = jax.random.split(epoch_rng)

            # Shard data across devices: (batch, ...) -> (num_devices, batch/num_devices, ...)
            batch_enc_sharded = shard(batch_enc_np)
            batch_tar_sharded = shard(batch_tar_np)
            
            # Since step_rng needs to be unique per device, we can split it
            step_rngs = jax.random.split(step_rng, jax.local_device_count())

            params, opt_state, loss_val = train_step(
                params,
                opt_state,
                batch_enc_sharded,
                batch_tar_sharded,
                step_rngs
            )
            # loss_val is now replicated across devices, just take the first one
            epoch_loss += float(loss_val[0])
            num_batches += 1
            global_step += 1

            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("Loss/train_step", float(loss_val[0]), global_step)

        elapsed = time.time() - start_time
        avg_loss = float(epoch_loss) / num_batches
        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

        if writer is not None:
            writer.add_scalar("Loss/train_epoch", avg_loss, global_step)

        # Evaluate on validation split
        print(f"Evaluating validation split at epoch {epoch}...")
        def val_predict(batch_inputs):
            # Re-pad to be divisible by device_count if necessary
            num_devices = jax.local_device_count()
            pad_amount = (num_devices - (len(batch_inputs) % num_devices)) % num_devices
            if pad_amount > 0:
                batch_inputs = np.pad(batch_inputs, ((0, pad_amount), (0, 0)), mode="constant")
            
            batch_enc_sharded = shard(jnp.array(batch_inputs))
            preds_sharded = predict_step(params, batch_enc_sharded)
            # Unshard predictions
            preds = preds_sharded.reshape((-1, preds_sharded.shape[-1]))
            if pad_amount > 0:
                preds = preds[:-pad_amount]
            return preds

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
            params_to_save = unreplicate(params)
            opt_state_to_save = unreplicate(opt_state)
            
            checkpoint_path = save_checkpoint(
                params_to_save,
                opt_state_to_save,
                epoch,
                early_stopper.get_best("NDCG@10"),
                args.checkpoint_dir,
                "best_checkpoint.msgpack",
                semantic_ids_hash=ids_hash,
            )
            print(f"Checkpoint saved to {checkpoint_path}")
        elif early_stopper.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

        save_checkpoint(
            unreplicate(params), unreplicate(opt_state), epoch,
            early_stopper.get_best("NDCG@10"),
            args.checkpoint_dir, filename="latest_checkpoint.msgpack",
            semantic_ids_hash=ids_hash,
        )

    # 9. Final test evaluation
    best_params = early_stopper.best_params if early_stopper.best_params is not None else params
    num_devices = jax.local_device_count()

    # 9. Test Evaluation
    print("\nRunning final test evaluation...")
    def test_predict(batch_inputs):
        pad_amount = (num_devices - (len(batch_inputs) % num_devices)) % num_devices
        if pad_amount > 0:
            batch_inputs = np.pad(batch_inputs, ((0, pad_amount), (0, 0)), mode="constant")
        batch_enc_sharded = shard(jnp.array(batch_inputs))
        preds_sharded = predict_step(best_params, batch_enc_sharded)
        preds = preds_sharded.reshape((-1, preds_sharded.shape[-1]))
        if pad_amount > 0:
            preds = preds[:-pad_amount]
        return preds

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
