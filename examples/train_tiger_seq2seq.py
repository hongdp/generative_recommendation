"""Full-scale training and evaluation script for the TIGER Seq2Seq (Encoder-Decoder) recommendation model."""

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
import flax.serialization
from flax.jax_utils import replicate, unreplicate
from flax.training.common_utils import shard
import grain.python as grain
import sys
from absl import flags
try:
    flags.FLAGS(sys.argv)
except Exception:
    pass

from datasets import MovieLensDataLoader, AmazonDataLoader, SteamDataLoader
from models.tiger_seq2seq import TIGERSeq2SeqModel
from models.tiger_tokenization import (
    load_semantic_ids,
    semantic_ids_hash,
    build_semantic_id_to_item,
    sequence_to_encoder_tokens,
    preprocess_seq2seq_training_data,
)
from evaluation.evaluator import Evaluator
from evaluation.tiger_decode import make_seq2seq_predictors, beam_search_decode_seq2seq
from evaluation.training_utils import (
    EarlyStopper,
    save_checkpoint,
    load_checkpoint,
    log_results_to_markdown,
    verify_semantic_ids_hash,
    assert_decode_validity,
)


def main():
    parser = argparse.ArgumentParser(description="TIGER Seq2Seq training and evaluation on recommendation datasets.")
    parser.add_argument("--checkpoint_dir", type=str, default="./data/tiger_seq2seq_checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--resume_path", type=str, default="", help="Path to checkpoint to resume training or evaluate.")
    parser.add_argument("--eval_only", action="store_true", help="Only run test set evaluation using --resume_path.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--tb_log_dir", type=str, default="./data/tensorboard/tiger_seq2seq_ml1m", help="TensorBoard log directory.")
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
    parser.add_argument("--num_levels", type=int, default=3, help="Number of Semantic-ID levels (L).")
    parser.add_argument("--num_codes", type=int, default=256, help="Codebook size per level (K).")
    parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine"], help="LR schedule.")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps for cosine schedule.")
    parser.add_argument("--eval_every", type=int, default=1, help="Run validation every N epochs.")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    if args.checkpoint_dir == "./data/tiger_seq2seq_checkpoints" and dataset != "ml-1m":
        args.checkpoint_dir = f"./data/tiger_seq2seq_{dataset}_checkpoints"
    if args.tb_log_dir == "./data/tensorboard/tiger_seq2seq_ml1m" and dataset != "ml-1m":
        args.tb_log_dir = f"./data/tensorboard/tiger_seq2seq_{dataset}"
    if args.semantic_ids_path == "./data/semantic_ids.json" and dataset != "ml-1m":
        args.semantic_ids_path = f"./data/semantic_ids_{dataset}.json"

    print(f"--- Replicating TIGER Seq2Seq Results on {args.dataset.upper()} ---")
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
    K = args.num_codes  # Codebook size per level
    L = args.num_levels  # Number of Semantic-ID levels
    vocab_size = L * K + 2  # level i: [i*K+1, (i+1)*K], start: L*K+1, pad: 0
    start_token = vocab_size - 1
    max_len = 20 if dataset in ["beauty", "sports", "toys", "steam"] else 50
    beam_size = 20

    # 2. Get splits and format to JAX/TIGER tokens
    print("Preprocessing datasets into TIGER tokens...")
    train_dataset = loader.get_split("train", max_len=max_len, format_type="index")
    train_in, train_tar = train_dataset.to_numpy()
    train_enc_in, train_dec_in, train_dec_tar = preprocess_seq2seq_training_data(
        train_in, train_tar, semantic_ids, K, start_token, num_levels=L
    )
    print(f"Train split: {len(train_dec_tar)} samples")

    val_dataset = loader.get_split("val", max_len=max_len, format_type="index")
    val_in, val_tar = val_dataset.to_numpy()
    val_enc_in = sequence_to_encoder_tokens(val_in, semantic_ids, K, num_levels=L)

    test_dataset = loader.get_split("test", max_len=max_len, format_type="index")
    test_in, test_tar = test_dataset.to_numpy()
    test_enc_in = sequence_to_encoder_tokens(test_in, semantic_ids, K, num_levels=L)

    # 3. Setup Model
    print("Initializing TIGER Seq2Seq Model...")
    model = TIGERSeq2SeqModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        max_encoder_len=L * max_len + L + 1,
        max_decoder_len=L + 1,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
    )

    key = jax.random.PRNGKey(42)
    dummy_enc = jnp.zeros((1, L * max_len), dtype=jnp.int32)
    dummy_dec = jnp.zeros((1, L), dtype=jnp.int32)
    variables = model.init(key, dummy_enc, dummy_dec)
    params = variables["params"]

    # 4. Setup Optimizer
    if args.lr_schedule == "cosine":
        steps_per_epoch = max(1, len(train_dec_tar) // args.batch_size)
        total_steps = steps_per_epoch * args.epochs
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=args.learning_rate,
            warmup_steps=args.warmup_steps, decay_steps=total_steps, end_value=args.learning_rate * 0.02)
        print(f"LR schedule: cosine (warmup {args.warmup_steps}, total {total_steps} steps)")
    else:
        lr = args.learning_rate
    optimizer = optax.adamw(learning_rate=lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    params = replicate(params)
    opt_state = replicate(opt_state)

    # 5. Define training step
    @functools.partial(jax.pmap, axis_name="batch")
    def train_step(params, opt_state, batch_enc, batch_dec_in, batch_dec_tar, dropout_key):
        def loss_fn(p):
            logits = model.apply(
                {"params": p},
                batch_enc,
                batch_dec_in,
                rngs={"dropout": dropout_key},
                deterministic=False,
            )
            # Cross entropy loss across the 3 positions of the decoder
            loss_vals = optax.softmax_cross_entropy_with_integer_labels(logits, batch_dec_tar)
            return jnp.mean(loss_vals)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        loss = lax.pmean(loss, axis_name="batch")
        grads = lax.pmean(grads, axis_name="batch")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # 6. Beam search decoding (shared implementation, parametrized by codebook K)
    predictors = make_seq2seq_predictors(model)

    def beam_search_decode(params, batch_enc_in, B=10):
        return beam_search_decode_seq2seq(params, batch_enc_in, predictors, start_token, K, B=B, num_levels=L)

    # 8. Evaluation setup
    semantic_id_to_item = build_semantic_id_to_item(semantic_ids)
    evaluator = Evaluator(k_list=[1, 5, 10, 20])

    # 9. Resume training setup
    writer = None
    if args.tb_log_dir and not args.eval_only:
        # Removed inline import to prevent UnboundLocalError
        # Removed inline import to prevent UnboundLocalError
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        print(f"TensorBoard logging enabled. Logs saved to {args.tb_log_dir}")

    start_epoch = 1
    early_stopper = EarlyStopper(patience=args.patience)

    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_checkpoint.msgpack")
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.msgpack")

    if args.eval_only:
        # Evaluate the best-val checkpoint (explicit --resume_path overrides).
        eval_ckpt = args.resume_path or best_ckpt_path
        if not os.path.exists(eval_ckpt):
            raise ValueError(f"No checkpoint to evaluate at {eval_ckpt}.")
        print(f"Loading checkpoint for evaluation from {eval_ckpt}...")
        verify_semantic_ids_hash(eval_ckpt, ids_hash)
        checkpoint_state = load_checkpoint(eval_ckpt, unreplicate(params), unreplicate(opt_state))
        best_params = checkpoint_state["params"]
        print(f"Loaded checkpoint (epoch {checkpoint_state['epoch']}, "
              f"best val NDCG@10={checkpoint_state['best_val_ndcg']:.5f}).")
        def test_predict(batch_inputs):
            return beam_search_decode(best_params, jnp.array(batch_inputs), B=beam_size)
        test_results = evaluator.evaluate_generative_discrete(
            test_predict, semantic_id_to_item, test_enc_in, test_tar,
            beam_size=beam_size, batch_size=args.batch_size,
        )
        assert_decode_validity(test_results)
        print("\n--- Test Evaluation Results ---")
        for metric, score in test_results.items():
            print(f"{metric}: {score:.5f}")
        return

    # Training auto-resume: continue from the latest checkpoint if present.
    if not args.resume_path and os.path.exists(latest_ckpt_path):
        args.resume_path = latest_ckpt_path

    if args.resume_path:
        print(f"Loading checkpoint from {args.resume_path}...")
        try:
            verify_semantic_ids_hash(args.resume_path, ids_hash)
            checkpoint_state = load_checkpoint(args.resume_path, unreplicate(params), unreplicate(opt_state))
            params = replicate(checkpoint_state["params"])
            opt_state = replicate(checkpoint_state["opt_state"])
            start_epoch = checkpoint_state["epoch"] + 1
            early_stopper.best_metrics["NDCG@10"] = float(checkpoint_state["best_val_ndcg"])
            # Restore the true best-val params from the best checkpoint so a resumed
            # run's final test eval uses best-val weights, not the latest epoch.
            if os.path.exists(best_ckpt_path):
                best_state = load_checkpoint(best_ckpt_path, unreplicate(params), unreplicate(opt_state))
                early_stopper.best_params = best_state["params"]
            else:
                early_stopper.best_params = unreplicate(params)
            print(f"Resumed from epoch {checkpoint_state['epoch']} with best validation NDCG@10 = {checkpoint_state['best_val_ndcg']:.5f}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")

    # 10. Training Loop
    epochs = args.epochs
    batch_size = args.batch_size
    num_samples = len(train_dec_tar)
    epoch_rng = jax.random.PRNGKey(777)

    num_batches = num_samples // batch_size
    global_step = (start_epoch - 1) * num_batches

    class InMemoryDataSource:
        def __init__(self, enc, dec_in, dec_tar):
            self.enc = enc
            self.dec_in = dec_in
            self.dec_tar = dec_tar
        def __len__(self):
            return len(self.enc)
        def __getitem__(self, idx):
            return self.enc[idx], self.dec_in[idx], self.dec_tar[idx]

    source = InMemoryDataSource(train_enc_in, train_dec_in, train_dec_tar)
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=epochs - start_epoch + 1,
        shard_options=grain.NoSharding(),
        shuffle=True,
        seed=777,
    )
    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        # worker_count=0: the source is already in-memory numpy, so multiprocess
        # workers add no throughput and trigger a grain/absl flags-parsing crash in
        # child processes (UnparsedFlagAccessError). See test_grain2.py.
        worker_count=0,
        worker_buffer_size=2,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=True)
        ]
    )

    print(f"\nTraining TIGER Seq2Seq model for {epochs} epochs starting from epoch {start_epoch}...")
    
    iterator = iter(dataloader)
    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = 0.0
        num_batches_processed = 0
        start_time = time.time()
        
        for _ in range(num_batches):
            batch_enc_np, batch_dec_in_np, batch_dec_tar_np = next(iterator)
            batch_enc = shard(batch_enc_np)
            batch_dec_in = shard(batch_dec_in_np)
            batch_dec_tar = shard(batch_dec_tar_np)
            
            epoch_rng, step_rng = jax.random.split(epoch_rng)
            step_rngs = jax.random.split(step_rng, jax.local_device_count())
            
            params, opt_state, loss_val = train_step(
                params,
                opt_state,
                batch_enc,
                batch_dec_in,
                batch_dec_tar,
                step_rngs
            )
            epoch_loss += float(loss_val[0])
            num_batches_processed += 1
            global_step += 1

            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("Loss/train_step", float(loss_val[0]), global_step)

        elapsed = time.time() - start_time
        avg_loss = float(epoch_loss) / num_batches_processed
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        
        if writer is not None:
            writer.add_scalar("Loss/train_epoch", avg_loss, global_step)

        # Evaluate on validation split every --eval_every epochs
        if epoch % args.eval_every != 0:
            save_checkpoint(
                unreplicate(params), unreplicate(opt_state), epoch,
                early_stopper.get_best("NDCG@10"),
                args.checkpoint_dir, filename="latest_checkpoint.msgpack",
                semantic_ids_hash=ids_hash,
            )
            continue
        print(f"Evaluating validation split at epoch {epoch}...")
        val_params = unreplicate(params)
        def val_predict(batch_inputs):
            return beam_search_decode(val_params, jnp.array(batch_inputs), B=beam_size)
            
        val_results = evaluator.evaluate_generative_discrete(
            val_predict, semantic_id_to_item, val_enc_in, val_tar,
            beam_size=beam_size, batch_size=args.batch_size,
        )
        val_ndcg = val_results["NDCG@10"]
        val_hr = val_results["HR@10"]
        val_mrr = val_results["MRR"]
        print(f"--- Validation @ Epoch {epoch} | NDCG@10: {val_ndcg:.5f} | HR@10: {val_hr:.5f} | MRR: {val_mrr:.5f}")

        if writer is not None:
            for metric, score in val_results.items():
                writer.add_scalar(f"Val/{metric}", score, global_step)

        improved = early_stopper.check(val_results, unreplicate(params))
        if improved:
            print(">>> New best validation score! Saving checkpoint...")
            ckpt_path = save_checkpoint(
                unreplicate(params), unreplicate(opt_state), epoch,
                early_stopper.get_best("NDCG@10"),
                args.checkpoint_dir, semantic_ids_hash=ids_hash,
            )
            print(f"Checkpoint saved to {ckpt_path}")
        elif early_stopper.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

        # Save latest checkpoint
        save_checkpoint(
            unreplicate(params), unreplicate(opt_state), epoch,
            early_stopper.get_best("NDCG@10"),
            args.checkpoint_dir, filename="latest_checkpoint.msgpack",
            semantic_ids_hash=ids_hash,
        )

    # 11. Final Test evaluation using best checkpoint
    best_params = early_stopper.best_params if early_stopper.best_params is not None else unreplicate(params)

    print("\nRunning final test evaluation...")
    def test_predict(batch_inputs):
        return beam_search_decode(best_params, jnp.array(batch_inputs), B=beam_size)
    test_results = evaluator.evaluate_generative_discrete(
        test_predict, semantic_id_to_item, test_enc_in, test_tar,
        beam_size=beam_size, batch_size=args.batch_size,
    )
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
    model_desc = f"TIGER Seq2Seq (blocks={args.num_blocks}, embed={args.embedding_dim})"
    log_results_to_markdown(
        model_desc, args.dataset, test_results,
        early_stopper.get_best("NDCG@10"),
    )


if __name__ == "__main__":
    main()
