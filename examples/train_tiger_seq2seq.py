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

from datasets import MovieLensDataLoader, AmazonDataLoader, SteamDataLoader
from models.tiger_seq2seq import TIGERSeq2SeqModel
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
            # Write to position shifting padding to the left
            write_pos = 3 * num_pad + 3 * idx
            encoder_inputs[i, write_pos] = c1 + 1
            encoder_inputs[i, write_pos + 1] = c2 + K + 1
            encoder_inputs[i, write_pos + 2] = c3 + 2 * K + 1
            
    return encoder_inputs


def preprocess_seq2seq_training_data(inputs, targets, semantic_ids, K, start_token):
    """Formats inputs and targets into Seq2Seq encoder inputs, decoder inputs, and targets."""
    batch_size = len(inputs)
    max_len = inputs.shape[1]
    
    # 1. encoder_inputs
    encoder_inputs = np.zeros((batch_size, 3 * max_len), dtype=np.int32)
    for i in range(batch_size):
        seq = inputs[i]
        non_pad_indices = np.where(seq != 0)[0]
        num_pad = max_len - len(non_pad_indices)
        for idx, pos in enumerate(non_pad_indices):
            item = seq[pos]
            c1, c2, c3 = semantic_ids[item]
            write_pos = 3 * num_pad + 3 * idx
            encoder_inputs[i, write_pos] = c1 + 1
            encoder_inputs[i, write_pos + 1] = c2 + K + 1
            encoder_inputs[i, write_pos + 2] = c3 + 2 * K + 1
            
    # 2. decoder_inputs and decoder_targets
    decoder_inputs = np.zeros((batch_size, 3), dtype=np.int32)
    decoder_targets = np.zeros((batch_size, 3), dtype=np.int32)
    decoder_inputs[:, 0] = start_token
    
    for i in range(batch_size):
        tar = targets[i]
        c1, c2, c3 = semantic_ids[tar]
        # decoder inputs: [start, c1+1, c2+K+1]
        decoder_inputs[i, 1] = c1 + 1
        decoder_inputs[i, 2] = c2 + K + 1
        # decoder targets: [c1+1, c2+K+1, c3+2*K+1]
        decoder_targets[i, 0] = c1 + 1
        decoder_targets[i, 1] = c2 + K + 1
        decoder_targets[i, 2] = c3 + 2 * K + 1
        
    return encoder_inputs, decoder_inputs, decoder_targets


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
    
    with open(ids_path, "r") as f:
        semantic_ids = {int(k): v for k, v in json.load(f).items()}

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
    train_enc_in, train_dec_in, train_dec_tar = preprocess_seq2seq_training_data(
        train_in, train_tar, semantic_ids, K, start_token
    )
    print(f"Train split: {len(train_dec_tar)} samples")

    val_dataset = loader.get_split("val", max_len=max_len, format_type="index")
    val_in, val_tar = val_dataset.to_numpy()
    val_enc_in = sequence_to_tiger_tokens(val_in, semantic_ids, K)

    test_dataset = loader.get_split("test", max_len=max_len, format_type="index")
    test_in, test_tar = test_dataset.to_numpy()
    test_enc_in = sequence_to_tiger_tokens(test_in, semantic_ids, K)

    # 3. Setup Model
    print("Initializing TIGER Seq2Seq Model...")
    model = TIGERSeq2SeqModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        max_encoder_len=3 * max_len + 4,
        max_decoder_len=4,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
    )

    key = jax.random.PRNGKey(42)
    dummy_enc = jnp.zeros((1, 3 * max_len), dtype=jnp.int32)
    dummy_dec = jnp.zeros((1, 3), dtype=jnp.int32)
    variables = model.init(key, dummy_enc, dummy_dec)
    params = variables["params"]

    # 4. Setup Optimizer
    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
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

    @jax.jit
    def predict_dec_step_beams(params, decoder_tokens_beams, encoder_outputs, encoder_tokens):
        # Vectorize across the beam dimension (axis 1)
        # decoder_tokens_beams: [batch_size, B, dec_len]
        # encoder_outputs: [batch_size, enc_len, dim]
        # encoder_tokens: [batch_size, enc_len]
        vmap_fn = jax.vmap(
            lambda dec: model.apply(
                {"params": params},
                dec,
                encoder_outputs,
                encoder_tokens,
                method=model.decode_step,
                deterministic=True,
            ),
            in_axes=1,
            out_axes=1
        )
        return vmap_fn(decoder_tokens_beams)

    # 7. Batched Beam Search decoder
    def beam_search_decode(params, batch_enc_in, B=10):
        batch_size = len(batch_enc_in)
        
        # Step 0: Encode the user sequences to get encoder outputs
        encoder_outputs = predict_enc(params, batch_enc_in)
        
        # Step 1: Decode Level 1 token
        # Initialize decoder inputs with start token: shape [batch_size, 1]
        dec_in1 = jnp.ones((batch_size, 1), dtype=jnp.int32) * start_token
        logits1 = predict_dec_step(params, dec_in1, encoder_outputs, batch_enc_in)
        
        # Log-probs for Level 1 tokens (indices 1 to 256)
        log_probs1 = jax.nn.log_softmax(logits1[:, 1 : 257], axis=-1)
        top_probs, top_indices = jax.lax.top_k(log_probs1, k=B)  # [batch_size, B]
        top_tokens1 = top_indices + 1

        # Step 2: Decode Level 2 token (Batched across beams)
        dec_in2_start = jnp.ones((batch_size, B, 1), dtype=jnp.int32) * start_token
        dec_in2 = jnp.concatenate([dec_in2_start, top_tokens1[:, :, None]], axis=-1) # [batch_size, B, 2]
        
        logits2 = predict_dec_step_beams(params, dec_in2, encoder_outputs, batch_enc_in)
        log_probs2 = jax.nn.log_softmax(logits2[:, :, 257 : 513], axis=-1)  # [batch_size, B, 256]
        
        # Cumulative probability
        cum_probs2 = top_probs[:, :, None] + log_probs2  # [batch_size, B, 256]
        cum_probs2 = cum_probs2.reshape(batch_size, -1)  # [batch_size, B * 256]
        top_probs2, top_flat_indices2 = jax.lax.top_k(cum_probs2, k=B)  # [batch_size, B]
        
        beam_idx2 = top_flat_indices2 // 256
        c2 = top_flat_indices2 % 256
        c1 = top_indices[jnp.arange(batch_size)[:, None], beam_idx2]
        
        top_tokens1_expanded = c1 + 1
        top_tokens2 = c2 + 257

        # Step 3: Decode Level 3 token
        dec_in3_start = jnp.ones((batch_size, B, 1), dtype=jnp.int32) * start_token
        dec_in3 = jnp.concatenate([
            dec_in3_start,
            top_tokens1_expanded[:, :, None],
            top_tokens2[:, :, None]
        ], axis=-1) # [batch_size, B, 3]
        
        logits3 = predict_dec_step_beams(params, dec_in3, encoder_outputs, batch_enc_in)
        log_probs3 = jax.nn.log_softmax(logits3[:, :, 513 : 769], axis=-1)  # [batch_size, B, 256]
        
        cum_probs3 = top_probs2[:, :, None] + log_probs3  # [batch_size, B, 256]
        cum_probs3 = cum_probs3.reshape(batch_size, -1)  # [batch_size, B * 256]
        top_probs3, top_flat_indices3 = jax.lax.top_k(cum_probs3, k=B)  # [batch_size, B]
        
        beam_idx3 = top_flat_indices3 // 256
        c3 = top_flat_indices3 % 256
        c2_final = c2[jnp.arange(batch_size)[:, None], beam_idx3]
        c1_final = c1[jnp.arange(batch_size)[:, None], beam_idx3]

        return np.array(c1_final), np.array(c2_final), np.array(c3)

    # 8. Evaluation setup
    semantic_id_to_item = {tuple(v): k for k, v in semantic_ids.items()}
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
    
    # Auto-resume from latest checkpoint if resume_path not set
    if not args.resume_path:
        latest_ckpt = os.path.join(args.checkpoint_dir, "latest_checkpoint.msgpack")
        if os.path.exists(latest_ckpt):
            args.resume_path = latest_ckpt

    if args.resume_path:
        print(f"Loading checkpoint from {args.resume_path}...")
        try:
            checkpoint_state = load_checkpoint(args.resume_path, unreplicate(params), unreplicate(opt_state))
            params = replicate(checkpoint_state["params"])
            opt_state = replicate(checkpoint_state["opt_state"])
            start_epoch = checkpoint_state["epoch"] + 1
            early_stopper.best_metrics["NDCG@10"] = float(checkpoint_state["best_val_ndcg"])
            early_stopper.best_params = unreplicate(params)
            print(f"Resumed from epoch {checkpoint_state['epoch']} with best validation NDCG@10 = {checkpoint_state['best_val_ndcg']:.5f}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")

    if args.eval_only:
        if not args.resume_path:
            raise ValueError("Must specify --resume_path when using --eval_only.")
        print("\nRunning test evaluation only...")
        best_params = early_stopper.best_params if early_stopper.best_params is not None else unreplicate(params)
        def test_predict(batch_inputs):
            return beam_search_decode(best_params, jnp.array(batch_inputs), B=beam_size)
        test_results = evaluator.evaluate_generative_discrete(
            test_predict, semantic_id_to_item, test_enc_in, test_tar,
            beam_size=beam_size, batch_size=args.batch_size,
        )
        print("\n--- Test Evaluation Results ---")
        for metric, score in test_results.items():
            print(f"{metric}: {score:.5f}")
        return

    # 10. Training Loop
    epochs = args.epochs
    batch_size = args.batch_size
    num_samples = len(train_dec_tar)
    epoch_rng = jax.random.PRNGKey(777)

    num_batches = num_samples // batch_size
    global_step = (start_epoch - 1) * num_batches

    print(f"\nTraining TIGER Seq2Seq model for {epochs} epochs starting from epoch {start_epoch}...")
    for epoch in range(start_epoch, epochs + 1):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_enc_in = train_enc_in[indices]
        shuffled_dec_in = train_dec_in[indices]
        shuffled_dec_tar = train_dec_tar[indices]

        epoch_loss = 0.0
        num_batches_processed = 0
        start_time = time.time()
        
        for i in range(0, num_samples, batch_size):
            if i + batch_size > num_samples:
                break
                
            batch_enc = shard(jnp.array(shuffled_enc_in[i : i + batch_size]))
            batch_dec_in = shard(jnp.array(shuffled_dec_in[i : i + batch_size]))
            batch_dec_tar = shard(jnp.array(shuffled_dec_tar[i : i + batch_size]))
            
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

        # Evaluate on validation split every epoch
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
                args.checkpoint_dir,
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
