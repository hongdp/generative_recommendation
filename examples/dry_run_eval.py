"""Dry-run script to verify the HSTU model and evaluation framework on MovieLens-100K.

Instantiates HSTUModel, trains it for a few epochs using Optax on the training split,
and evaluates its performance on the test split.
"""

import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax

from generative_recommendation.datasets.movielens import MovieLensDataLoader
from generative_recommendation.evaluation.evaluator import Evaluator
from generative_recommendation.models.hstu import HSTUModel


def main():
    print("--- Starting Dry Run for HSTU Model and Evaluation Framework ---")

    # 1. Initialize data loader (downloads ML-100K if not present)
    data_dir = "./data"
    print(f"Loading MovieLens-100K dataset from {data_dir}...")
    loader = MovieLensDataLoader(dataset_name="ml-100k", data_dir=data_dir, min_rating=0)
    print(f"Dataset stats: Users = {loader.num_users}, Items = {loader.num_items}")

    # 2. Get splits (training and test)
    print("Generating train and test splits...")
    train_dataset = loader.get_split("train", max_len=10, format_type="index")
    train_inputs, train_targets = train_dataset.to_numpy()
    print(f"Train split: {len(train_targets)} samples")

    test_dataset = loader.get_split("test", max_len=10, format_type="index")
    test_inputs, test_targets = test_dataset.to_numpy()
    print(f"Test split: {len(test_targets)} samples")

    # 3. Instantiate HSTU Model
    print("Initializing HSTU Model...")
    model = HSTUModel(
        num_items=loader.num_items,
        embedding_dim=64,
        num_blocks=2,
        num_heads=2,
        attention_dim=32,
        linear_dim=128,
        max_sequence_len=10,
    )

    # Initialize JAX variables
    key = jax.random.PRNGKey(42)
    dummy_seq = jnp.zeros((1, 10), dtype=jnp.int32)
    variables = model.init(key, dummy_seq)
    params = variables["params"]

    # 4. Set up Optax Optimizer and Train Step
    learning_rate = 0.005
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_inputs, batch_targets):
        def loss_fn(p):
            # logits shape: [batch, seq_len, num_items + 1]
            logits = model.apply({"params": p}, batch_inputs, deterministic=True)
            # Evaluate next-item prediction on the last sequence position
            logits_last = logits[:, -1, :]
            # Cross-entropy loss
            loss_vals = optax.softmax_cross_entropy_with_integer_labels(logits_last, batch_targets)
            return jnp.mean(loss_vals)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # JIT-compiled predict function for evaluation
    @jax.jit
    def predict_fn(params, batch_inputs):
        # We need the scores over all items for the last sequence position
        logits = model.apply({"params": params}, batch_inputs, deterministic=True)
        return logits[:, -1, :]

    # Wrapper to pass to evaluator
    def hstu_eval_predict(batch_inputs):
        return predict_fn(params, batch_inputs)

    # 5. Evaluate BEFORE Training (Randomly initialized weights)
    print("\nEvaluating HSTU model before training...")
    evaluator = Evaluator(k_list=[1, 5, 10])
    pre_results = evaluator.evaluate_index_based(
        hstu_eval_predict, test_inputs, test_targets, batch_size=128
    )
    print("--- Pre-training Evaluation Results ---")
    for metric, score in pre_results.items():
        print(f"{metric}: {score:.5f}")

    # 6. Train the HSTU Model
    epochs = 5
    batch_size = 128
    num_samples = len(train_targets)
    print(f"\nTraining HSTU Model for {epochs} epochs (batch_size={batch_size})...")

    for epoch in range(1, epochs + 1):
        # Shuffle training data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        shuffled_inputs = train_inputs[indices]
        shuffled_targets = train_targets[indices]

        epoch_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        for i in range(0, num_samples, batch_size):
            batch_in = shuffled_inputs[i : i + batch_size]
            batch_tar = shuffled_targets[i : i + batch_size]
            
            # Update parameters
            params, opt_state, loss_val = train_step(params, opt_state, batch_in, batch_tar)
            epoch_loss += float(loss_val)
            num_batches += 1
            
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

    # 7. Evaluate AFTER Training
    print("\nEvaluating HSTU model after training...")
    post_results = evaluator.evaluate_index_based(
        hstu_eval_predict, test_inputs, test_targets, batch_size=128
    )
    print("--- Post-training Evaluation Results ---")
    for metric, score in post_results.items():
        print(f"{metric}: {score:.5f}")

    # Verify that metrics improve after training
    mrr_diff = post_results["MRR"] - pre_results["MRR"]
    print(f"\nMRR Improvement: {mrr_diff:+.5f}")
    if mrr_diff > 0.0:
        print("Model converged successfully and improved recommendation rankings!")
    else:
        print("Warning: Model performance did not improve. Check hyperparameters.")

    print("\n--- Dry Run and Replicating HSTU Completed Successfully! ---")


if __name__ == "__main__":
    main()
