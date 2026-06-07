"""Script to train RQ-VAE on MovieLens-1M title embeddings and export Semantic IDs."""

import argparse
import os
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax

from datasets import MovieLensDataLoader, AmazonDataLoader, SteamDataLoader
from datasets.embeddings import extract_movie_embeddings
from models.rqvae import RQVAE


def main():
    parser = argparse.ArgumentParser(description="Train RQ-VAE on sequential recommendation datasets.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for data.")
    parser.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "beauty", "sports", "toys", "steam"], help="Dataset name.")
    args = parser.parse_args()

    print(f"--- Training RQ-VAE for TIGER Semantic IDs on {args.dataset} ---")
    print("Device list:", jax.devices())
    device = "cuda" if jax.local_devices()[0].platform == "gpu" else "cpu"
    print(f"Using device: {device}")

    # 1. Load dataset to get item mapping and titles
    data_dir = args.data_dir
    dataset = args.dataset.lower()
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

    # 2. Extract title embeddings
    # Using lightweight sentence transformer all-MiniLM-L6-v2
    # embeddings shape: (loader.num_items + 1, embedding_dim)
    embeddings = extract_movie_embeddings(
        loader.token_to_title,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=256,
        device="cuda" if device == "cuda" else "cpu"
    )
    print(f"Movie embeddings shape: {embeddings.shape}")

    # 3. Setup RQ-VAE Model
    latent_dim = 32
    num_levels = 3
    num_codes = 256
    embedding_dim = embeddings.shape[-1]
    
    print(f"Initializing RQ-VAE (latent={latent_dim}, levels={num_levels}, codes={num_codes})...")
    model = RQVAE(
        latent_dim=latent_dim,
        num_levels=num_levels,
        num_codes=num_codes,
        embedding_dim=embedding_dim,
        commitment_weight=0.25,
    )

    key = jax.random.PRNGKey(1234)
    dummy_input = jnp.zeros((1, embedding_dim))
    variables = model.init(key, dummy_input)
    params = variables["params"]

    # 4. Set up Optimizer
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    # 5. Define JIT training step
    @jax.jit
    def train_step(params, opt_state, batch_x):
        def loss_fn(p):
            outputs = model.apply({"params": p}, batch_x)
            recon_loss = jnp.mean((outputs["x_recon"] - batch_x) ** 2)
            codebook_loss = outputs["codebook_loss"]
            commitment_loss = outputs["commitment_loss"]
            total_loss = recon_loss + codebook_loss + model.commitment_weight * commitment_loss
            return total_loss, (recon_loss, codebook_loss, commitment_loss)

        (loss, (recon_l, code_l, commit_l)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, recon_l, code_l, commit_l

    # 6. Training loop
    epochs = args.epochs
    batch_size = 256
    num_samples = len(embeddings)
    
    # Exclude the padding token embedding (index 0) during training
    train_x = embeddings[1:] 
    num_train_samples = len(train_x)

    print(f"Training RQ-VAE on {num_train_samples} items for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Shuffle inputs
        indices = np.arange(num_train_samples)
        np.random.shuffle(indices)
        shuffled_x = train_x[indices]

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_code = 0.0
        epoch_commit = 0.0
        num_batches = 0

        for i in range(0, num_train_samples, batch_size):
            batch_x = shuffled_x[i : i + batch_size]
            # Convert to jax array
            batch_x_jax = jnp.array(batch_x)
            params, opt_state, loss_val, recon_val, code_val, commit_val = train_step(
                params, opt_state, batch_x_jax
            )
            
            # Accumulate scalars asynchronously
            epoch_loss += loss_val
            epoch_recon += recon_val
            epoch_code += code_val
            epoch_commit += commit_val
            num_batches += 1

        if epoch % 20 == 0 or epoch == 1:
            avg_loss = float(epoch_loss) / num_batches
            avg_recon = float(epoch_recon) / num_batches
            avg_code = float(epoch_code) / num_batches
            avg_commit = float(epoch_commit) / num_batches
            print(
                f"Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.5f} | "
                f"Recon MSE: {avg_recon:.5f} | Quant Code: {avg_code:.5f} | Commit: {avg_commit:.5f}"
            )

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")

    # 7. Generate Semantic IDs for all items (including padding 0)
    print("Generating discrete Semantic IDs for all items...")
    # Convert full embedding matrix to jax array
    full_x_jax = jnp.array(embeddings)
    indices_jax = model.apply({"params": params}, full_x_jax, method=model.encode)
    indices = np.array(indices_jax)

    # 8. Compute and print codebook utilization metrics
    print("\n--- Codebook Utilization ---")
    for c in range(num_levels):
        # Exclude padding token at index 0
        level_codes = indices[1:, c]
        unique_codes = len(np.unique(level_codes))
        print(f"Level {c+1} codebook usage: {unique_codes} / {num_codes} codes used.")

    # 9. Save Semantic IDs to JSON
    semantic_ids_dict = {
        str(i): [int(x) for x in indices[i]]
        for i in range(len(indices))
    }
    
    output_filename = "semantic_ids.json" if dataset == "ml-1m" else f"semantic_ids_{dataset}.json"
    output_path = os.path.join(data_dir, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(semantic_ids_dict, f, indent=2)
    print(f"\nSemantic IDs successfully written to {output_path}!")


if __name__ == "__main__":
    main()
