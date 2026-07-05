"""Script to run RQ-KMeans on MovieLens-1M title embeddings and export Semantic IDs."""

import argparse
import os
import json
import time
import numpy as np

from datasets import MovieLensDataLoader, AmazonDataLoader, SteamDataLoader
from datasets.embeddings import extract_movie_embeddings


def kmeans(X, K, max_iter=30, tol=1e-4):
    """NumPy implementation of K-Means clustering."""
    N, D = X.shape
    # Initialize centroids randomly from X
    indices = np.random.choice(N, K, replace=False)
    centroids = X[indices].copy()
    
    for iteration in range(max_iter):
        # Compute pairwise squared Euclidean distances: shape [N, K]
        X_sq = np.sum(X**2, axis=1, keepdims=True)  # [N, 1]
        c_sq = np.sum(centroids**2, axis=1, keepdims=True).T  # [1, K]
        dists = X_sq + c_sq - 2 * np.dot(X, centroids.T)  # [N, K]
        
        # Assign each sample to the nearest centroid
        labels = np.argmin(dists, axis=1)
        
        # Recompute centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            members = X[labels == k]
            if len(members) > 0:
                new_centroids[k] = np.mean(members, axis=0)
            else:
                # Reinitialize empty cluster with a random sample
                new_centroids[k] = X[np.random.choice(N)]
                
        # Check convergence
        diff = np.max(np.abs(centroids - new_centroids))
        if diff < tol:
            centroids = new_centroids
            break
        centroids = new_centroids
        
    return centroids, labels


def main():
    parser = argparse.ArgumentParser(description="Run RQ-KMeans on sequential recommendation datasets.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for data.")
    parser.add_argument("--num_levels", type=int, default=3, help="Number of quantization levels.")
    parser.add_argument("--num_codes", type=int, default=256, help="Codebook size (K) per level.")
    parser.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "beauty", "sports", "toys", "steam"], help="Dataset name.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible cluster assignments.")
    args = parser.parse_args()

    # RQ-KMeans centroid initialization uses np.random; seed it so the exported
    # Semantic IDs are reproducible. Without this, every regeneration yields a
    # different code assignment, silently breaking any checkpoint trained on the
    # previous IDs (see assert_decode_validity / verify_semantic_ids_hash).
    np.random.seed(args.seed)

    print(f"--- Running RQ-KMeans for TIGER Semantic IDs on {args.dataset} (seed={args.seed}) ---")

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

    # 2. Extract title embeddings (CPU execution to avoid GPU contention)
    embeddings = extract_movie_embeddings(
        loader.token_to_title,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=256,
        device="cpu"
    )
    print(f"Movie embeddings shape: {embeddings.shape}")

    # Exclude padding token at index 0 during clustering
    train_x = embeddings[1:].copy()
    num_train_samples = len(train_x)
    num_levels = args.num_levels
    num_codes = args.num_codes

    print(f"\nRunning Residual Quantization K-Means (levels={num_levels}, codes={num_codes}) on {num_train_samples} items...")
    
    start_time = time.time()
    residuals = train_x.copy()
    all_labels = []
    all_centroids = []

    for level in range(num_levels):
        print(f"Level {level + 1} clustering...")
        centroids, labels = kmeans(residuals, num_codes, max_iter=30)
        
        # Calculate residuals for the next level
        recon = centroids[labels]
        residuals = residuals - recon
        
        all_labels.append(labels)
        all_centroids.append(centroids)

    elapsed = time.time() - start_time
    print(f"RQ-KMeans completed in {elapsed:.2f} seconds.")

    # Shape: [num_train_samples, num_levels]
    indices_train = np.stack(all_labels, axis=1)

    # Reconstruction MSE calculation
    total_recon = np.zeros_like(train_x)
    for level in range(num_levels):
        total_recon += all_centroids[level][indices_train[:, level]]
    recon_mse = np.mean((train_x - total_recon) ** 2)
    print(f"\nReconstruction MSE: {recon_mse:.5f}")

    # 3. Create full indices matrix including padding item 0 at index 0
    full_indices = np.zeros((len(embeddings), num_levels), dtype=np.int32)
    full_indices[1:] = indices_train

    # 4. Compute and print codebook utilization metrics
    print("\n--- Codebook Utilization ---")
    for c in range(num_levels):
        # Exclude padding token at index 0
        level_codes = full_indices[1:, c]
        unique_codes = len(np.unique(level_codes))
        print(f"Level {c+1} codebook usage: {unique_codes} / {num_codes} codes used.")

    # 5. Save Semantic IDs to JSON
    semantic_ids_dict = {
        str(i): [int(x) for x in full_indices[i]]
        for i in range(len(full_indices))
    }
    
    output_filename = "semantic_ids_kmeans.json" if dataset == "ml-1m" else f"semantic_ids_kmeans_{dataset}.json"
    output_path = os.path.join(data_dir, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(semantic_ids_dict, f, indent=2)
    print(f"\nSemantic IDs successfully written to {output_path}!")


if __name__ == "__main__":
    main()
