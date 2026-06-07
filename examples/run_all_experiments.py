"""Script to automate all experiments across Beauty, Sports, Toys, and Steam datasets.

Runs:
1. RQ-KMeans and RQ-VAE Semantic ID generation.
2. HSTU model training and evaluation.
3. TIGER (VAE) model training and evaluation.
4. TIGER (K-Means) model training and evaluation.
"""

import os
import subprocess
import sys


def run_cmd(cmd: str):
    print(f"\n=======================================================")
    print(f"RUNNING: {cmd}")
    print(f"=======================================================\n")
    # Set PYTHONPATH=src to ensure correct imports
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(cmd, shell=True, check=True, env=env)


def main():
    datasets = ["beauty", "sports", "toys", "steam"]
    epochs = 5  # 5 epochs per run to ensure fast and reliable completion of all 12 experiments

    print("--- Starting Full Evaluation Suite ---")

    # Step 1: Pre-generate all Semantic IDs (fast)
    for dataset in datasets:
        print(f"\n--- Generating Semantic IDs for {dataset.upper()} ---")
        # Run KMeans (takes ~3s)
        run_cmd(f"python examples/train_rqkmeans.py --dataset {dataset}")
        # Run VAE (takes ~15s for 50 epochs)
        run_cmd(f"python examples/train_rqvae.py --dataset {dataset} --epochs 50")

    # Step 2: Train and evaluate all sequential models
    for dataset in datasets:
        print(f"\n--- Evaluating Models on {dataset.upper()} ---")

        # 1. HSTU Model
        run_cmd(
            f"python examples/train_hstu.py --model hstu --dataset {dataset} --epochs {epochs}"
        )

        # 2. TIGER Model (VAE Semantic IDs)
        run_cmd(
            f"python examples/train_tiger.py --dataset {dataset} --epochs {epochs} "
            f"--semantic_ids_path ./data/semantic_ids_{dataset}.json"
        )

        # 3. TIGER Model (K-Means Semantic IDs)
        run_cmd(
            f"python examples/train_tiger.py --dataset {dataset} --epochs {epochs} "
            f"--semantic_ids_path ./data/semantic_ids_kmeans_{dataset}.json "
            f"--checkpoint_dir ./data/tiger_kmeans_{dataset}_checkpoints "
            f"--tb_log_dir ./data/tensorboard/tiger_kmeans_{dataset}"
        )

    print("\n--- Full Evaluation Suite Completed Successfully! ---")
    print("Results are appended to experiment_results.md and logged to TensorBoard.")


if __name__ == "__main__":
    main()
