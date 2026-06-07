"""Script to train HSTU, TIGER (VAE), and TIGER (K-Means) to full convergence on the Steam dataset.

Each model will run for up to 30 epochs with a patience of 5 epochs.
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
    epochs = 30
    patience = 5
    dataset = "steam"

    print("--- Starting Steam Full Convergence Suite ---")

    # 1. HSTU Model
    run_cmd(
        f"python examples/train_hstu.py --model hstu --dataset {dataset} --epochs {epochs} --patience {patience} "
        f"--checkpoint_dir ./data/hstu_steam_convergence_checkpoints --tb_log_dir ./data/tensorboard/hstu_steam_convergence"
    )

    # 2. TIGER Model (VAE Semantic IDs)
    run_cmd(
        f"python examples/train_tiger.py --dataset {dataset} --epochs {epochs} --patience {patience} "
        f"--semantic_ids_path ./data/semantic_ids_{dataset}.json "
        f"--checkpoint_dir ./data/tiger_vae_steam_convergence_checkpoints --tb_log_dir ./data/tensorboard/tiger_vae_steam_convergence"
    )

    # 3. TIGER Model (K-Means Semantic IDs)
    run_cmd(
        f"python examples/train_tiger.py --dataset {dataset} --epochs {epochs} --patience {patience} "
        f"--semantic_ids_path ./data/semantic_ids_kmeans_{dataset}.json "
        f"--checkpoint_dir ./data/tiger_kmeans_steam_convergence_checkpoints --tb_log_dir ./data/tensorboard/tiger_kmeans_steam_convergence"
    )

    print("\n--- Steam Full Convergence Suite Completed Successfully! ---")
    print("All results are documented in experiment_results.md and logged to TensorBoard.")


if __name__ == "__main__":
    main()
