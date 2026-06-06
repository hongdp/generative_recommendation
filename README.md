# Generative Recommendation System

A JAX/Flax-based generative recommendation system utilizing LLM/Transformer architectures.

## Getting Started

1. Set up the Conda environment (see details in [SKILL.md](file:///home/hongdp/Workspace/generative_recommendation/SKILL.md)).
2. Install the package in editable mode:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests to verify setup:
   ```bash
   pytest
   ```

## Project Structure

- `src/`: Core library code containing `models`, `datasets`, and `evaluation` components.
- `examples/`: Executable scripts for training and evaluation.
- `tests/`: Unit and integration tests.
- `SKILL.md`: Documented development principles, environment guidelines, and learnings.
- `experiment_results.md`: Logs for training runs and evaluation results.
- `tasks.md`: Track backlog and execution progress.

## Running Experiments

All runner scripts reside in the `examples/` directory. Ensure `PYTHONPATH=src` is prefixed when running.

### 1. Index-Based Sequential Models (HSTU / Pluggable Transformer)
To train an index-based recommendation model:
```bash
# Train HSTU Model (default)
PYTHONPATH=src python examples/train_full_movielens.py --model hstu --epochs 40

# Train other pluggable architectures (e.g. Transformer)
PYTHONPATH=src python examples/train_full_movielens.py --model transformer --epochs 40
```

### 2. Generative Recommendation Model (TIGER)
TIGER requires discrete item Semantic IDs to be generated first.

#### Step A: Generate Semantic IDs
You can choose between the deep autoencoder method (RQ-VAE) or the clustering-based method (RQ-KMeans):
```bash
# Option A: Train Residual Quantization VAE (200 epochs)
PYTHONPATH=src python examples/train_rqvae.py --epochs 200

# Option B: Run Residual Quantization K-Means (fast, CPU-only)
PYTHONPATH=src python examples/train_rqkmeans.py
```

#### Step B: Train TIGER Sequence Model
Once Semantic IDs are written to `./data/`, train TIGER:
```bash
# Train TIGER with VAE Semantic IDs (default)
PYTHONPATH=src python examples/train_tiger.py --epochs 30 --semantic_ids_path ./data/semantic_ids.json

# Train TIGER with KMeans Semantic IDs
PYTHONPATH=src python examples/train_tiger.py --epochs 30 --semantic_ids_path ./data/semantic_ids_kmeans.json --checkpoint_dir ./data/tiger_kmeans_checkpoints --tb_log_dir ./data/tensorboard/tiger_kmeans
```

### 3. Monitoring with TensorBoard
Training runs automatically log to `./data/tensorboard/`. Monitor progress by running:
```bash
tensorboard --logdir ./data/tensorboard
```
