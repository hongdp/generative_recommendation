# Generative & Index-Based Sequential Recommendation System (JAX/Flax)

A high-performance sequential recommendation library built using JAX and Flax, featuring state-of-the-art architectures:
1. **HSTU (Hierarchical Sequential Transduction Unit)**: The index-based sequence model reported in the ICML 2024 paper.
2. **TIGER (Generative Retrieval)**: A generative recommendation framework using Semantic ID quantization (supporting both RQ-VAE and RQ-KMeans).
3. **SASRec-style Transformer**: Standard causal attention model architecture.

---

## 🚀 Getting Started

### 1. Setup Environment
Ensure you have a JAX-compatible environment set up. Detailed instructions are available in [SKILL.md](file:///home/hongdp/Workspace/generative_recommendation/SKILL.md).

### 2. Install Project
Install the codebase in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
```

### 3. Run Unit & Integration Tests
Verify the entire library and its modules (data loaders, model shapes, step logic, ranking metrics):
```bash
PYTHONPATH=src pytest
```

---

## 📂 Project Structure

* `src/`: Core library modules.
  * `datasets/`: Preprocessing, sequence splitting, and data loading pipelines for **MovieLens-1M**, **Amazon (Beauty, Sports, Toys)**, and **Steam** datasets.
  * `models/`: Implementations of `HSTUModel`, `TransformerModel`, `TIGERModel`, and `RQVAE`.
  * `evaluation/`: Evaluator components computing standard ranking metrics: Hit Rate (HR@K), Normalized Discounted Cumulative Gain (NDCG@K), and Mean Reciprocal Rank (MRR).
* `examples/`: Train & evaluation runner scripts.
  * **Index-Based Models**:
    * `train_hstu.py`: Generic runner for index-based sequential architectures (`--model hstu` or `--model transformer`).
  * **Semantic ID Generation**:
    * `train_rqvae.py` / `train_rqkmeans.py`: Semantic ID generation scripts using deep autoencoders or fast clustering.
  * **Generative Discrete Models (TIGER family)**:
    * `train_tiger.py`: Standard TIGER model.
    * `train_tiger_seq2seq.py`: TIGER using standard encoder-decoder seq2seq architecture.
    * `train_tiger_cot.py`: TIGER with Chain-of-Thought reasoning.
    * `train_tiger_rl_cot.py`: TIGER CoT fine-tuned via Reinforcement Learning.
    * `train_tiger_encoder_ce.py`: **Ablation Model** testing TIGER encoder with direct CE projection instead of 3-level decoding.
  * **Generative Continuous Models (Flow Matching)**:
    * `train_hstu_flow.py`: Base Flow Matching framework on top of HSTU representations.
    * `train_hstu_flow_ce.py`: Flow Matching with Contrastive Cross-Entropy auxiliary loss.
    * `train_hstu_flow_t5_vae.py`: Flow Matching targeting T5 latent space via VAE.
    * `train_tiger_flow.py`: Flow Matching targeting TIGER embedding spaces.
  * **Batch Scripts**:
    * `run_all_experiments.py`: Script to automate ID generation, training, and evaluation across Amazon and Steam datasets.
* `SKILL.md`: Documented JAX/Flax development guidelines, memory-efficient training rules, and experience log.
* `experiment_results.md`: Complete comparative baseline records of all experiments.
* `walkthrough.md`: Detailed analysis and destructive testing conclusions (e.g. Random IDs vs RQVAE IDs).

---

## 🧪 Running Experiments

All scripts support pluggable datasets via the `--dataset` CLI parameter (`ml-1m`, `beauty`, `sports`, `toys`, `steam`).

### 1. Sequential Index-Based Model
```bash
# Train HSTU on Steam
PYTHONPATH=src python examples/train_hstu.py --model hstu --dataset steam --epochs 30

# Train standard Transformer on MovieLens-1M
PYTHONPATH=src python examples/train_hstu.py --model transformer --dataset ml-1m --epochs 40
```

### 2. Generative Retrieval Model (TIGER)
Generative retrieval first tokenizes items into discrete Semantic IDs, then trains the TIGER sequence-to-sequence model using teacher-forcing.

#### Step A: Generate Semantic IDs (RQ-VAE or RQ-KMeans)
Choose between the neural Residual Quantization VAE or the fast sequential K-Means clustering method:
```bash
# Option A: Train RQ-VAE (e.g. on Sports dataset)
PYTHONPATH=src python examples/train_rqvae.py --dataset sports --epochs 200

# Option B: Run CPU-friendly RQ-KMeans (runs in seconds)
PYTHONPATH=src python examples/train_rqkmeans.py --dataset sports
```

#### Step B: Train TIGER
```bash
# Train TIGER with VAE Semantic IDs
PYTHONPATH=src python examples/train_tiger.py --dataset sports --epochs 30 --semantic_ids_path ./data/semantic_ids_sports.json

# Train TIGER with KMeans Semantic IDs
PYTHONPATH=src python examples/train_tiger.py --dataset sports --epochs 30 --semantic_ids_path ./data/semantic_ids_kmeans_sports.json
```

### 3. Automated Benchmark Suite
To run all experiments (semantic ID generation, HSTU training, TIGER VAE, and TIGER KMeans) across all Amazon/Steam datasets sequentially:
```bash
PYTHONPATH=src python examples/run_all_experiments.py
```

### 4. Monitoring Progress
We support batch-step metrics tracking. Launch TensorBoard to inspect train loss, validation metric curves, and benchmarks:
```bash
tensorboard --logdir ./data/tensorboard
```
