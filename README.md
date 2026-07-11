# Generative & Index-Based Sequential Recommendation System (JAX/Flax)

A high-performance sequential recommendation library built using JAX and Flax, featuring state-of-the-art architectures:
1. **HSTU (Hierarchical Sequential Transduction Unit)**: The index-based sequence model reported in the ICML 2024 paper.
2. **TIGER (Generative Retrieval)**: A generative recommendation framework using Semantic ID quantization (supporting both RQ-VAE and RQ-KMeans).
3. **SASRec-style Transformer**: Standard causal attention model architecture.
4. **Readout-token HSTU**: A two-tower retrieval variant that moves the readout off the last item's residual stream onto dedicated anchor-masked `<begin>` tokens, with optional request-time conditioning (see [dedicated_readout_token_design.md](dedicated_readout_token_design.md)).

---

## 📊 Current Best Results (test, leave-one-out)

| Dataset | Best model | HR@10 | NDCG@10 | Notes |
|---|---|---|---|---|
| Amazon Beauty | Readout-HSTU `<begin>` + zero-init request-time | **0.05605** | **0.03198** | New repo best on all 7 metrics (2026-07-09, 3 seeds) |
| Amazon Beauty | TIGER (rich XXL IDs + std-RQ-VAE + dedup) | 0.0526 | — | 87.6% of the LIGER-paper TIGER (0.0601) |
| Steam | Readout-HSTU `<begin>` + request-time | **0.26161** | **0.19240** | +26% over the prior repo best (TIGER 0.2077 / HSTU 0.2074) |

Full run-by-run records live in [experiment_results.md](experiment_results.md); analysis lives in [dedicated_readout_token_design.md](dedicated_readout_token_design.md) §13.

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
  * `models/`: Implementations of `HSTUModel`, `TransformerModel`, `TIGERModel`, `ReadoutHSTUModel`, and `RQVAE`.
  * `evaluation/`: Evaluator components computing standard ranking metrics: Hit Rate (HR_K), Normalized Discounted Cumulative Gain (NDCG_K), and Mean Reciprocal Rank (MRR).
* `examples/`: Train & evaluation runner scripts.
  * **Index-Based Models**:
    * `train_hstu.py`: Generic runner for index-based sequential architectures (`--model hstu` or `--model transformer`).
    * `train_readout_hstu.py`: Dedicated readout-token A/B (`--readout item|begin`), with gated transition-prior re-injection (`--reinject`) and request-time conditioning of `<begin>` (`--time_features`, input or late fusion).
  * **Semantic ID Generation**:
    * `train_rqvae.py` / `train_rqkmeans.py`: Semantic ID generation using RQ-VAE (linear or MLP encoder via `--hidden_dims`) or fast clustering.
    * `build_semantic_ids_rich.py`: Semantic IDs from rich Amazon item text (title + brand + category + price) with Sentence-T5-XXL — the winning recipe in the Beauty TIGER-gap campaign.
    * `build_semantic_ids_mgcl.py`: Multi-granularity contrastive Semantic IDs (UniSID-style MGCL without the MLLM backbone).
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
* `dedicated_readout_token_design.md`: Design doc + running lab notebook for the readout-token line: dual-role interference theory, leakage diagnostics, transition-prior re-injection, direct-term logit decomposition, and request-time conditioning (§13 holds all empirical findings).

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

#### Readout-token A/B (dedicated `<begin>` readout)
```bash
# Baseline arm: readout from the last item's own residual stream
PYTHONPATH=src python examples/train_readout_hstu.py --dataset beauty --readout item

# Clean arm: dedicated anchor-masked <begin> readout
PYTHONPATH=src python examples/train_readout_hstu.py --dataset beauty --readout begin

# Best known configuration: <begin> + zero-init request-time conditioning
PYTHONPATH=src python examples/train_readout_hstu.py --dataset steam --readout begin --time_features --feat_zero_init
```
Each run appends a results row to `experiment_results.md` and logs leakage diagnostics (readout leak, table alignment). See the design doc for the full ablation map (§11) and findings (§13).

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

### 4. ☁️ Cloud TPU Distributed Training
For high-speed training on large datasets using Google Cloud TPU v5p or v5e (e.g. `v5p-8` or `v5litepod-8`), the repository supports automated 8-core data parallelism via `jax.pmap`. 

Use the provided bash scripts to provision a TPU VM, sync data, run all 4 ablation experiments (`batch_size=2048`), and stream TensorBoard logs to your local machine in real-time:
```bash
# Provision TPU and launch experiments (Warning: TPU will persist until manually stopped)
./scripts/run_on_tpu.sh
```
*Note on TPU Lifecycle: NEVER `delete` the TPU unless you are abandoning the instance forever. Always `stop` the TPU via `gcloud compute tpus tpu-vm stop <name>` to pause billing while preserving your 1.2GB Steam dataset and JAX environment on the persistent disk!*

### 5. Monitoring Progress
We support batch-step metrics tracking. Launch TensorBoard to inspect train loss, validation metric curves, and benchmarks:
```bash
tensorboard --logdir ./data/tensorboard
```
