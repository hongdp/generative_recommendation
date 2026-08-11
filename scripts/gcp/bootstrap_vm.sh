#!/bin/bash
# Runs ON the VM (first login): wait for driver, build venv, pull code + data.
# Idempotent — safe to rerun.
set -e

BUCKET="gs://llm-pretraining-workstation-185016/datasets/generative_recommendation/kuairand"
REPO_DIR="$HOME/generative_recommendation"

echo "=== Waiting for NVIDIA driver (async install, up to ~20 min) ==="
for i in $(seq 1 60); do
    if nvidia-smi >/dev/null 2>&1; then break; fi
    sleep 20
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== venv + JAX cuda ==="
if [ ! -d "$HOME/venvs/kuairand" ]; then
    python3 -m venv "$HOME/venvs/kuairand"
fi
source "$HOME/venvs/kuairand/bin/activate"
pip -q install --upgrade pip
pip -q install "jax[cuda12]" flax optax numpy pandas tensorboard
pip -q install torch --index-url https://download.pytorch.org/whl/cpu   # SummaryWriter only

echo "=== data from GCS (hash-gated) ==="
mkdir -p "$REPO_DIR/data/kuairand"
cd "$REPO_DIR/data/kuairand"
gcloud storage cp -n "$BUCKET/kuairand27k_top100000.npz" "$BUCKET/kuairand27k_top100000.meta.json" "$BUCKET/kuairand_cache.sha256" .
sha256sum -c kuairand_cache.sha256
mkdir -p "$REPO_DIR/experiments/kuairand_readout_ab_20260801/item"
gcloud storage cp -n "$BUCKET/ab_item_best_checkpoint.msgpack" "$REPO_DIR/experiments/kuairand_readout_ab_20260801/item/best_checkpoint.msgpack"
gcloud storage cp -n "$BUCKET/ab_item_config.json" "$REPO_DIR/experiments/kuairand_readout_ab_20260801/item/config.json"

python -c "import jax; print('devices:', jax.devices())"
echo "=== bootstrap complete ==="
