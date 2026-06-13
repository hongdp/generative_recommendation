#!/bin/bash
set -e

# Default configurations
INSTANCE_NAME=${INSTANCE_NAME:-"tiger-tpu-$(date +%s)"}
# Automatic zone and accelerator fallback for TPU provisioning
CONFIGS=(
    "v5p-8 us-central1-a"
    "v5p-8 us-central1-b"
    "v5p-8 us-central1-c"
    "v5p-8 us-east1-c"
    "v5p-8 us-east1-d"
    "v5p-8 us-central1-d"
    "v5litepod-8 us-east1-c"
    "v5litepod-8 us-central1-a"
    "v5litepod-8 us-west4-a"
    "v5litepod-4 us-east1-c"
    "v5litepod-4 us-central1-a"
    "v5litepod-4 us-west4-a"
    "v6e-8 us-central2-b"
    "v6e-4 us-central2-b"
)
TPU_VERSION=${TPU_VERSION:-"tpu-ubuntu2204-base"}

LOCAL_DIR=$(pwd)
REMOTE_DIR="/home/$(whoami)/generative_recommendation"
REMOTE_CMD=${REMOTE_CMD:-"bash scripts/tpu_experiments.sh"}

# Cleanup function guaranteed to run
function cleanup {
    echo "=========================================="
    echo "CLEANUP: Deleting TPU VM $INSTANCE_NAME..."
    echo "=========================================="
    gcloud compute tpus tpu-vm delete $INSTANCE_NAME --zone=$SUCCESS_ZONE --quiet || true
    echo "TPU VM $INSTANCE_NAME deleted."
}

# Register the trap
trap cleanup EXIT

echo "=========================================="
echo "PROVISIONING: Creating TPU VM $INSTANCE_NAME"
echo "=========================================="

SUCCESS_ZONE=""
SUCCESS_ACCEL=""
for conf in "${CONFIGS[@]}"; do
    ACCEL=$(echo $conf | awk '{print $1}')
    Z=$(echo $conf | awk '{print $2}')
    echo "Attempting to create TPU $ACCEL in zone $Z..."
    if gcloud compute tpus tpu-vm create $INSTANCE_NAME \
        --zone=$Z \
        --accelerator-type=$ACCEL \
        --version=$TPU_VERSION; then
        echo "Successfully provisioned $ACCEL in $Z!"
        SUCCESS_ZONE=$Z
        SUCCESS_ACCEL=$ACCEL
        break
    else
        echo "Failed $ACCEL in $Z. Trying next..."
    fi
done

if [ -z "$SUCCESS_ZONE" ]; then
    echo "FATAL: Exhausted all fallback configs. No TPU capacity available."
    exit 1
fi
ZONE=$SUCCESS_ZONE

echo "Waiting for SSH to be ready..."
sleep 30

echo "=========================================="
echo "SYNCHRONIZATION (UP): Copying files to TPU"
echo "=========================================="
# Create target directory
gcloud compute tpus tpu-vm ssh $INSTANCE_NAME --zone=$ZONE --command="mkdir -p $REMOTE_DIR/data"

# Copy source code and files
gcloud compute tpus tpu-vm scp --recurse --zone=$ZONE \
    src examples scripts pyproject.toml README.md experiment_results.md \
    $INSTANCE_NAME:$REMOTE_DIR/

# Copy necessary data (Semantic IDs and Steam Cache)
gcloud compute tpus tpu-vm scp --recurse --zone=$ZONE \
    data/semantic_ids*.json \
    data/steam \
    $INSTANCE_NAME:$REMOTE_DIR/data/ 2>/dev/null || true

echo "=========================================="
echo "EXECUTION & REAL-TIME MONITORING"
echo "=========================================="
# Start a background process to continuously sync TensorBoard logs
mkdir -p ./data/tensorboard
(
  while true; do
    gcloud compute tpus tpu-vm scp --recurse --zone=$ZONE $INSTANCE_NAME:$REMOTE_DIR/data/tensorboard/* ./data/tensorboard/ 2>/dev/null || true
    sleep 30
  done
) &
SYNC_PID=$!

# Run the command and stream output
set +e
# For TPU, we need to install the JAX TPU wheels
gcloud compute tpus tpu-vm ssh $INSTANCE_NAME --zone=$ZONE --command="cd $REMOTE_DIR && pip install -U pip && pip install \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip install -e . && $REMOTE_CMD"
TRAIN_EXIT_CODE=$?
set -e

# Stop the background sync loop
kill $SYNC_PID || true

echo "=========================================="
echo "SYNCHRONIZATION (DOWN): Copying logs back"
echo "=========================================="
mkdir -p ./tpu_sync
gcloud compute tpus tpu-vm scp --recurse --zone=$ZONE \
    $INSTANCE_NAME:$REMOTE_DIR/data/*_checkpoints \
    $INSTANCE_NAME:$REMOTE_DIR/data/tensorboard \
    ./tpu_sync/ || echo "Warning: Sync down failed."

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "Training script failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
else
    echo "Training completed successfully."
fi
