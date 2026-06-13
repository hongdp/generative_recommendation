#!/bin/bash
set -e

INSTANCE_NAME="tiger-tpu-1781377472"
ZONE="us-west4-a"
REMOTE_DIR="/home/$(whoami)/generative_recommendation"
REMOTE_CMD="bash scripts/tpu_experiments.sh"

# No cleanup trap. Keep the TPU running indefinitely.

echo "=========================================="
echo "SYNCHRONIZATION (UP): Copying files to TPU"
echo "=========================================="
# Create target directory
gcloud compute tpus tpu-vm ssh $INSTANCE_NAME --zone=$ZONE --command="mkdir -p $REMOTE_DIR/data"

# Copy source code and files
gcloud compute tpus tpu-vm scp --recurse --zone=$ZONE \
    src examples scripts pyproject.toml README.md experiment_results.md \
    $INSTANCE_NAME:$REMOTE_DIR/

# Skip copying heavy datasets
# gcloud compute tpus tpu-vm scp --recurse --zone=$ZONE \
#     data/semantic_ids*.json \
#     data/steam \
#     $INSTANCE_NAME:$REMOTE_DIR/data/ 2>/dev/null || true

echo "=========================================="
echo "EXECUTION & REAL-TIME MONITORING"
echo "=========================================="
mkdir -p ./data/tensorboard
(
  while true; do
    gcloud compute tpus tpu-vm scp --recurse --zone=$ZONE $INSTANCE_NAME:$REMOTE_DIR/data/tensorboard/* ./data/tensorboard/ 2>/dev/null || true
    sleep 30
  done
) &
SYNC_PID=$!

set +e
gcloud compute tpus tpu-vm ssh $INSTANCE_NAME --zone=$ZONE --command="cd $REMOTE_DIR && pip install -U pip && pip install \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip install -e . && $REMOTE_CMD"
TRAIN_EXIT_CODE=$?
set -e

kill $SYNC_PID || true

echo "=========================================="
echo "SYNCHRONIZATION (DOWN): Copying logs back"
echo "=========================================="
mkdir -p ./tpu_sync
gcloud compute tpus tpu-vm scp --recurse --zone=$ZONE \
    $INSTANCE_NAME:$REMOTE_DIR/data/*_checkpoints \
    $INSTANCE_NAME:$REMOTE_DIR/data/tensorboard \
    ./tpu_sync/ || echo "Warning: Sync down failed."

exit $TRAIN_EXIT_CODE
