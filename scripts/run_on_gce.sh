#!/bin/bash
set -e

# Default configurations
INSTANCE_NAME=${INSTANCE_NAME:-"tiger-train-$(date +%s)"}
ZONE=${ZONE:-"us-central1-a"}
MACHINE_TYPE=${MACHINE_TYPE:-"g2-standard-12"}
ACCELERATOR=${ACCELERATOR:-"type=nvidia-l4,count=1"}
IMAGE_FAMILY=${IMAGE_FAMILY:-"common-cu121-debian-11"}
IMAGE_PROJECT=${IMAGE_PROJECT:-"deeplearning-platform-release"}

# Local directories to sync
LOCAL_DIR=$(pwd)
REMOTE_DIR="/home/jupyter/generative_recommendation"

# The command to run on the VM
REMOTE_CMD=${REMOTE_CMD:-"PYTHONPATH=src python examples/train_tiger_seq2seq.py --dataset steam --semantic_ids_path ./data/semantic_ids_random_steam.json --epochs 1"}

# Cleanup function guaranteed to run
function cleanup {
    echo "=========================================="
    echo "CLEANUP: Deleting instance $INSTANCE_NAME..."
    echo "=========================================="
    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet || true
    echo "Instance $INSTANCE_NAME deleted."
}

# Register the trap
trap cleanup EXIT

echo "=========================================="
echo "PROVISIONING: Creating VM $INSTANCE_NAME"
echo "=========================================="

# Create the instance
if [ "$ACCELERATOR" == "none" ]; then
    gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --boot-disk-size=100GB \
        --metadata="install-nvidia-driver=False" \
        --maintenance-policy=TERMINATE
else
    gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator=$ACCELERATOR \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --boot-disk-size=100GB \
        --metadata="install-nvidia-driver=True" \
        --maintenance-policy=TERMINATE
fi

echo "Waiting for SSH to be ready..."
sleep 20

echo "=========================================="
echo "SYNCHRONIZATION (UP): Copying files to VM"
echo "=========================================="
# Create target directory
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="mkdir -p $REMOTE_DIR"

# Copy source code and files
gcloud compute scp --recurse --zone=$ZONE \
    src examples scripts pyproject.toml README.md experiment_results.md \
    $INSTANCE_NAME:$REMOTE_DIR/

# Create data dir and copy only semantic ids and cache
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="mkdir -p $REMOTE_DIR/data"
gcloud compute scp --recurse --zone=$ZONE \
    data/semantic_ids*.json \
    data/steam \
    $INSTANCE_NAME:$REMOTE_DIR/data/ 2>/dev/null || true

echo "=========================================="
echo "EXECUTION: Running training script on VM"
echo "=========================================="
# Run the command and stream output
set +e # Don't exit immediately if training fails, so we can still download logs
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="cd $REMOTE_DIR && pip install -e . || true && $REMOTE_CMD"
TRAIN_EXIT_CODE=$?
set -e

echo "=========================================="
echo "SYNCHRONIZATION (DOWN): Copying logs back"
echo "=========================================="
# Download checkpoints and tensorboard logs
mkdir -p ./gce_sync
gcloud compute scp --recurse --zone=$ZONE \
    $INSTANCE_NAME:$REMOTE_DIR/data/*_checkpoints \
    $INSTANCE_NAME:$REMOTE_DIR/data/tensorboard \
    ./gce_sync/ || echo "Warning: Sync down failed or no files to copy."

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "Training script failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
else
    echo "Training completed successfully."
fi
