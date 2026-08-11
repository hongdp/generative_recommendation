#!/bin/bash
# Runs ON the VM: execute a training command with salvage-and-shutdown semantics.
#   bash scripts/gcp/run_and_upload.sh <run_name> <command...>
# EXIT trap uploads the experiment dir + log to GCS on EVERY exit path (crash
# included), writes a TRAIN_EXIT marker, then powers the VM off (flex-start VMs
# are ephemeral: DELETE on termination — state lives in GCS).
set -u

RUN_NAME=$1; shift
BUCKET="gs://llm-pretraining-workstation-185016/datasets/generative_recommendation/kuairand/runs/$RUN_NAME"
REPO_DIR="$HOME/generative_recommendation"
LOG="$HOME/${RUN_NAME}.log"
NOSHUTDOWN=${NOSHUTDOWN:-0}

salvage() {
    code=$?
    echo "TRAIN_EXIT=$code" | tee "$HOME/TRAIN_EXIT"
    nvidia-smi > "$HOME/gpu_info.txt" 2>&1 || true
    pip freeze > "$HOME/pip_freeze.txt" 2>&1 || true
    gcloud storage cp "$LOG" "$HOME/TRAIN_EXIT" "$HOME/gpu_info.txt" "$HOME/pip_freeze.txt" "$BUCKET/" || true
    gcloud storage rsync -r "$REPO_DIR/experiments" "$BUCKET/experiments" || true
    if [[ "$NOSHUTDOWN" != "1" ]]; then sudo shutdown -h now; fi
}
trap salvage EXIT

source "$HOME/venvs/kuairand/bin/activate"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1 PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "=== $RUN_NAME: $* ===" | tee "$LOG"
"$@" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
