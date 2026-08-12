#!/bin/bash
# One-shot cloud run: provision -> bootstrap -> launch -> (VM self-persists to
# GCS and self-deletes on completion). Local side exits once the run is launched.
#   bash scripts/gcp/launch_run.sh <run_name> <training command...>
set -e
RUN_NAME=$1; shift

# GCE names must be lowercase RFC1035 (an uppercase run name silently fails
# every zone; found the hard way behind a | tail).
SAFE_NAME=$(echo "${RUN_NAME//_/-}" | tr '[:upper:]' '[:lower:]')
export VM_NAME="ka-${SAFE_NAME}-$(date +%s)"
bash "$(dirname "$0")/start_gpu_vm.sh"
# Zone via lookup, not the shared .gcp_vm_current file (parallel-launch race).
ZONE=$(gcloud compute instances list --project=workstation-185016 --filter="name=$VM_NAME" --format="value(zone)" | awk -F/ '{print $NF}')
VM="$VM_NAME.$ZONE.workstation-185016"

echo "=== waiting for ssh on $VM ==="
for i in $(seq 1 90); do ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$VM" true 2>/dev/null && break; sleep 20; done
ssh "$VM" true

rsync -az --exclude .git --exclude data --exclude experiments --exclude __pycache__ --exclude tpu_sync \
    /home/hongdp/Workspace/generative_recommendation/ "$VM":generative_recommendation/
scp -q "$(dirname "$0")/bootstrap_vm.sh" "$VM":
ssh "$VM" 'sudo apt-get install -y -q python3-venv python3.10-venv >/dev/null 2>&1; bash bootstrap_vm.sh' 2>&1 | tail -2
ssh "$VM" 'cd ~/generative_recommendation && mkdir -p experiments/kuairand_locality_stage1_20260809/item2vec && gcloud storage cp -n gs://llm-pretraining-workstation-185016/datasets/generative_recommendation/kuairand/item2vec_table.msgpack experiments/kuairand_locality_stage1_20260809/item2vec/'
ssh "$VM" "nohup bash generative_recommendation/scripts/gcp/run_and_upload.sh $RUN_NAME $* >/dev/null 2>&1 & disown; sleep 2; pgrep -f run_and_upload >/dev/null && echo LAUNCHED_OK || echo LAUNCH_FAILED"
