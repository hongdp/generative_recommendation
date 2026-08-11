#!/bin/bash
# Create (or fall back across zones for) an A100 flex-start VM for KuaiRand training.
# Follows the gcp-trainer skill: scopes at create time, max-run-duration, zone
# fallback, timestamped ephemeral names, config-ssh at the end.
#
#   PROVISIONING=ondemand ZONE=us-central1-b bash scripts/gcp/start_gpu_vm.sh
set -e

PROJECT_ID=${PROJECT_ID:-"workstation-185016"}
VM_NAME=${VM_NAME:-"kuairand-a100-$(date +%s)"}
MACHINE_TYPE=${MACHINE_TYPE:-"a2-highgpu-1g"}       # 1x A100 40GB
IMAGE_FAMILY=${IMAGE_FAMILY:-"common-cu124-ubuntu-2204-nvidia-550"}
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_GB=${BOOT_DISK_GB:-200}
MAX_RUN=${MAX_RUN:-"48h"}
PROVISIONING=${PROVISIONING:-flex}                   # flex | ondemand
ZONES=${ZONES:-"us-central1-b us-central1-c us-central1-f us-east1-b europe-west4-a"}

EXTRA=()
if [[ "$PROVISIONING" == "flex" ]]; then
    # DWS flex-start: ~-45% price, no preemption up to 7d, draws preemptible quota.
    EXTRA+=(--provisioning-model=FLEX_START --instance-termination-action=DELETE
            --max-run-duration="$MAX_RUN" --request-valid-for-duration=2h)
else
    EXTRA+=(--max-run-duration="$MAX_RUN" --instance-termination-action=STOP)
fi

for ZONE in $ZONES; do
    echo "=== Trying $MACHINE_TYPE in $ZONE ($PROVISIONING) ==="
    if gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT_ID" --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="${BOOT_DISK_GB}GB" --boot-disk-type=pd-balanced \
        --maintenance-policy=TERMINATE \
        --scopes=storage-rw,logging-write,monitoring-write \
        --metadata=install-nvidia-driver=True \
        "${EXTRA[@]}"; then
        echo "$VM_NAME $ZONE" > .gcp_vm_current
        gcloud compute config-ssh --project="$PROJECT_ID" >/dev/null
        echo "=== VM up: $VM_NAME.$ZONE.$PROJECT_ID (ssh alias); recorded in .gcp_vm_current ==="
        exit 0
    fi
    echo "--- $ZONE failed (stockout/quota), trying next ---"
done
echo "!!! No zone could provision $MACHINE_TYPE" >&2
exit 1
