#!/usr/bin/env bash
set -euo pipefail

# Creates a GCP GPU VM using gcloud.
# Requirements: gcloud installed + authenticated.

: "${PROJECT_ID:?Set PROJECT_ID (e.g. export PROJECT_ID=my-project)}"

NAME="${NAME:-gpu-vm-1}"
ZONE="${ZONE:-me-central1-a}"

# Default for G2: machine type includes NVIDIA L4 (1 GPU on g2-standard-8).
# If you switch to a non-G2 machine type, you can still set GPU_TYPE/GPU_COUNT.
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
GPU_COUNT="${GPU_COUNT:-1}"

# Middle East default: G2 (L4) with 1 GPU.
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"

# Deep Learning VM image with CUDA preinstalled (recommended).
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu121}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200GB}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-balanced}"

# Network access controls
NETWORK_TAGS="${NETWORK_TAGS:-gpu-vm,allow-ssh,allow-app}"
FIREWALL_RULE_NAME="${FIREWALL_RULE_NAME:-allow-ssh-and-app}"
OPEN_PORTS="${OPEN_PORTS:-22,8080}"
SOURCE_RANGES="${SOURCE_RANGES:-0.0.0.0/0}"

STARTUP_SCRIPT_PATH="${STARTUP_SCRIPT_PATH:-$(cd "$(dirname "$0")" && pwd)/gcp_gpu_vm_startup.sh}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found. Install Google Cloud CLI first." >&2
  exit 1
fi

if [[ ! -f "$STARTUP_SCRIPT_PATH" ]]; then
  echo "Startup script not found at: $STARTUP_SCRIPT_PATH" >&2
  exit 1
fi

# Ensure correct project

gcloud config set project "$PROJECT_ID" >/dev/null

# Enable Compute API (safe to re-run)
gcloud services enable compute.googleapis.com >/dev/null

# Create/ensure firewall rule (best-effort; no-op if it already exists)
if ! gcloud compute firewall-rules describe "$FIREWALL_RULE_NAME" --project "$PROJECT_ID" >/dev/null 2>&1; then
  # gcloud expects rules like: tcp:22,tcp:8080
  RULES=""
  IFS=',' read -r -a _PORTS <<< "$OPEN_PORTS"
  for p in "${_PORTS[@]}"; do
    p="${p//[[:space:]]/}"
    [[ -z "$p" ]] && continue
    if [[ -n "$RULES" ]]; then RULES+=",";
    fi
    RULES+="tcp:${p}"
  done

  echo "Creating firewall rule: $FIREWALL_RULE_NAME (ports: $OPEN_PORTS, source: $SOURCE_RANGES)"
  gcloud compute firewall-rules create "$FIREWALL_RULE_NAME" \
    --project "$PROJECT_ID" \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules="$RULES" \
    --source-ranges="$SOURCE_RANGES" \
    --target-tags="$NETWORK_TAGS" \
    >/dev/null
else
  echo "Firewall rule already exists: $FIREWALL_RULE_NAME"
fi

if [[ "$MACHINE_TYPE" == g2-* ]]; then
  echo "Creating G2 VM '$NAME' in $ZONE (machine-type: $MACHINE_TYPE, includes NVIDIA L4 GPU)"
else
  echo "Creating VM '$NAME' in $ZONE with $GPU_COUNT x $GPU_TYPE (machine-type: $MACHINE_TYPE)"
fi

CREATE_ARGS=(
  compute instances create "$NAME"
  --project "$PROJECT_ID"
  --zone "$ZONE"
  --machine-type "$MACHINE_TYPE"
  --maintenance-policy TERMINATE
  --restart-on-failure
  --provisioning-model STANDARD
  --image-family "$IMAGE_FAMILY"
  --image-project "$IMAGE_PROJECT"
  --boot-disk-size "$BOOT_DISK_SIZE"
  --boot-disk-type "$BOOT_DISK_TYPE"
  --tags "$NETWORK_TAGS"
  --scopes "https://www.googleapis.com/auth/cloud-platform"
  --metadata-from-file startup-script="$STARTUP_SCRIPT_PATH"
  --no-shielded-secure-boot
  --shielded-vtpm
  --shielded-integrity-monitoring
)

# For non-G2 machines, attach a GPU explicitly.
if [[ "$MACHINE_TYPE" != g2-* ]]; then
  CREATE_ARGS+=( --accelerator "type=${GPU_TYPE},count=${GPU_COUNT}" )
fi

gcloud "${CREATE_ARGS[@]}"

echo

echo "VM created. Connect with:"
echo "  gcloud compute ssh $NAME --project $PROJECT_ID --zone $ZONE"
echo
echo "Check GPU:"
echo "  nvidia-smi"
echo
echo "Docker GPU sanity check:"
echo "  docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi"
