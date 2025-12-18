#!/usr/bin/env bash
set -euo pipefail

# Deletes the VM created by gcp_gpu_vm_create.sh.

: "${PROJECT_ID:?Set PROJECT_ID (e.g. export PROJECT_ID=my-project)}"

NAME="${NAME:-gpu-vm-1}"
ZONE="${ZONE:-us-central1-a}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found. Install Google Cloud CLI first." >&2
  exit 1
fi

gcloud config set project "$PROJECT_ID" >/dev/null

echo "Deleting VM '$NAME' in $ZONE"
gcloud compute instances delete "$NAME" --project "$PROJECT_ID" --zone "$ZONE" --quiet

echo "Done."