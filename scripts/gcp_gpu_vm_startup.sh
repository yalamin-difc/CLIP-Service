#!/usr/bin/env bash
set -euo pipefail

# Startup script for a GCP GPU VM.
# - Installs Docker
# - Configures NVIDIA Container Toolkit (for GPU containers)
# - Tries to ensure NVIDIA drivers are present (DLVM images already include them)

export DEBIAN_FRONTEND=noninteractive

log() { echo "[startup] $*"; }

log "Updating apt metadata"
apt-get update -y

log "Installing base packages"
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  jq \
  git \
  unzip

if ! command -v docker >/dev/null 2>&1; then
  log "Installing Docker"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg

  UBUNTU_CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list

  apt-get update -y
  apt-get install -y --no-install-recommends docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

log "Enabling Docker"
systemctl enable --now docker

# Add the default user to docker group (on GCE this is usually 'ubuntu' or the OS Login user).
# We best-effort add common usernames if they exist.
for u in ubuntu debian google; do
  if id "$u" >/dev/null 2>&1; then
    log "Adding user '$u' to docker group"
    usermod -aG docker "$u" || true
  fi
done

# NVIDIA Container Toolkit
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi not found (driver may still be installing or image lacks driver)."
  log "If you used a Deep Learning VM image, drivers should appear shortly."
else
  log "NVIDIA driver appears installed" 
  nvidia-smi || true
fi

if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  log "Installing NVIDIA Container Toolkit"

  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list

  apt-get update -y
  apt-get install -y --no-install-recommends nvidia-container-toolkit

  # Configure Docker runtime
  if command -v nvidia-ctk >/dev/null 2>&1; then
    nvidia-ctk runtime configure --runtime=docker || true
  fi
  systemctl restart docker || true
fi

log "Startup complete"