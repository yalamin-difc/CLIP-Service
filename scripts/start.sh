#!/usr/bin/env bash
set -euo pipefail
 
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
APP="${APP_MODULE:-app:app}"
 
exec uvicorn "$APP" --host "$HOST" --port "$PORT"
