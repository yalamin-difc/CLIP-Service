"""Environment-controlled configuration for the multi-engine (P13) surface.

Nothing here is read by the legacy CLIP endpoints in app.py -- those keep
using their own existing constants untouched. This module only backs the
new engine registry and the /v2 API / A/B / Festival "AI Model Comparison"
surfaces, all of which default to off.
"""
from __future__ import annotations

import os


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value.strip())


DEFAULT_EMBEDDING_ENGINE = os.environ.get("DEFAULT_EMBEDDING_ENGINE", "clip_v1").strip() or "clip_v1"

SIGLIP2_ENABLED = _bool_env("SIGLIP2_ENABLED", False)
SIGLIP2_MODEL_ID = os.environ.get("SIGLIP2_MODEL_ID", "google/siglip2-so400m-patch14-384").strip()

# IMPORTANT -- revision pinning:
# This service was built in a sandbox with no outbound access to
# huggingface.co, so a specific upstream commit SHA for
# google/siglip2-so400m-patch14-384 could not be resolved from here. "main"
# is a real, resolvable git ref (never a fabricated hash), used only as a
# development-time placeholder. Before enabling SIGLIP2_ENABLED in any
# staging or production environment, an operator with real network access
# MUST resolve and set an explicit commit revision -- see
# docs/P13_SIGLIP2_AB_IMPLEMENTATION.md, section "Pinning the SigLIP2
# revision" -- and set SIGLIP2_MODEL_REVISION to that commit hash so the
# model can never silently drift underneath a running deployment.
SIGLIP2_MODEL_REVISION = os.environ.get("SIGLIP2_MODEL_REVISION", "main").strip() or "main"

SIGLIP2_DEVICE = os.environ.get("SIGLIP2_DEVICE", "cpu").strip().lower() or "cpu"
SIGLIP2_LOAD_ON_START = _bool_env("SIGLIP2_LOAD_ON_START", False)
SIGLIP2_EXPECTED_DIMENSION = _int_env("SIGLIP2_EXPECTED_DIMENSION", 1152)
SIGLIP2_INFERENCE_TIMEOUT_MS = _int_env("SIGLIP2_INFERENCE_TIMEOUT_MS", 60000)
SIGLIP2_TEXT_MAX_TOKENS = _int_env("SIGLIP2_TEXT_MAX_TOKENS", 64)

AB_TEST_ENABLED = _bool_env("AB_TEST_ENABLED", False)
AB_UI_ENABLED = _bool_env("AB_UI_ENABLED", False)

# SigLIP2 scoring is a separate, uncalibrated space -- CLIP's minScore/
# minMargin thresholds (CONF_MIN_SCORE/CONF_MIN_MARGIN in app.py) must
# never be reused here. These exist so a future calibration pass has a
# named place to put real, evaluation-derived thresholds without touching
# CLIP's configuration.
SIGLIP2_CALIBRATION_STATUS = "uncalibrated"
SIGLIP2_MIN_SCORE = os.environ.get("SIGLIP2_MIN_SCORE")
SIGLIP2_MIN_MARGIN = os.environ.get("SIGLIP2_MIN_MARGIN")

V2_SCORING_VERSION = os.environ.get("V2_SCORING_VERSION", "engine-match-v2").strip() or "engine-match-v2"
