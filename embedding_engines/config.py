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
#
# D-3: this used to be documentation-only -- nothing stopped a deployment
# from setting SIGLIP2_ENABLED=true while leaving SIGLIP2_MODEL_REVISION at
# its "main" default. validate_siglip2_configuration() (called from
# app.py's initialize_runtime(), unconditionally and before any engine
# warmup) now turns that into a hard startup failure.
SIGLIP2_MODEL_REVISION = os.environ.get("SIGLIP2_MODEL_REVISION", "main").strip() or "main"

# D-3: revision values that can silently point at different actual weights
# over time -- a mutable branch ref, or "no value configured at all" (which
# the assignment above already folds into "main"). Compared case-
# insensitively so "Main"/"LATEST" etc. are caught too.
_UNPINNED_SIGLIP2_REVISIONS = frozenset({"", "main", "latest"})


def validate_siglip2_configuration() -> None:
    """D-3: reject startup if SigLIP2 is enabled but its revision is not a
    real pinned commit. Checked unconditionally whenever SIGLIP2_ENABLED is
    true -- regardless of SIGLIP2_LOAD_ON_START -- so a misconfigured
    deployment fails fast at process startup rather than silently serving
    an unpinned, driftable model the first time SigLIP2 is actually
    invoked (which, with the default SIGLIP2_LOAD_ON_START=false, could be
    long after startup, on a live request).

    This is a configuration-validation failure, not a transient load/
    network error: it raises (like internal_auth.validate_auth_configuration()
    and app.py's configured_device()) rather than logging and continuing,
    which is the deliberate, documented behavior of Siglip2Engine.load()/
    warmup() for genuine runtime failures.
    """
    if not SIGLIP2_ENABLED:
        return
    if SIGLIP2_MODEL_REVISION.strip().lower() in _UNPINNED_SIGLIP2_REVISIONS:
        raise RuntimeError(
            "SIGLIP2_MODEL_REVISION must be pinned to a real commit SHA when SIGLIP2_ENABLED=true "
            f"(got {SIGLIP2_MODEL_REVISION!r}). See docs/P13_SIGLIP2_AB_IMPLEMENTATION.md, "
            "section 'Pinning the SigLIP2 revision', for how to resolve and set one."
        )

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
