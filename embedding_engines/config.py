"""Environment-controlled configuration for the multi-engine (P13) surface.

Nothing here is read by the legacy CLIP endpoints in app.py -- those keep
using their own existing constants untouched. This module only backs the
new engine registry and the /v2 API / A/B / Festival "AI Model Comparison"
surfaces, all of which default to off.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


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
# never be reused here. SIGLIP2_MIN_SCORE/SIGLIP2_MIN_MARGIN, when set
# directly via environment, are an explicit operator override and always
# win over calibration.json (see get_siglip2_min_score()/
# get_siglip2_min_margin() below). CLIP's own configuration is completely
# untouched by anything in this module.
SIGLIP2_MIN_SCORE = os.environ.get("SIGLIP2_MIN_SCORE")
SIGLIP2_MIN_MARGIN = os.environ.get("SIGLIP2_MIN_MARGIN")

# D-4: scripts/calibrate.py sweeps a labelled eval set and writes the
# winning (minScore, minMargin) pair here, together with the date the
# sweep was run. Presence of a *valid* file at this path is what flips
# get_siglip2_calibration_status() from "uncalibrated" to
# "calibrated:<date>" -- never a manual flag, so the status can never say
# "calibrated" without a real file backing it.
SIGLIP2_CALIBRATION_FILE = (
    os.environ.get("SIGLIP2_CALIBRATION_FILE", "docs/eval/calibration.json").strip() or "docs/eval/calibration.json"
)

_REQUIRED_CALIBRATION_KEYS = ("minScore", "minMargin", "evalDate")


def _load_siglip2_calibration_file() -> Optional[Dict[str, Any]]:
    """Reads and validates SIGLIP2_CALIBRATION_FILE. Returns None -- never
    a fabricated/default calibration -- if the file is missing, unreadable,
    not a JSON object, or missing any required key. Re-read on every call
    rather than cached at import time, so a calibration.json that appears
    (or is replaced) while the process is running takes effect without a
    restart, and so tests can flip SIGLIP2_CALIBRATION_FILE per-case
    without needing to reimport this module.
    """
    path = Path(SIGLIP2_CALIBRATION_FILE)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not all(key in data for key in _REQUIRED_CALIBRATION_KEYS):
        return None
    try:
        min_score = float(data["minScore"])
        min_margin = float(data["minMargin"])
    except (TypeError, ValueError):
        return None
    eval_date = str(data["evalDate"]).strip()
    if not eval_date:
        return None
    return {"minScore": min_score, "minMargin": min_margin, "evalDate": eval_date}


def get_siglip2_calibration_status() -> str:
    """"uncalibrated" until a valid calibration.json exists, then
    "calibrated:<evalDate>" from that file -- the exact field/behavior D-4
    asks for. Never returns "calibrated" without evalDate: a bare
    "calibrated" with no date would be exactly the kind of unverifiable
    claim this whole calibration mechanism exists to avoid.
    """
    calibration = _load_siglip2_calibration_file()
    if calibration is None:
        return "uncalibrated"
    return f"calibrated:{calibration['evalDate']}"


def get_siglip2_min_score() -> Optional[float]:
    """SIGLIP2_MIN_SCORE (an explicit operator override) wins if set;
    otherwise calibration.json's minScore if present; otherwise None
    (uncalibrated -- callers must not gate on this)."""
    if SIGLIP2_MIN_SCORE is not None and str(SIGLIP2_MIN_SCORE).strip() != "":
        try:
            return float(SIGLIP2_MIN_SCORE)
        except (TypeError, ValueError):
            pass
    calibration = _load_siglip2_calibration_file()
    return calibration["minScore"] if calibration else None


def get_siglip2_min_margin() -> float:
    """Same precedence as get_siglip2_min_score(); defaults to 0.0 (no
    margin requirement) only once a min_score is actually in effect --
    callers gate on get_siglip2_min_score() being non-None first."""
    if SIGLIP2_MIN_MARGIN is not None and str(SIGLIP2_MIN_MARGIN).strip() != "":
        try:
            return float(SIGLIP2_MIN_MARGIN)
        except (TypeError, ValueError):
            pass
    calibration = _load_siglip2_calibration_file()
    return calibration["minMargin"] if calibration else 0.0


V2_SCORING_VERSION = os.environ.get("V2_SCORING_VERSION", "engine-match-v2").strip() or "engine-match-v2"
