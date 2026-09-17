"""P14: environment-controlled configuration for the public Dubai AI
Festival demo facade (demo_router.py). Disabled by default.

Deliberately standalone (no import of embedding_engines/app) so it can be
imported early and cheaply, and so patching it in tests never risks a
circular import.
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


DEMO_FACADE_ENABLED = _bool_env("DEMO_FACADE_ENABLED", False)
DEMO_TENANT_ID = os.environ.get("DEMO_TENANT_ID", "").strip()
DEMO_SITE_ID = os.environ.get("DEMO_SITE_ID", "").strip()
DEMO_DATASET_VERSION = os.environ.get("DEMO_DATASET_VERSION", "").strip()
DEMO_ASSET_DIR = os.environ.get("DEMO_ASSET_DIR", "").strip()
DEMO_RATE_LIMIT_PER_MINUTE = _int_env("DEMO_RATE_LIMIT_PER_MINUTE", 12)
DEMO_TOP_K = _int_env("DEMO_TOP_K", 3)

# Not an operator-facing env var (not in the required list) -- the facade
# always calls back into this same process over loopback only.
DEMO_INTERNAL_BASE_URL = os.environ.get("DEMO_INTERNAL_BASE_URL", "").strip() or "http://127.0.0.1:{}".format(
    os.environ.get("PORT", "8080").strip() or "8080"
)

# Hard cap, not configurable: "exp <= current time + 120 seconds" (P14 spec
# section 2). A shorter TTL is fine; a longer one is not.
DEMO_JWT_TTL_SECONDS = 120


def demo_config_ready() -> bool:
    """Fail closed: the facade is only really usable when the master flag
    AND every other required value are present. A half-configured facade
    (flag on, tenant/site/dataset/asset-dir missing) behaves exactly like
    a disabled one -- it never falls back to a blank/default tenant."""
    return bool(
        DEMO_FACADE_ENABLED
        and DEMO_TENANT_ID
        and DEMO_SITE_ID
        and DEMO_DATASET_VERSION
        and DEMO_ASSET_DIR
    )
