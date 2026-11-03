"""F1: pilot configuration for CLIP-Service.

1. DEMO_FACADE_ENABLED=false (the default) keeps EVERY /demo/* route
   returning 404 -- enumerated from the live app's route table, so a route
   added later is covered automatically -- even when every other demo
   setting is fully configured.
2. The internal JWT issuer defaults to the neutral "lost-found-backend"
   (matching the Backend's CLIP_INTERNAL_JWT_ISSUER default), stays
   overridable via INTERNAL_JWT_ISSUER, and accepts an extra issuer only
   when one is explicitly listed in INTERNAL_JWT_ACCEPTED_ISSUERS.
"""
import importlib.util
import itertools
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import app as clip_service
import demo_config
import internal_auth
from test_app import TEST_SECRET

ROOT = Path(__file__).resolve().parents[1]


def _iter_routes(routes):
    # FastAPI >= 0.13x wraps include_router() results in an included-router
    # object with no .path of its own; walk into it so routes mounted via
    # app.include_router(demo_router) are enumerated too.
    for route in routes:
        nested = getattr(route, "original_router", None)
        if nested is not None:
            yield from _iter_routes(nested.routes)
        else:
            yield route


def _demo_routes():
    routes = []
    for route in _iter_routes(clip_service.app.routes):
        path = getattr(route, "path", "")
        if path == "/demo" or path.startswith("/demo/"):
            concrete = re.sub(r"\{[^}]+\}", "wallet-01.jpg", path)
            for method in sorted(getattr(route, "methods", None) or {"GET"}):
                if method in {"HEAD", "OPTIONS"}:
                    continue
                routes.append((method, concrete))
    return routes


_ISOLATED_COUNTER = itertools.count()


def _load_isolated(module_name: str, env_overrides: dict):
    """Execute a fresh, private copy of a top-level module under a patched
    environment, so its module-level env parsing runs exactly as it does at
    service start-up. The copy gets a unique name and is registered in
    sys.modules only while it executes (dataclasses need that), so the
    shared internal_auth/demo_config modules the app uses are untouched."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("INTERNAL_JWT_", "DEMO_"))}
    env.update(env_overrides)
    spec = importlib.util.spec_from_file_location(
        f"_f1_isolated_{module_name}_{next(_ISOLATED_COUNTER)}", ROOT / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
    return module


class DemoFacadeDisabledForPilotTests(unittest.TestCase):
    def setUp(self):
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        clip_service.rate_limit_windows.clear()
        self.client = TestClient(clip_service.app)
        self._tmpdir = tempfile.TemporaryDirectory()
        (Path(self._tmpdir.name) / "candidates").mkdir()
        (Path(self._tmpdir.name) / "candidates" / "wallet-01.jpg").write_bytes(b"\xff\xd8\xff")

    def tearDown(self):
        self._tmpdir.cleanup()
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        clip_service.rate_limit_windows.clear()

    def test_demo_routes_are_mounted(self):
        # Guard against the enumeration below silently testing nothing.
        paths = {path for _method, path in _demo_routes()}
        self.assertIn("/demo/status", paths)
        self.assertIn("/demo/match", paths)
        self.assertIn("/demo/assets/candidates/wallet-01.jpg", paths)

    def test_every_demo_route_returns_404_when_flag_false_even_if_fully_configured(self):
        with patch.object(demo_config, "DEMO_FACADE_ENABLED", False), patch.object(
            demo_config, "DEMO_TENANT_ID", "pilot-tenant"
        ), patch.object(demo_config, "DEMO_SITE_ID", "pilot-site"), patch.object(
            demo_config, "DEMO_DATASET_VERSION", "pilot-v1"
        ), patch.object(demo_config, "DEMO_ASSET_DIR", self._tmpdir.name):
            self.assertFalse(demo_config.demo_config_ready())
            for method, path in _demo_routes():
                with self.subTest(method=method, path=path):
                    if method == "POST":
                        response = self.client.post(path, data={"text": "red wallet", "mode": "clip"})
                    else:
                        response = self.client.request(method, path)
                    self.assertEqual(response.status_code, 404)
                    # Same not-found body as any unknown route: no hint the
                    # facade exists, no tenant/site/dataset value echoed.
                    body = response.text
                    for value in ("pilot-tenant", "pilot-site", "pilot-v1", self._tmpdir.name):
                        self.assertNotIn(value, body)

    def test_demo_facade_enabled_false_env_value_disables_facade_at_startup(self):
        for raw in ("false", "False", "0", "no", "", "off"):
            with self.subTest(DEMO_FACADE_ENABLED=raw):
                isolated = _load_isolated(
                    "demo_config",
                    {
                        "DEMO_FACADE_ENABLED": raw,
                        "DEMO_TENANT_ID": "pilot-tenant",
                        "DEMO_SITE_ID": "pilot-site",
                        "DEMO_DATASET_VERSION": "pilot-v1",
                        "DEMO_ASSET_DIR": "/opt/demo-data",
                    },
                )
                self.assertFalse(isolated.DEMO_FACADE_ENABLED)
                self.assertFalse(isolated.demo_config_ready())

    def test_demo_facade_unset_defaults_to_disabled(self):
        self.assertFalse(_load_isolated("demo_config", {}).DEMO_FACADE_ENABLED)


def _token(issuer: str) -> str:
    return internal_auth.sign_internal_token(
        {
            "iss": issuer,
            "aud": internal_auth.JWT_AUDIENCE,
            "service": "backend-api",
            "tenantId": "tenant-a",
            "siteId": "site-1",
            "siteIds": ["site-1"],
            "actions": ["match:execute"],
            "requestId": "req-f1",
            "demoData": False,
            "datasetVersion": None,
            "exp": int(time.time()) + 60,
        },
        TEST_SECRET,
    )


class IssuerConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.secret_patch = patch.object(internal_auth, "INTERNAL_JWT_SECRET", TEST_SECRET)
        self.secret_patch.start()

    def tearDown(self):
        self.secret_patch.stop()

    def test_default_issuer_is_neutral(self):
        self.assertEqual(internal_auth.DEFAULT_JWT_ISSUER, "lost-found-backend")
        isolated = _load_isolated("internal_auth", {})
        self.assertEqual(isolated.JWT_ISSUER, "lost-found-backend")
        self.assertEqual(isolated.JWT_ACCEPTED_ISSUERS, frozenset({"lost-found-backend"}))

    def test_issuer_overridable_via_configuration(self):
        isolated = _load_isolated("internal_auth", {"INTERNAL_JWT_ISSUER": "pilot-backend"})
        self.assertEqual(isolated.JWT_ISSUER, "pilot-backend")
        self.assertEqual(isolated.JWT_ACCEPTED_ISSUERS, frozenset({"pilot-backend"}))

    def test_blank_issuer_env_falls_back_to_default_not_empty(self):
        isolated = _load_isolated("internal_auth", {"INTERNAL_JWT_ISSUER": "  "})
        self.assertEqual(isolated.JWT_ISSUER, "lost-found-backend")

    def test_accepted_issuers_list_parsed_for_rollout(self):
        isolated = _load_isolated("internal_auth", {"INTERNAL_JWT_ACCEPTED_ISSUERS": " dubai-festival-backend , ,"})
        self.assertEqual(isolated.JWT_ACCEPTED_ISSUERS, frozenset({"dubai-festival-backend", "lost-found-backend"}))

    def test_configured_issuer_accepted(self):
        with patch.object(internal_auth, "JWT_ACCEPTED_ISSUERS", frozenset({"lost-found-backend"})):
            claims = internal_auth.decode_internal_jwt(_token("lost-found-backend"))
        self.assertEqual(claims["iss"], "lost-found-backend")

    def test_legacy_festival_issuer_rejected_by_default(self):
        with patch.object(internal_auth, "JWT_ACCEPTED_ISSUERS", frozenset({"lost-found-backend"})):
            with self.assertRaises(HTTPException) as ctx:
                internal_auth.decode_internal_jwt(_token("dubai-festival-backend"))
        self.assertEqual(ctx.exception.status_code, 401)

    def test_legacy_issuer_accepted_only_when_explicitly_listed(self):
        accepted = frozenset({"lost-found-backend", "dubai-festival-backend"})
        with patch.object(internal_auth, "JWT_ACCEPTED_ISSUERS", accepted):
            claims = internal_auth.decode_internal_jwt(_token("dubai-festival-backend"))
            self.assertEqual(claims["iss"], "dubai-festival-backend")
            with self.assertRaises(HTTPException):
                internal_auth.decode_internal_jwt(_token("untrusted-backend"))


if __name__ == "__main__":
    unittest.main()
