"""F4: the public festival console ("/", "/ui", "/static/*") is served only
when CONSOLE_ENABLED=true (default false). Disabled, every console path is
indistinguishable from an unknown path; enabled (demo/festival builds),
the console is unchanged. Health, docs and the authenticated API are never
affected.
"""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as clip_service
import demo_config

CONSOLE_PATHS = ["/", "/ui", "/static/clip-console.css", "/static/clip-console.js"]
FESTIVAL_MARKERS = ["Festival", "Multimodal AL Engine", "Urban Intelligence", "clip-console", "<html"]


class ConsoleDisabledByDefaultTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(clip_service.app)

    def test_flag_defaults_to_off(self):
        self.assertFalse(clip_service.CONSOLE_ENABLED)

    def test_every_console_path_is_identical_to_an_unknown_path(self):
        unknown = self.client.get("/this-path-does-not-exist")
        self.assertEqual(unknown.status_code, 404)
        for path in CONSOLE_PATHS:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json(), unknown.json())
                self.assertEqual(response.headers.get("content-type"), unknown.headers.get("content-type"))
                # No console-only security headers that would reveal the page.
                self.assertNotIn("content-security-policy", response.headers)
                self.assertNotIn("x-frame-options", response.headers)

    def test_no_festival_text_is_served(self):
        for path in CONSOLE_PATHS:
            with self.subTest(path=path):
                body = self.client.get(path).text
                for marker in FESTIVAL_MARKERS:
                    self.assertNotIn(marker, body)

    def test_demo_facade_routes_stay_hidden_too(self):
        with patch.object(demo_config, "DEMO_FACADE_ENABLED", False):
            for path in ("/demo/status",):
                self.assertEqual(self.client.get(path).status_code, 404)

    def test_health_and_docs_are_unaffected(self):
        self.assertEqual(self.client.get("/health/live").status_code, 200)
        self.assertEqual(self.client.get("/docs").status_code, 200)
        self.assertEqual(self.client.get("/openapi.json").status_code, 200)

    def test_authenticated_api_is_unaffected(self):
        # Still reachable and still protected (401), not hidden.
        self.assertEqual(self.client.get("/items").status_code, 401)
        self.assertEqual(self.client.post("/match").status_code, 401)


class ConsoleEnabledDemoBuildTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(clip_service, "CONSOLE_ENABLED", True)
        self.patch.start()
        self.client = TestClient(clip_service.app)

    def tearDown(self):
        self.patch.stop()

    def test_console_pages_and_assets_are_served(self):
        for path in CONSOLE_PATHS:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_console_keeps_its_security_headers(self):
        response = self.client.get("/")
        self.assertIn("content-security-policy", response.headers)
        self.assertEqual(response.headers.get("x-frame-options"), "DENY")


if __name__ == "__main__":
    unittest.main()
