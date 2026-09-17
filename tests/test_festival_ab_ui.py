"""P13 section 24: the Festival console's "AI Model Comparison" section.

Follows the same security model already proven for the console's existing
"Image Analysis" card (see tests/test_festival_console.py): the public
console never signs a JWT and never calls a protected endpoint, so this
section is explanatory only and gated entirely server-side by
AB_UI_ENABLED, defaulting to hidden.
"""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as clip_service
import internal_auth
from embedding_engines import config as engine_config
from test_app import TEST_SECRET


class FestivalAbUiTests(unittest.TestCase):
    def setUp(self):
        self.auth_patch = patch.object(internal_auth, "INTERNAL_JWT_SECRET", TEST_SECRET)
        self.auth_patch.start()
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        self.client = TestClient(clip_service.app)

    def tearDown(self):
        self.auth_patch.stop()
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())

    def test_ab_section_hidden_by_default(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("ab-comparison-card", response.text)

    def test_ab_section_rendered_when_enabled(self):
        with patch.object(engine_config, "AB_UI_ENABLED", True):
            response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ab-comparison-card", response.text)
        self.assertIn("AI-assisted candidate retrieval", response.text)
        self.assertIn("uncalibrated", response.text)

    def test_ab_section_never_renders_raw_vector_placeholders(self):
        with patch.object(engine_config, "AB_UI_ENABLED", True):
            response = self.client.get("/")
        self.assertNotIn("[0.", response.text)

    def test_console_js_still_never_signs_a_jwt_or_calls_v2(self):
        js = open("static/clip-console.js", encoding="utf-8").read()
        self.assertNotIn("Authorization", js)
        self.assertNotIn("/v2/", js)

    def test_disabling_never_requires_a_database_change(self):
        # Rollback story (section 27): toggling AB_UI_ENABLED off is a pure
        # config change -- it never touches the repository/storage layer.
        with patch.object(engine_config, "AB_UI_ENABLED", True):
            self.client.get("/")
        with patch.object(engine_config, "AB_UI_ENABLED", False):
            response = self.client.get("/")
        self.assertNotIn("ab-comparison-card", response.text)


if __name__ == "__main__":
    unittest.main()
