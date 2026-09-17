"""
Festival console (docs/FESTIVAL_CLIP_CONSOLE.md; rebranded "Multimodal AL
Engine" in P14.1, see docs/P14_FESTIVAL_LIVE_GUI.md): GET "/"
used to return a bare {"status": "ok"} JSON body -- there was no browser
demonstration surface for the Dubai AI Festival, even though every other
control (signed identity, tenant/site scoping, demo governance, model
provenance, no-raw-embeddings) was already hardened and safe to describe
publicly. This proves the new "/" HTML console exists, is bilingual, and
-- most importantly -- proves that adding it did not touch, weaken, or
expose anything the hardened API already protects: /docs, /openapi.json,
/health/live and /health/ready behave exactly as before, /items, /match,
and /analyze-image remain fully protected, and neither the rendered page
nor its JavaScript ever contains a service-signing secret, a Mongo URI,
or a raw embedding.
"""
import pathlib
import re
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as clip_service
import internal_auth
from test_app import FakeModel, FakeProcessor, TEST_SECRET, identity_headers, make_image_bytes

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class FestivalConsoleTests(unittest.TestCase):
    def setUp(self):
        self.auth_patch = patch.object(internal_auth, "INTERNAL_JWT_SECRET", TEST_SECRET)
        self.auth_patch.start()
        self.model_patch = patch.object(clip_service, "load_model", return_value=(FakeModel(), FakeProcessor()))
        self.model_patch.start()
        self.dimension_patch = patch.object(clip_service, "EXPECTED_EMBEDDING_DIMENSION", 3)
        self.dimension_patch.start()
        clip_service.embedding_dimension = 3
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        clip_service.rate_limit_windows.clear()
        self.client = TestClient(clip_service.app)

    def tearDown(self):
        self.dimension_patch.stop()
        self.model_patch.stop()
        self.auth_patch.stop()
        clip_service.embedding_dimension = None
        clip_service.preprocessing_version = None
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())

    # 1-2: GET / returns 200 text/html.
    def test_root_returns_200_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))

    # 3 (P14.1 rebrand): contains "Multimodal AL Engine", the product's
    # current visible name -- "CLIP AI Engine" was the pre-P14.1 name and
    # is deliberately no longer shown in the GUI (see
    # docs/P14_FESTIVAL_LIVE_GUI.md and tests/test_rebrand_p14_1.py for the
    # full rebrand test coverage).
    def test_root_contains_title(self):
        response = self.client.get("/")
        self.assertIn("Multimodal AL Engine", response.text)

    # 4: contains "Dubai AI Festival 2026".
    def test_root_contains_festival_banner(self):
        response = self.client.get("/")
        self.assertIn("Dubai AI Festival 2026", response.text)

    # 5: identifies demo data.
    def test_root_identifies_demo_data(self):
        response = self.client.get("/")
        self.assertIn("DEMO DATA", response.text)
        self.assertIn("بيانات تجريبية", response.text)

    # 6: Arabic mode/resources exist.
    def test_arabic_resources_exist_and_use_rtl(self):
        response = self.client.get("/")
        self.assertIn('data-lang="ar"', response.text)
        js = (REPO_ROOT / "static" / "clip-console.js").read_text(encoding="utf-8")
        self.assertIn('setAttribute("dir", "rtl")', js)
        self.assertIn("محرك الذكاء متعدد الوسائط", js)  # P14.1: Arabic for "Multimodal AL Engine"

    # 7: /docs still works.
    def test_docs_still_works(self):
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)

    # 8: /openapi.json still works.
    def test_openapi_json_still_works(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("openapi", response.json())

    # 9: /health/live remains unchanged (public, {"status": "ok", ...}).
    def test_health_live_unchanged(self):
        response = self.client.get("/health/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    # 10: /health/ready remains unchanged (public, same checks shape).
    def test_health_ready_unchanged(self):
        response = self.client.get("/health/ready")
        self.assertIn(response.status_code, (200, 503))
        body = response.json()
        self.assertIn("checks", body)
        self.assertIn("modelLoaded", body["checks"])
        self.assertIn("databaseReady", body["checks"])

    # 11: protected /items remains protected.
    def test_items_still_protected(self):
        response = self.client.post("/items")
        self.assertEqual(response.status_code, 401)

    # 12: protected /match remains protected.
    def test_match_still_protected(self):
        response = self.client.post("/match")
        self.assertEqual(response.status_code, 401)

    # 13: protected /analyze-image remains protected.
    def test_analyze_image_still_protected(self):
        response = self.client.post("/analyze-image")
        self.assertEqual(response.status_code, 401)

    # 14: GUI does not contain INTERNAL_JWT_SECRET. The console source is
    # allowed to *name* the env var in a comment (SECURITY_NOTES.md does
    # the same, for human readers) -- what must never appear is the
    # secret's actual value, and the value is never read into the HTML
    # response at all.
    def test_console_never_contains_jwt_secret_value(self):
        response = self.client.get("/")
        self.assertNotIn(TEST_SECRET, response.text)
        self.assertNotIn("INTERNAL_JWT_SECRET", response.text)
        for path in ("clip-console.html", "clip-console.js", "clip-console.css"):
            for directory in ("templates", "static"):
                candidate = REPO_ROOT / directory / path
                if candidate.is_file():
                    self.assertNotIn(TEST_SECRET, candidate.read_text(encoding="utf-8"))

    # 15: GUI JavaScript does not create/sign service JWTs.
    def test_console_js_never_signs_a_jwt(self):
        js = (REPO_ROOT / "static" / "clip-console.js").read_text(encoding="utf-8")
        for forbidden in ("sign_internal_token", "HS256", "hmac", "jsonwebtoken", "signInternalToken"):
            self.assertNotIn(forbidden, js)
        self.assertNotIn("Authorization", js)

    # 16: GUI does not expose MongoDB URI. As with the JWT secret above,
    # the console source may *name* MONGODB_URI in a comment; it must
    # never contain a real connection string, and the rendered page must
    # never contain one either.
    def test_console_never_contains_mongodb_uri(self):
        response = self.client.get("/")
        self.assertNotIn("mongodb://", response.text)
        self.assertNotIn("mongodb+srv://", response.text)
        js = (REPO_ROOT / "static" / "clip-console.js").read_text(encoding="utf-8")
        self.assertNotIn("mongodb://", js)
        self.assertNotIn("mongodb+srv://", js)

    # 17: no raw embedding/vector is rendered. A long run of comma-separated
    # floats (the shape a real 512-d embedding would take if ever dumped
    # into the page) must never appear anywhere in the rendered HTML or
    # the console's own JavaScript source.
    def test_console_never_renders_a_raw_embedding(self):
        vector_pattern = r"\[\s*-?\d+\.\d+(?:,\s*-?\d+\.\d+){10,}"
        response = self.client.get("/")
        self.assertNotRegex(response.text, vector_pattern)
        js = (REPO_ROOT / "static" / "clip-console.js").read_text(encoding="utf-8")
        self.assertNotRegex(js, vector_pattern)

    # 18: existing Backend interoperability (governance claim) tests still
    # pass -- exercised by running the full suite (see docs), spot-checked
    # here for a signed identity missing the demoData claim.
    def test_backend_governance_contract_still_enforced(self):
        headers = identity_headers(request_id="console-governance-check", actions=["corpus:read"], omit=("demoData",))
        response = self.client.get("/items", headers=headers)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "missing_demo_data")

    # 19: existing governance (site/tenant scoping) tests still pass --
    # spot-checked here for a site the identity is not permitted to use.
    def test_site_scoping_still_enforced(self):
        headers = identity_headers(request_id="console-site-check", site="site-not-permitted", sites=["site-1"])
        response = self.client.get("/items", headers=headers)
        self.assertEqual(response.status_code, 403)

    # 20: existing OCR/barcode tests still pass -- spot-checked here via a
    # real authenticated /analyze-image call still returning OCR/barcode
    # keys (full coverage lives in test_ocr_integration.py, run as part of
    # the complete suite per the acceptance report).
    def test_analyze_image_still_returns_ocr_and_barcode_shape(self):
        headers = identity_headers(request_id="console-analyze-check", actions=["match:execute"])
        response = self.client.post(
            "/analyze-image",
            headers=headers,
            files={"file": ("item.png", make_image_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("ocr", body)
        self.assertIn("barcode", body)
        self.assertIn("model", body)

    # Model provenance panel: real live values, never fabricated.
    def test_root_renders_real_model_provenance_not_a_fake_placeholder(self):
        response = self.client.get("/")
        self.assertIn(clip_service.MODEL_NAME, response.text)
        self.assertIn(clip_service.MODEL_REVISION, response.text)
        self.assertIn(clip_service.SCORING_VERSION, response.text)

    # Console CSP must not leak onto Swagger/ReDoc/OpenAPI.
    def test_csp_is_scoped_to_console_only(self):
        root = self.client.get("/")
        self.assertIn("Content-Security-Policy", root.headers)
        docs = self.client.get("/docs")
        self.assertNotIn("Content-Security-Policy", docs.headers)
        openapi = self.client.get("/openapi.json")
        self.assertNotIn("Content-Security-Policy", openapi.headers)


if __name__ == "__main__":
    unittest.main()
