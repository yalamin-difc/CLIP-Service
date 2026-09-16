"""P14: tests for the public Dubai AI Festival live demo facade
(demo_config.py, demo_router.py). Model inference is always mocked here
(via demo_router._forward) -- no real network call is made to /v2/match
or /v2/ab/match, and SigLIP2 is never downloaded in CI, per the P14 spec.
"""
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app as clip_service
import demo_config
import demo_router
import internal_auth
from test_app import TEST_SECRET, make_image_bytes


CLIP_SINGLE_BODY = {
    "governance": {
        "latencyMs": 42,
        "modelId": "openai/clip-vit-base-patch32",
        "embeddingDimension": 512,
        "calibrationStatus": "calibrated",
    },
    "topK": [
        {
            "candidateId": "item-1",
            "score": 0.91,
            "item": {
                "title": "Black wallet",
                "description": "A black leather wallet",
                "candidateFilename": "wallet-01.jpg",
            },
            "explanation": {"summary": "visual similarity", "reason": "visual similarity"},
        },
        {
            "candidateId": "item-2",
            "score": 0.55,
            "item": {"title": "Brown wallet", "description": None, "candidateFilename": None},
            "explanation": {"summary": "visual similarity", "reason": "visual similarity"},
        },
    ],
}

SIGLIP2_SINGLE_BODY = {
    "governance": {
        "latencyMs": 2100,
        "modelId": "google/siglip2-so400m-patch14-384",
        "embeddingDimension": 1152,
        "calibrationStatus": "uncalibrated",
    },
    "topK": [
        {
            "candidateId": "item-1",
            "score": 0.62,
            "item": {"title": "Black wallet", "description": "A black leather wallet", "candidateFilename": "wallet-01.jpg"},
            "explanation": {"summary": "visual similarity", "reason": "visual similarity"},
        }
    ],
}

COMPARE_BODY = {
    "clip": {
        "status": "success",
        "provenance": {"modelId": "openai/clip-vit-base-patch32", "embeddingDimension": 512, "calibrationStatus": "calibrated"},
        "latencyMs": 40,
        "candidates": CLIP_SINGLE_BODY["topK"],
    },
    "siglip2": {
        "status": "success",
        "provenance": {
            "modelId": "google/siglip2-so400m-patch14-384",
            "embeddingDimension": 1152,
            "calibrationStatus": "uncalibrated",
        },
        "latencyMs": 2050,
        "candidates": SIGLIP2_SINGLE_BODY["topK"],
        "calibrationStatus": "uncalibrated",
    },
    "comparison": {"sameTop1": True, "topKOverlap": 1, "rankChanges": [], "latencyDifferenceMs": 2010},
}

COMPARE_BODY_SIGLIP2_DOWN = {
    "clip": COMPARE_BODY["clip"],
    "siglip2": {"status": "unavailable", "errorCode": "engine_disabled"},
    "comparison": {"sameTop1": None, "topKOverlap": None, "rankChanges": None, "latencyDifferenceMs": None},
}


class DemoFacadeTestsBase(unittest.TestCase):
    def setUp(self):
        self.auth_patch = patch.object(internal_auth, "INTERNAL_JWT_SECRET", TEST_SECRET)
        self.auth_patch.start()
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        clip_service.rate_limit_windows.clear()
        demo_router._rate_limit_windows.clear()
        self.client = TestClient(clip_service.app)

        self._tmpdir = tempfile.TemporaryDirectory()
        self.asset_dir = Path(self._tmpdir.name)
        (self.asset_dir / "candidates").mkdir(parents=True, exist_ok=True)

        self.patches = [
            patch.object(demo_config, "DEMO_FACADE_ENABLED", True),
            patch.object(demo_config, "DEMO_TENANT_ID", "festival-demo"),
            patch.object(demo_config, "DEMO_SITE_ID", "dubai-ai-festival"),
            patch.object(demo_config, "DEMO_DATASET_VERSION", "daf-2026-v1"),
            patch.object(demo_config, "DEMO_ASSET_DIR", str(self.asset_dir)),
            patch.object(demo_config, "DEMO_RATE_LIMIT_PER_MINUTE", 1000),
            patch.object(demo_config, "DEMO_TOP_K", 3),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self._tmpdir.cleanup()
        self.auth_patch.stop()
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        clip_service.rate_limit_windows.clear()
        demo_router._rate_limit_windows.clear()

    def _write_candidate_image(self, filename="wallet-01.jpg"):
        image = Image.new("RGB", (4, 4), (10, 20, 30))
        image.save(self.asset_dir / "candidates" / filename)


class FacadeDisabledTests(unittest.TestCase):
    """1-3: disabled by default, and fails closed on incomplete config --
    deliberately does NOT use DemoFacadeTestsBase's enabling patches."""

    def setUp(self):
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        clip_service.rate_limit_windows.clear()
        self.client = TestClient(clip_service.app)

    def tearDown(self):
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        clip_service.rate_limit_windows.clear()

    def test_1_facade_disabled_by_default(self):
        self.assertFalse(demo_config.DEMO_FACADE_ENABLED)

    def test_2_demo_status_unavailable_when_disabled(self):
        with patch.object(demo_config, "DEMO_FACADE_ENABLED", False):
            response = self.client.get("/demo/status")
        self.assertEqual(response.status_code, 404)

    def test_2b_demo_match_unavailable_when_disabled(self):
        with patch.object(demo_config, "DEMO_FACADE_ENABLED", False):
            response = self.client.post("/demo/match", data={"text": "red wallet", "mode": "clip"})
        self.assertEqual(response.status_code, 404)

    def test_3_missing_demo_config_fails_closed(self):
        # Flag on, but every other required value left blank -- must behave
        # exactly like disabled, never fall back to an empty tenant/site.
        with patch.object(demo_config, "DEMO_FACADE_ENABLED", True), patch.object(
            demo_config, "DEMO_TENANT_ID", ""
        ), patch.object(demo_config, "DEMO_SITE_ID", ""), patch.object(
            demo_config, "DEMO_DATASET_VERSION", ""
        ), patch.object(demo_config, "DEMO_ASSET_DIR", ""):
            self.assertFalse(demo_config.demo_config_ready())
            response = self.client.get("/demo/status")
            self.assertEqual(response.status_code, 404)

    def test_3b_partially_configured_still_fails_closed(self):
        with patch.object(demo_config, "DEMO_FACADE_ENABLED", True), patch.object(
            demo_config, "DEMO_TENANT_ID", "festival-demo"
        ), patch.object(demo_config, "DEMO_SITE_ID", ""), patch.object(
            demo_config, "DEMO_DATASET_VERSION", "daf-2026-v1"
        ), patch.object(demo_config, "DEMO_ASSET_DIR", "/opt/demo-data"):
            response = self.client.get("/demo/status")
            self.assertEqual(response.status_code, 404)


class ServerControlledIdentityTests(DemoFacadeTestsBase):
    def test_4_client_cannot_select_tenant_site_dataset_or_engine(self):
        captured = {}

        async def fake_forward(path, *, engine, file_field, text):
            captured["path"] = path
            captured["engine"] = engine
            return CLIP_SINGLE_BODY

        with patch.object(demo_router, "_forward", side_effect=fake_forward):
            response = self.client.post(
                "/demo/match",
                data={
                    "text": "red wallet",
                    "mode": "clip",
                    # A malicious/naive client trying to smuggle system fields --
                    # none of these are accepted by the endpoint signature, so
                    # FastAPI silently drops them; this proves they never reach
                    # the forwarded call.
                    "tenantId": "attacker-tenant",
                    "siteId": "attacker-site",
                    "datasetVersion": "attacker-dataset",
                    "demoData": "false",
                    "actions": "corpus:write",
                    "engine": "siglip2_v1",
                },
            )
        self.assertEqual(response.status_code, 200)
        # mode=clip always forwards engine=clip_v1, never the client-supplied "engine" field.
        self.assertEqual(captured["engine"], "clip_v1")

    def test_5_minted_identity_is_demodata_true(self):
        token, request_id = demo_router._mint_demo_identity()
        claims = internal_auth.decode_internal_jwt(token)
        self.assertIs(claims["demoData"], True)
        self.assertEqual(claims["requestId"], request_id)

    def test_6_minted_identity_uses_configured_dataset_version(self):
        token, _request_id = demo_router._mint_demo_identity()
        claims = internal_auth.decode_internal_jwt(token)
        self.assertEqual(claims["datasetVersion"], "daf-2026-v1")
        self.assertEqual(claims["tenantId"], "festival-demo")
        self.assertEqual(claims["siteId"], "dubai-ai-festival")
        self.assertEqual(claims["siteIds"], ["dubai-ai-festival"])
        self.assertEqual(claims["actions"], ["match:execute"])
        self.assertLessEqual(claims["exp"], int(time.time()) + 120)

    def test_7_browser_never_receives_a_jwt(self):
        with patch.object(demo_router, "_forward", side_effect=self._fake(CLIP_SINGLE_BODY)):
            response = self.client.post("/demo/match", data={"text": "red wallet", "mode": "clip"})
        self.assertEqual(response.status_code, 200)
        body_text = response.text
        self.assertNotIn("eyJ", body_text)  # base64url JWT header prefix ('{"alg"')
        self.assertNotIn("Bearer", body_text)
        status_response = self.client.get("/demo/status")
        self.assertNotIn("eyJ", status_response.text)

    @staticmethod
    def _fake(body, status=200):
        async def _inner(path, *, engine, file_field, text):
            return status, body

        return _inner


class SafeResponseShapeTests(DemoFacadeTestsBase):
    def test_8_no_raw_vectors_in_match_response(self):
        with patch.object(demo_router, "_forward", side_effect=self._forward(CLIP_SINGLE_BODY)):
            response = self.client.post("/demo/match", data={"text": "red wallet", "mode": "clip"})
        self.assertNotIn('"vector"', response.text)
        self.assertNotRegex(response.text, r"\[\s*-?\d+\.\d+(?:,\s*-?\d+\.\d+){10,}")

    def test_8b_no_raw_vectors_in_status_response(self):
        response = self.client.get("/demo/status")
        self.assertNotIn('"vector"', response.text)

    def test_16_candidate_image_url_is_safe_basename_only(self):
        self._write_candidate_image("wallet-01.jpg")
        body = json.loads(json.dumps(CLIP_SINGLE_BODY))
        body["topK"][0]["item"]["candidateFilename"] = "../../etc/passwd.jpg"
        with patch.object(demo_router, "_forward", side_effect=self._forward(body)):
            response = self.client.post("/demo/match", data={"text": "red wallet", "mode": "clip"})
        candidates = response.json()["candidates"]
        self.assertIsNone(candidates[0]["imageUrl"])

    def test_16b_candidate_image_url_uses_demo_assets_path(self):
        self._write_candidate_image("wallet-01.jpg")
        with patch.object(demo_router, "_forward", side_effect=self._forward(CLIP_SINGLE_BODY)):
            response = self.client.post("/demo/match", data={"text": "red wallet", "mode": "clip"})
        candidates = response.json()["candidates"]
        self.assertEqual(candidates[0]["imageUrl"], "/demo/assets/candidates/wallet-01.jpg")
        self.assertIsNone(candidates[1]["imageUrl"])  # no candidateFilename supplied

    @staticmethod
    def _forward(body, status=200):
        async def _inner(path, *, engine, file_field, text):
            return status, body

        return _inner


class AssetTraversalTests(DemoFacadeTestsBase):
    def test_9_dotdot_traversal_rejected(self):
        self._write_candidate_image("wallet-01.jpg")
        for attempt in ("..%2F..%2Fetc%2Fpasswd", "..", "%2e%2e", "..%2Fwallet-01.jpg"):
            response = self.client.get(f"/demo/assets/candidates/{attempt}")
            self.assertEqual(response.status_code, 404, attempt)

    def test_9b_absolute_path_style_rejected(self):
        response = self.client.get("/demo/assets/candidates/etc%2Fpasswd")
        self.assertEqual(response.status_code, 404)

    def test_10_unsupported_extension_rejected(self):
        (self.asset_dir / "candidates" / "evil.exe").write_bytes(b"not-an-image")
        response = self.client.get("/demo/assets/candidates/evil.exe")
        self.assertEqual(response.status_code, 404)

    def test_valid_candidate_asset_served(self):
        self._write_candidate_image("wallet-01.jpg")
        response = self.client.get("/demo/assets/candidates/wallet-01.jpg")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")

    def test_missing_asset_returns_404_not_500(self):
        response = self.client.get("/demo/assets/candidates/does-not-exist.png")
        self.assertEqual(response.status_code, 404)


class MimeValidationTests(DemoFacadeTestsBase):
    def test_10_unsupported_mime_rejected_on_match(self):
        with patch.object(demo_router, "_forward", side_effect=SafeResponseShapeTests._forward(CLIP_SINGLE_BODY)):
            response = self.client.post(
                "/demo/match",
                data={"mode": "clip"},
                files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
            )
        self.assertEqual(response.status_code, 415)


class RateLimitTests(DemoFacadeTestsBase):
    def test_11_rate_limit_enforced(self):
        with patch.object(demo_config, "DEMO_RATE_LIMIT_PER_MINUTE", 2), patch.object(
            demo_router, "_forward", side_effect=SafeResponseShapeTests._forward(CLIP_SINGLE_BODY)
        ):
            statuses = [
                self.client.post("/demo/match", data={"text": "red wallet", "mode": "clip"}).status_code
                for _ in range(4)
            ]
        self.assertEqual(statuses[:2], [200, 200])
        self.assertTrue(all(code == 429 for code in statuses[2:]))


class ModeTests(DemoFacadeTestsBase):
    def test_12_clip_mode_works(self):
        with patch.object(demo_router, "_forward", side_effect=self._forward("/v2/match", CLIP_SINGLE_BODY)):
            response = self.client.post("/demo/match", data={"text": "red wallet", "mode": "clip"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "clip")
        self.assertEqual(body["status"], "success")
        self.assertEqual(len(body["candidates"]), 2)
        self.assertEqual(body["candidates"][0]["rank"], 1)
        self.assertEqual(body["model"]["modelId"], "openai/clip-vit-base-patch32")
        self.assertNotIn("calibrationStatus", body["model"])  # clip has no calibration badge

    def test_13_siglip2_mode_works(self):
        with patch.object(demo_router, "_forward", side_effect=self._forward("/v2/match", SIGLIP2_SINGLE_BODY)):
            response = self.client.post("/demo/match", data={"text": "red wallet", "mode": "siglip2"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "siglip2")
        self.assertEqual(body["model"]["calibrationStatus"], "uncalibrated")

    def test_14_compare_mode_works(self):
        with patch.object(demo_router, "_forward", side_effect=self._forward("/v2/ab/match", COMPARE_BODY)):
            response = self.client.post("/demo/match", data={"text": "red wallet", "mode": "compare"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "compare")
        self.assertEqual(body["clip"]["status"], "success")
        self.assertEqual(body["siglip2"]["status"], "success")
        self.assertEqual(body["siglip2"]["calibrationStatus"], "uncalibrated")
        self.assertIn("sameTop1", body["comparison"])
        self.assertIn("topKOverlap", body["comparison"])
        self.assertIn("latencyDifferenceMs", body["comparison"])
        self.assertNotIn("winner", body["comparison"])
        self.assertNotIn("winner", json.dumps(body))

    def test_15_one_engine_failure_does_not_destroy_other_result(self):
        with patch.object(demo_router, "_forward", side_effect=self._forward("/v2/ab/match", COMPARE_BODY_SIGLIP2_DOWN)):
            response = self.client.post("/demo/match", data={"text": "red wallet", "mode": "compare"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["clip"]["status"], "success")
        self.assertEqual(body["siglip2"]["status"], "unavailable")
        self.assertEqual(body["siglip2"]["candidates"], [])
        self.assertTrue(len(body["clip"]["candidates"]) > 0)

    def test_invalid_mode_rejected(self):
        response = self.client.post("/demo/match", data={"text": "red wallet", "mode": "bogus"})
        self.assertEqual(response.status_code, 400)

    @staticmethod
    def _forward(expected_path, body, status=200):
        async def _inner(path, *, engine, file_field, text):
            assert path == expected_path
            return status, body

        return _inner


class QueryShapeTests(DemoFacadeTestsBase):
    ENGLISH = "black suitcase with a red ribbon on the handle"
    ARABIC = "حقيبة سفر سوداء عليها شريط أحمر على المقبض"
    MIXED = "Black Samsonite حقيبة with red ribbon"

    def test_17_arabic_text_accepted(self):
        captured = {}

        async def fake_forward(path, *, engine, file_field, text):
            captured["text"] = text
            return 200, CLIP_SINGLE_BODY

        with patch.object(demo_router, "_forward", side_effect=fake_forward):
            response = self.client.post("/demo/match", data={"text": self.ARABIC, "mode": "clip"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["text"], self.ARABIC)

    def test_18_mixed_arabic_english_text_accepted(self):
        captured = {}

        async def fake_forward(path, *, engine, file_field, text):
            captured["text"] = text
            return 200, CLIP_SINGLE_BODY

        with patch.object(demo_router, "_forward", side_effect=fake_forward):
            response = self.client.post("/demo/match", data={"text": self.MIXED, "mode": "clip"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["text"], self.MIXED)

    def test_19_text_only_request_works(self):
        with patch.object(demo_router, "_forward", side_effect=ModeTests._forward("/v2/match", CLIP_SINGLE_BODY)):
            response = self.client.post("/demo/match", data={"text": self.ENGLISH, "mode": "clip"})
        self.assertEqual(response.status_code, 200)

    def test_20_image_only_request_works(self):
        captured = {}

        async def fake_forward(path, *, engine, file_field, text):
            captured["has_file"] = file_field is not None
            captured["text"] = text
            return 200, CLIP_SINGLE_BODY

        with patch.object(demo_router, "_forward", side_effect=fake_forward):
            response = self.client.post(
                "/demo/match",
                data={"mode": "clip"},
                files={"file": ("item.png", make_image_bytes(), "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(captured["has_file"])
        self.assertIsNone(captured["text"])

    def test_21_request_without_image_or_text_rejected(self):
        response = self.client.post("/demo/match", data={"mode": "clip"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "missing_query")


class AuditTests(DemoFacadeTestsBase):
    def test_audit_event_recorded_for_demo_match(self):
        with patch.object(demo_router, "_forward", side_effect=ModeTests._forward("/v2/match", CLIP_SINGLE_BODY)):
            self.client.post("/demo/match", data={"text": "red wallet", "mode": "clip"})
        logs = clip_service.get_repository().list_audit_logs(
            _FakeIdentity(tenant_id="festival-demo"), limit=10
        )
        events = [log for log in logs if log["eventType"] == "festival_demo_match"]
        self.assertTrue(events)
        payload = events[0]["payload"]
        self.assertEqual(payload["mode"], "clip")
        self.assertEqual(payload["tenantId"], "festival-demo")
        self.assertEqual(payload["siteId"], "dubai-ai-festival")
        self.assertEqual(payload["datasetVersion"], "daf-2026-v1")
        self.assertTrue(payload["hasText"])
        self.assertFalse(payload["hasImage"])
        self.assertNotIn("vector", json.dumps(payload))


class _FakeIdentity:
    """Minimal stand-in for ServiceIdentity, just enough for
    InMemoryItemRepository.list_audit_logs' tenant filter."""

    def __init__(self, tenant_id):
        self.tenant_id = tenant_id


if __name__ == "__main__":
    unittest.main()
