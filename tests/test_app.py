import io
import time
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app as clip_service
import internal_auth


TEST_SECRET = "festival-test-secret-with-at-least-32-bytes"
ALL_ACTIONS = [
    "corpus:write",
    "corpus:release",
    "corpus:read",
    "match:execute",
    "health:read",
    "metrics:read",
]


def make_image_bytes(color=(255, 0, 0), size=(32, 32), image_format="PNG"):
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


class FakeProcessor:
    def __call__(self, images=None, text=None, **kwargs):
        if images is not None:
            batch = images if isinstance(images, list) else [images]
            vectors = []
            for image in batch:
                mean = np.asarray(image, dtype=float).reshape(-1, 3).mean(axis=0)
                vectors.append(mean + 1.0)
            return {"image_features": np.asarray(vectors)}
        if text is not None:
            value = text[0].lower()
            return {"text_features": np.asarray([[255.0, 1.0, 1.0] if "red" in value else [1.0, 1.0, 255.0]])}
        raise AssertionError("missing model input")


class FakeModel:
    def get_image_features(self, **inputs):
        return inputs["image_features"]

    def get_text_features(self, **inputs):
        return inputs["text_features"]


def identity_headers(
    *,
    tenant="tenant-a",
    site="site-1",
    sites=None,
    actions=None,
    service="festival-backend",
    request_id=None,
):
    request_id = request_id or f"req-{time.time_ns()}"
    claims = {
        "iss": internal_auth.JWT_ISSUER,
        "aud": internal_auth.JWT_AUDIENCE,
        "service": service,
        "tenantId": tenant,
        "siteId": site,
        "siteIds": sites or [site],
        "datasetVersion": "festival-v1",
        "demoData": True,
        "actions": actions if actions is not None else ALL_ACTIONS,
        "requestId": request_id,
        "exp": int(time.time()) + 300,
    }
    token = internal_auth.sign_internal_token(claims, TEST_SECRET)
    return {"Authorization": f"Bearer {token}", "X-Request-Id": request_id}


class ClipServiceSecurityTests(unittest.TestCase):
    def setUp(self):
        self.auth_patch = patch.object(internal_auth, "INTERNAL_JWT_SECRET", TEST_SECRET)
        self.auth_patch.start()
        self.model_patch = patch.object(clip_service, "load_model", return_value=(FakeModel(), FakeProcessor()))
        self.model_patch.start()
        self.dimension_patch = patch.object(clip_service, "EXPECTED_EMBEDDING_DIMENSION", 3)
        self.dimension_patch.start()
        clip_service.embedding_dimension = 3
        clip_service.model_warmup_completed = False
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        self.client = TestClient(clip_service.app)

    def tearDown(self):
        self.dimension_patch.stop()
        self.model_patch.stop()
        self.auth_patch.stop()
        clip_service.embedding_dimension = None
        clip_service.preprocessing_version = None
        clip_service.model_warmup_completed = False
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())

    def create_item(self, item_id="item-1", tenant="tenant-a", site="site-1", color=(255, 0, 0)):
        return self.client.post(
            "/items",
            headers=identity_headers(tenant=tenant, site=site),
            data={"id": item_id, "title": "Festival synthetic wallet"},
            files={"file": ("item.png", make_image_bytes(color), "image/png")},
        )

    def release_item(self, item_id="item-1", tenant="tenant-a", site="site-1"):
        return self.client.post(
            f"/items/{item_id}/release",
            headers=identity_headers(tenant=tenant, site=site, actions=["corpus:release"]),
        )

    def test_public_liveness_is_minimal(self):
        response = self.client.get("/health/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"status", "requestId"})

    def test_readiness_reports_safe_checks_and_stays_false_before_warmup(self):
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 503)
        body = response.json()
        checks = body["checks"]
        self.assertEqual(
            set(checks),
            {
                "modelLoaded",
                "embeddingDimensionKnown",
                "embeddingDimensionMatchesExpected",
                "preprocessingVersionKnown",
                "ocrDependencyReady",
                "databaseReady",
                "device",
                "warmupCompleted",
            },
        )
        self.assertFalse(checks["warmupCompleted"])
        self.assertEqual(body["expectedEmbeddingDimension"], clip_service.EXPECTED_EMBEDDING_DIMENSION)

    def test_readiness_passes_once_warmup_dimension_matches_expected(self):
        # OCR's real dependency (tesseract binary + language data) isn't
        # installed in this test environment -- irrelevant to what this
        # test is proving (dimension/preprocessing readiness), so it's
        # patched out here the same way EXPECTED_EMBEDDING_DIMENSION
        # already is above.
        with patch.object(clip_service, "OCR_ENABLED", False), patch.object(
            clip_service, "model", FakeModel()
        ), patch.object(clip_service, "processor", FakeProcessor()):
            clip_service.embedding_dimension = 3
            clip_service.preprocessing_version = "fingerprint-abc"
            clip_service.model_warmup_completed = True
            response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 200)
        checks = response.json()["checks"]
        self.assertTrue(checks["embeddingDimensionMatchesExpected"])
        self.assertTrue(checks["preprocessingVersionKnown"])

    def test_readiness_fails_when_live_embedding_dimension_does_not_match_expected(self):
        # P1-7: a real, silent incompatibility -- e.g. a different model
        # revision loaded than this deployment expects -- must fail
        # readiness even though a dimension was produced ("known").
        clip_service.embedding_dimension = 999
        clip_service.preprocessing_version = "fingerprint-abc"
        clip_service.model_warmup_completed = True
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 503)
        checks = response.json()["checks"]
        self.assertTrue(checks["embeddingDimensionKnown"])
        self.assertFalse(checks["embeddingDimensionMatchesExpected"])

    def test_readiness_fails_when_preprocessing_version_was_never_computed(self):
        clip_service.embedding_dimension = 3
        clip_service.preprocessing_version = None
        clip_service.model_warmup_completed = True
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["checks"]["preprocessingVersionKnown"])

    def test_health_dependency_report_carries_full_live_provenance(self):
        clip_service.embedding_dimension = 3
        clip_service.preprocessing_version = "fingerprint-abc"
        response = self.client.get("/health", headers=identity_headers())
        self.assertEqual(response.status_code, 200)
        model = response.json()["dependencies"]["model"]
        self.assertEqual(model["modelId"], clip_service.MODEL_NAME)
        self.assertEqual(model["modelRevision"], clip_service.MODEL_REVISION)
        self.assertEqual(model["embeddingDimension"], 3)
        self.assertEqual(model["preprocessingVersion"], "fingerprint-abc")
        self.assertEqual(model["scoringVersion"], clip_service.SCORING_VERSION)
        self.assertEqual(model["serviceVersion"], clip_service.SERVICE_VERSION)
        self.assertEqual(model["expectedEmbeddingDimension"], clip_service.EXPECTED_EMBEDDING_DIMENSION)
        self.assertTrue(model["embeddingDimensionMatchesExpected"])

    def test_service_identity_is_mandatory(self):
        response = self.client.post("/match", data={"text": "red wallet"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "missing_identity")

    def test_expired_service_identity_is_rejected(self):
        claims = {
            "iss": internal_auth.JWT_ISSUER,
            "aud": internal_auth.JWT_AUDIENCE,
            "service": "festival-backend",
            "tenantId": "tenant-a",
            "siteId": "site-1",
            "siteIds": ["site-1"],
            "actions": ["match:execute"],
            "requestId": "req-expired",
            "exp": int(time.time()) - 1,
        }
        token = internal_auth.sign_internal_token(claims, TEST_SECRET)
        response = self.client.post(
            "/match",
            headers={"Authorization": f"Bearer {token}", "X-Request-Id": "req-expired"},
            data={"text": "wallet"},
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "identity_expired")

    def test_identity_request_id_must_match_header(self):
        headers = identity_headers(actions=["match:execute"], request_id="req-signed")
        headers["X-Request-Id"] = "req-different"
        response = self.client.post("/match", headers=headers, data={"text": "wallet"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "request_id_mismatch")

    def test_action_permission_is_enforced(self):
        response = self.client.post(
            "/items",
            headers=identity_headers(actions=["match:execute"]),
            json={"id": "forbidden", "title": "No access"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "action_not_permitted")

    def test_corpus_metadata_comes_from_signed_identity_and_model(self):
        response = self.create_item()
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertEqual(item["tenantId"], "tenant-a")
        self.assertEqual(item["siteId"], "site-1")
        self.assertEqual(item["datasetVersion"], "festival-v1")
        self.assertTrue(item["demoData"])
        self.assertEqual(item["createdBy"], "festival-backend")
        self.assertEqual(item["modelId"], clip_service.MODEL_NAME)
        self.assertEqual(item["modelRevision"], clip_service.MODEL_REVISION)
        self.assertEqual(item["embeddingDimension"], 3)
        self.assertNotIn("embedding", item)
        self.assertNotIn("clipEmbedding", item)

    def test_client_cannot_set_system_fields_or_embedding(self):
        response = self.client.post(
            "/items",
            headers=identity_headers(),
            json={
                "id": "hostile",
                "title": "Hostile",
                "released": True,
                "status": "released",
                "clipEmbedding": [1, 0, 0],
                "tenantId": "other",
                "modelRevision": "browser-choice",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "system_fields_forbidden")

    def test_tenant_isolation_applies_to_item_reads(self):
        self.assertEqual(self.create_item().status_code, 200)
        denied = self.client.get("/items/item-1", headers=identity_headers(tenant="tenant-b"))
        self.assertEqual(denied.status_code, 404)
        allowed = self.client.get("/items/item-1", headers=identity_headers(actions=["corpus:read"]))
        self.assertEqual(allowed.status_code, 200)

    def test_site_permissions_filter_corpus(self):
        self.assertEqual(self.create_item(site="site-1").status_code, 200)
        denied = self.client.get(
            "/items",
            headers=identity_headers(site="site-2", sites=["site-2"], actions=["corpus:read"]),
        )
        self.assertEqual(denied.status_code, 200)
        self.assertEqual(denied.json()["items"], [])
        allowed = self.client.get(
            "/items",
            headers=identity_headers(site="site-2", sites=["site-1", "site-2"], actions=["corpus:read"]),
        )
        self.assertEqual(len(allowed.json()["items"]), 1)

    def test_only_released_compatible_candidates_are_matched(self):
        self.create_item("released-red")
        self.release_item("released-red")
        self.create_item("draft-blue", color=(0, 0, 255))
        response = self.client.post(
            "/match",
            headers=identity_headers(actions=["match:execute"]),
            data={"text": "red wallet", "topK": "5"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["governance"]["candidatesEvaluated"], 1)
        self.assertEqual(response.json()["topK"][0]["candidateId"], "released-red")

    def test_browser_cannot_control_match_policy(self):
        response = self.client.post(
            "/match",
            headers=identity_headers(actions=["match:execute"]),
            data={"text": "wallet", "minScore": "-1", "status": "draft", "doOcr": "false"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "system_controls_forbidden")

    def test_encode_endpoints_do_not_return_raw_embeddings(self):
        image_response = self.client.post(
            "/encode-image",
            headers=identity_headers(actions=["match:execute"]),
            files={"file": ("image.png", make_image_bytes(), "image/png")},
        )
        text_response = self.client.post(
            "/encode-text",
            headers=identity_headers(actions=["match:execute"]),
            data={"text": "red wallet"},
        )
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(text_response.status_code, 200)
        self.assertNotIn("embedding", image_response.json())
        self.assertNotIn("embedding", text_response.json())

    def test_metrics_require_operational_permission(self):
        denied = self.client.get("/metrics", headers=identity_headers(actions=["health:read"]))
        self.assertEqual(denied.status_code, 403)
        allowed = self.client.get("/metrics", headers=identity_headers(actions=["metrics:read"]))
        self.assertEqual(allowed.status_code, 200)

    def test_invalid_image_and_mime_are_rejected(self):
        bad_image = self.client.post(
            "/analyze-image",
            headers=identity_headers(actions=["match:execute"]),
            files={"file": ("bad.png", b"not-an-image", "image/png")},
        )
        bad_mime = self.client.post(
            "/analyze-image",
            headers=identity_headers(actions=["match:execute"]),
            files={"file": ("bad.gif", make_image_bytes(), "image/gif")},
        )
        self.assertEqual(bad_image.status_code, 400)
        self.assertEqual(bad_mime.status_code, 415)

    def test_oversized_upload_is_rejected(self):
        with patch.object(clip_service, "MAX_UPLOAD_BYTES", 16):
            response = self.client.post(
                "/analyze-image",
                headers=identity_headers(actions=["match:execute"]),
                files={"file": ("large.png", make_image_bytes(), "image/png")},
            )
        self.assertEqual(response.status_code, 413)

    def test_inference_timeout_does_not_block_async_route(self):
        def slow_embedding(image):
            time.sleep(0.2)
            return [1.0, 0.0, 0.0]

        with patch.object(clip_service, "image_embedding_for", slow_embedding), patch.object(
            clip_service, "INFERENCE_TIMEOUT_MS", 20
        ):
            response = self.client.post(
                "/encode-image",
                headers=identity_headers(actions=["match:execute"]),
                files={"file": ("image.png", make_image_bytes(), "image/png")},
            )
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["error"]["code"], "inference_timeout")

    def test_embedding_validation_rejects_nonfinite_and_dimension_mismatch(self):
        with self.assertRaises(Exception):
            clip_service.validate_stored_embedding([float("nan"), 0, 1])
        with self.assertRaises(Exception):
            clip_service.validate_stored_embedding([1, 0])

    def test_festival_storage_never_falls_back_to_memory(self):
        with patch.object(clip_service, "ENVIRONMENT", "festival"), patch.object(
            clip_service, "STORAGE_MODE", "memory"
        ):
            with self.assertRaises(RuntimeError):
                clip_service.build_repository()
        with patch.object(clip_service, "ENVIRONMENT", "festival"), patch.object(
            clip_service, "STORAGE_MODE", "mongodb"
        ), patch.object(clip_service, "MONGODB_URI", ""):
            with self.assertRaises(RuntimeError):
                clip_service.build_repository()


if __name__ == "__main__":
    unittest.main()
