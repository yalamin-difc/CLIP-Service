"""P13 section 25: integration tests for the /v2 API surface.

Reuses test_app.py's FakeModel/FakeProcessor/identity_headers fixtures --
same convention the rest of this suite already follows (see
tests/test_text_only_corpus.py) -- so clip_v1 runs through its real,
unmodified production code path with only the underlying transformers
model swapped out. siglip2_v1 is exercised through a lightweight stub
registered directly into embedding_engines.registry, since no real model
weights are available in CI/this sandbox; the stub still goes through the
*real* v2_router.py/app.py code paths (storage, isolation, auditing,
A/B comparison), only the tensor math is faked.
"""
import io
import json
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as clip_service
import internal_auth
from embedding_engines import config as engine_config
from embedding_engines import registry as engine_registry
from embedding_engines.base import EngineProvenance, EngineReadiness, EngineUnavailableError
from embedding_engines.siglip2_engine import Siglip2Engine
from test_app import ALL_ACTIONS as LEGACY_ACTIONS
from test_app import FakeModel, FakeProcessor, TEST_SECRET, identity_headers as _legacy_identity_headers, make_image_bytes

ALL_ACTIONS = LEGACY_ACTIONS + ["match:evaluate"]


def identity_headers(**kwargs):
    kwargs.setdefault("actions", ALL_ACTIONS)
    return _legacy_identity_headers(**kwargs)


class StubSiglip2Engine(Siglip2Engine):
    """A siglip2_v1 stand-in with deterministic, dependency-free vectors --
    real object identity (isinstance Siglip2Engine) but no real transformers
    call, so tests can drive the real v2_router.py code paths."""

    STUB_DIMENSION = 4

    def __init__(self, *, fail_text_substring=None):
        super().__init__()
        self._fail_text_substring = fail_text_substring

    # is_enabled() is intentionally NOT overridden: it must keep reading
    # engine_config.SIGLIP2_ENABLED (via the real base implementation) so
    # tests can exercise the disabled-by-configuration path honestly.

    @property
    def expected_dimension(self):
        return self.STUB_DIMENSION

    async def encode_image_async(self, image):
        # Deterministic image vector, distinct per RGB image so different
        # colors don't collide in cosine space.
        pixel = image.getpixel((0, 0)) if image.width and image.height else (1, 1, 1)
        vector = [float(pixel[0]) + 1.0, float(pixel[1]) + 1.0, float(pixel[2]) + 1.0, 1.0]
        return vector

    async def encode_text_async(self, text):
        if self._fail_text_substring and self._fail_text_substring in text:
            raise EngineUnavailableError("stub failure", error_category="inference_failed")
        if "red" in text.lower():
            return [255.0, 1.0, 1.0, 1.0]
        return [1.0, 1.0, 255.0, 1.0]

    def provenance(self):
        return EngineProvenance(
            engine="siglip2_v1",
            model_id="google/siglip2-so400m-patch14-384",
            model_revision="test-rev",
            embedding_dimension=self.STUB_DIMENSION,
            preprocessing_version="stub-fp",
            device="cpu",
            calibration_status="uncalibrated",
            scoring_version="engine-match-v2",
        )

    def readiness(self):
        return EngineReadiness(
            engine="siglip2_v1",
            enabled=True,
            loaded=True,
            ready=True,
            warmup_completed=True,
            device="cpu",
            expected_embedding_dimension=self.STUB_DIMENSION,
            embedding_dimension=self.STUB_DIMENSION,
            last_load_error_category=None,
        )


class V2ApiTestsBase(unittest.TestCase):
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

        self._original_siglip2 = engine_registry._ENGINES["siglip2_v1"]
        self.stub_siglip2 = StubSiglip2Engine()
        engine_registry._ENGINES["siglip2_v1"] = self.stub_siglip2

        self.siglip2_enabled_patch = patch.object(engine_config, "SIGLIP2_ENABLED", True)
        self.siglip2_enabled_patch.start()

        self.client = TestClient(clip_service.app)

    def tearDown(self):
        engine_registry._ENGINES["siglip2_v1"] = self._original_siglip2
        self.siglip2_enabled_patch.stop()
        self.dimension_patch.stop()
        self.model_patch.stop()
        self.auth_patch.stop()
        clip_service.embedding_dimension = None
        clip_service.preprocessing_version = None
        clip_service.model_warmup_completed = False
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())

    def create_and_release_item(self, item_id="item-1", color=(255, 0, 0), tenant="tenant-a", site="site-1", title="Festival wallet"):
        response = self.client.post(
            "/items",
            headers=identity_headers(tenant=tenant, site=site),
            data={"id": item_id, "title": title},
            files={"file": ("item.png", make_image_bytes(color), "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        release = self.client.post(
            f"/items/{item_id}/release",
            headers=identity_headers(tenant=tenant, site=site, actions=["corpus:release"]),
        )
        self.assertEqual(release.status_code, 200, release.text)
        return response.json()["item"]

    def write_siglip2_embedding(self, item_id, color=(10, 20, 30), tenant="tenant-a", site="site-1"):
        response = self.client.post(
            f"/v2/items/{item_id}/embeddings/siglip2_v1",
            headers=identity_headers(tenant=tenant, site=site),
            files={"file": ("item.png", make_image_bytes(color), "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["item"]


class LegacyCompatibilityTests(V2ApiTestsBase):
    """Section 25: legacy endpoints remain fully compatible after mounting
    the V2 router -- default to CLIP, 512-D constant unpatched, unaffected
    by SigLIP2 configuration."""

    def test_legacy_match_still_works_and_ignores_engine_field(self):
        self.create_and_release_item()
        response = self.client.post(
            "/match",
            headers=identity_headers(),
            data={"text": "red wallet", "engine": "siglip2_v1"},  # must be ignored entirely
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["governance"]["engine"], "clip")
        self.assertEqual(body["governance"]["modelId"], clip_service.MODEL_NAME)

    def test_legacy_items_and_health_endpoints_unaffected(self):
        self.create_and_release_item()
        items = self.client.get("/items", headers=identity_headers())
        self.assertEqual(items.status_code, 200)
        health = self.client.get("/health", headers=identity_headers())
        self.assertEqual(health.status_code, 200)
        self.assertIn("engines", health.json())  # additive field only

    def test_embeddings_field_cannot_be_injected_via_legacy_items_api(self):
        response = self.client.post(
            "/items",
            headers=identity_headers(),
            data={"id": "item-x", "title": "hack", "embeddings": json.dumps({"siglip2_v1": {"vector": [1, 2, 3, 4]}})},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "system_fields_forbidden")


class DualEmbeddingStorageTests(V2ApiTestsBase):
    def test_item_can_hold_both_engine_embeddings(self):
        # Creating via the legacy /items endpoint populates the legacy
        # embedding/clipEmbedding fields but does not, by itself, populate
        # embeddings.clip_v1 -- that block is only written the first time
        # an item is explicitly (re-)embedded through the V2 API. Writing
        # both engines through V2 is what proves a single item can hold
        # both representations at once (section 8).
        item = self.create_and_release_item()
        legacy_embedding_dimension = item["embeddingDimension"]
        self.assertEqual(legacy_embedding_dimension, 3)

        clip_reembed = self.client.post(
            f"/v2/items/{item['id']}/embeddings/clip_v1",
            headers=identity_headers(),
            files={"file": ("item.png", make_image_bytes(), "image/png")},
        )
        self.assertEqual(clip_reembed.status_code, 200, clip_reembed.text)

        saved = self.write_siglip2_embedding(item["id"])
        self.assertIn("clip_v1", saved["embeddings"] or {})
        self.assertIn("siglip2_v1", saved["embeddings"] or {})
        self.assertEqual(saved["embeddings"]["siglip2_v1"]["embeddingDimension"], StubSiglip2Engine.STUB_DIMENSION)
        # Legacy CLIP fields are untouched by writing a SigLIP2 embedding.
        self.assertEqual(saved["embeddingDimension"], 3)

    def test_v2_item_response_never_exposes_raw_vector(self):
        item = self.create_and_release_item()
        saved = self.write_siglip2_embedding(item["id"])
        self.assertNotIn("vector", json.dumps(saved["embeddings"]))

    def test_clip_v1_reembed_via_v2_refreshes_legacy_fields(self):
        item = self.create_and_release_item(color=(255, 0, 0))
        response = self.client.post(
            f"/v2/items/{item['id']}/embeddings/clip_v1",
            headers=identity_headers(),
            files={"file": ("item.png", make_image_bytes((0, 255, 0)), "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        saved = response.json()["item"]
        self.assertIn("clip_v1", saved["embeddings"])
        # Legacy top-level embeddingDimension still reflects CLIP's space.
        self.assertEqual(saved["embeddingDimension"], 3)


class CrossModelRejectionTests(V2ApiTestsBase):
    def test_v2_match_with_siglip2_excludes_clip_only_candidates(self):
        clip_only_item = self.create_and_release_item(item_id="clip-only", color=(0, 0, 255))
        siglip_item = self.create_and_release_item(item_id="both", color=(255, 0, 0))
        self.write_siglip2_embedding(siglip_item["id"])

        response = self.client.post("/v2/match", headers=identity_headers(), data={"text": "red", "engine": "siglip2_v1"})
        self.assertEqual(response.status_code, 200, response.text)
        candidate_ids = [c["candidateId"] for c in response.json()["topK"]]
        self.assertIn("both", candidate_ids)
        self.assertNotIn("clip-only", candidate_ids)

    def test_candidate_vector_helper_rejects_wrong_dimension(self):
        import v2_router

        candidate = {"embeddings": {"siglip2_v1": {"vector": [1.0, 2.0, 3.0]}}}  # 3-D, engine expects 4-D
        engine = engine_registry.get_engine("siglip2_v1")
        self.assertIsNone(v2_router._candidate_vector(candidate, engine))


class TenantSiteIsolationTests(V2ApiTestsBase):
    def test_tenant_isolation_in_v2_match(self):
        self.create_and_release_item(item_id="item-a", tenant="tenant-a", site="site-1", color=(255, 0, 0))
        self.create_and_release_item(item_id="item-b", tenant="tenant-b", site="site-1", color=(255, 0, 0))
        response = self.client.post(
            "/v2/match", headers=identity_headers(tenant="tenant-a", site="site-1"), data={"text": "red"}
        )
        candidate_ids = [c["candidateId"] for c in response.json()["topK"]]
        self.assertIn("item-a", candidate_ids)
        self.assertNotIn("item-b", candidate_ids)

    def test_site_isolation_in_v2_match(self):
        self.create_and_release_item(item_id="item-a", tenant="tenant-a", site="site-1", color=(255, 0, 0))
        self.create_and_release_item(item_id="item-b", tenant="tenant-a", site="site-2", color=(255, 0, 0))
        response = self.client.post(
            "/v2/match",
            headers=identity_headers(tenant="tenant-a", site="site-1", sites=["site-1"]),
            data={"text": "red"},
        )
        candidate_ids = [c["candidateId"] for c in response.json()["topK"]]
        self.assertIn("item-a", candidate_ids)
        self.assertNotIn("item-b", candidate_ids)


class ArabicTextTests(V2ApiTestsBase):
    ENGLISH = "black suitcase with a red ribbon on the handle"
    ARABIC = "حقيبة سفر سوداء عليها شريط أحمر على المقبض"
    MIXED = "Black Samsonite حقيبة with red ribbon"

    def test_arabic_text_encodes_without_corruption(self):
        for text in (self.ENGLISH, self.ARABIC, self.MIXED):
            response = self.client.post("/v2/embeddings/text", headers=identity_headers(), data={"text": text, "engine": "clip_v1"})
            self.assertEqual(response.status_code, 200, response.text)

    def test_arabic_item_round_trips_through_storage(self):
        response = self.client.post(
            "/items",
            headers=identity_headers(),
            data={"id": "item-arabic", "title": self.ARABIC, "description": self.MIXED},
            files={"file": ("item.png", make_image_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        stored = response.json()["item"]
        self.assertEqual(stored["title"], self.ARABIC)
        self.assertEqual(stored["description"], self.MIXED)
        fetched = self.client.get(f"/items/{stored['id']}", headers=identity_headers())
        self.assertEqual(fetched.json()["item"]["title"], self.ARABIC)

    def test_arabic_text_matches_via_v2(self):
        self.create_and_release_item(title=self.ARABIC)
        response = self.client.post("/v2/match", headers=identity_headers(), data={"text": self.ARABIC})
        self.assertEqual(response.status_code, 200, response.text)


class AbMatchTests(V2ApiTestsBase):
    def setUp(self):
        super().setUp()
        self.ab_enabled_patch = patch.object(engine_config, "AB_TEST_ENABLED", True)
        self.ab_enabled_patch.start()

    def tearDown(self):
        self.ab_enabled_patch.stop()
        super().tearDown()

    def test_ab_disabled_by_default_returns_503(self):
        self.ab_enabled_patch.stop()
        try:
            with patch.object(engine_config, "AB_TEST_ENABLED", False):
                response = self.client.post("/v2/ab/match", headers=identity_headers(), data={"text": "red wallet"})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["error"]["code"], "ab_testing_disabled")
        finally:
            self.ab_enabled_patch.start()

    def test_ab_match_runs_both_engines_independently(self):
        item = self.create_and_release_item(color=(255, 0, 0))
        self.write_siglip2_embedding(item["id"], color=(255, 0, 0))
        response = self.client.post("/v2/ab/match", headers=identity_headers(), data={"text": "red wallet"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["clip"]["status"], "success")
        self.assertEqual(body["siglip2"]["status"], "success")
        self.assertIn("comparison", body)
        self.assertIn("topKOverlap", body["comparison"])
        # Never declares a winner.
        self.assertNotIn("winner", body)
        self.assertNotIn("winner", body["comparison"])

    def test_ab_match_isolates_siglip2_failure_from_clip(self):
        engine_registry._ENGINES["siglip2_v1"] = StubSiglip2Engine(fail_text_substring="fail")
        try:
            self.create_and_release_item(color=(255, 0, 0))
            response = self.client.post("/v2/ab/match", headers=identity_headers(), data={"text": "please fail now"})
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["clip"]["status"], "success")
            self.assertEqual(body["siglip2"]["status"], "unavailable")
            self.assertIn("errorCode", body["siglip2"])
            # No Python traceback leaked.
            self.assertNotIn("Traceback", json.dumps(body))
        finally:
            engine_registry._ENGINES["siglip2_v1"] = self.stub_siglip2

    def test_ab_match_never_exposes_raw_vectors(self):
        item = self.create_and_release_item(color=(255, 0, 0))
        self.write_siglip2_embedding(item["id"], color=(255, 0, 0))
        response = self.client.post("/v2/ab/match", headers=identity_headers(), data={"text": "red wallet"})
        self.assertNotIn('"vector"', response.text)

    def test_siglip2_disabled_reports_unavailable_not_500(self):
        with patch.object(engine_config, "SIGLIP2_ENABLED", False):
            self.create_and_release_item(color=(255, 0, 0))
            response = self.client.post("/v2/ab/match", headers=identity_headers(), data={"text": "red wallet"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["siglip2"]["status"], "unavailable")
            self.assertEqual(response.json()["siglip2"]["errorCode"], "engine_disabled")
            self.assertEqual(response.json()["clip"]["status"], "success")


class EvaluationGatingTests(V2ApiTestsBase):
    def test_expected_candidate_id_requires_evaluate_permission(self):
        self.create_and_release_item()
        response = self.client.post(
            "/v2/match",
            headers=identity_headers(actions=[a for a in ALL_ACTIONS if a != "match:evaluate"]),
            data={"text": "red wallet", "expectedCandidateId": "item-1"},
        )
        self.assertEqual(response.status_code, 403)

    def test_evaluation_metrics_computed_when_permitted(self):
        self.create_and_release_item()
        response = self.client.post(
            "/v2/match", headers=identity_headers(), data={"text": "red wallet", "expectedCandidateId": "item-1"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        evaluation = response.json()["evaluation"]
        self.assertEqual(evaluation["expectedCandidateId"], "item-1")
        self.assertEqual(evaluation["expectedRank"], 1)
        self.assertTrue(evaluation["recallAt1"])


class ProvenanceAndAuditTests(V2ApiTestsBase):
    def test_v2_models_reports_provenance_without_secrets(self):
        response = self.client.get("/v2/models", headers=identity_headers())
        self.assertEqual(response.status_code, 200)
        models = {m["engine"]: m for m in response.json()["models"]}
        self.assertEqual(models["clip_v1"]["calibrationStatus"], "calibrated")
        self.assertEqual(models["siglip2_v1"]["calibrationStatus"], "uncalibrated")
        self.assertNotIn("vector", json.dumps(models))

    def test_v2_match_governance_identifies_engine(self):
        self.create_and_release_item()
        response = self.client.post("/v2/match", headers=identity_headers(), data={"text": "red wallet"})
        self.assertEqual(response.json()["governance"]["engine"], "clip_v1")

    def test_audit_log_records_engine_for_v2_match(self):
        self.create_and_release_item()
        self.client.post("/v2/match", headers=identity_headers(), data={"text": "red wallet"})
        logs = self.client.get("/audit-logs", headers=identity_headers())
        entries = [entry for entry in logs.json()["logs"] if entry["eventType"] == "match_execution_v2"]
        self.assertTrue(entries)
        self.assertEqual(entries[0]["payload"]["engine"], "clip_v1")

    def test_stale_revision_triggers_only_matching_engine_reembed(self):
        from scripts.reembed_model import needs_reembedding

        item = {
            "embeddings": {
                "clip_v1": {"vector": [1, 2, 3], "modelId": "openai/clip-vit-base-patch32", "modelRevision": "old-rev", "embeddingDimension": 3},
                "siglip2_v1": {"vector": [1, 2, 3, 4], "modelId": "google/siglip2-so400m-patch14-384", "modelRevision": "current", "embeddingDimension": 4},
            }
        }
        self.assertTrue(needs_reembedding(item, "clip_v1", "openai/clip-vit-base-patch32", "new-rev", 3))
        self.assertFalse(needs_reembedding(item, "siglip2_v1", "google/siglip2-so400m-patch14-384", "current", 4))


if __name__ == "__main__":
    unittest.main()
