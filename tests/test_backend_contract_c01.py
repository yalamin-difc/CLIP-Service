import base64
import io
import json
import time
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app as clip_service
import internal_auth


TEST_SECRET = "c01-backend-contract-secret-at-least-32-bytes"


def image_bytes(color):
    image = Image.new("RGB", (32, 32), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ContractProcessor:
    def __call__(self, images=None, text=None, **kwargs):
        if images is not None:
            batch = images if isinstance(images, list) else [images]
            vectors = []
            for image in batch:
                mean = np.asarray(image, dtype=float).reshape(-1, 3).mean(axis=0)
                vectors.append(mean + 1.0)
            return {"image_features": np.asarray(vectors, dtype=float)}
        if text is not None:
            return {"text_features": np.asarray([[255.0, 1.0, 1.0]], dtype=float)}
        raise AssertionError("CLIP processor input is required")


class ContractModel:
    def get_image_features(self, **inputs):
        return inputs["image_features"]

    def get_text_features(self, **inputs):
        return inputs["text_features"]


def service_headers(
    *,
    action,
    request_id,
    tenant_id="tenant-festival-a",
    site_id="site-difc",
    site_ids=None,
    service="backend-api",
    expires_at=None,
    issuer=None,
    audience=None,
    secret=TEST_SECRET,
    omit=(),
):
    claims = {
        "iss": issuer if issuer is not None else internal_auth.JWT_ISSUER,
        "aud": audience if audience is not None else internal_auth.JWT_AUDIENCE,
        "service": service,
        "tenantId": tenant_id,
        "siteId": site_id,
        "siteIds": site_ids if site_ids is not None else [site_id],
        "actions": [action],
        "exp": expires_at if expires_at is not None else int(time.time()) + 300,
        "requestId": request_id,
        "datasetVersion": "festival-c01-v1",
        "demoData": True,
    }
    for claim in omit:
        claims.pop(claim, None)
    token = internal_auth.sign_internal_token(claims, secret)
    return {"Authorization": f"Bearer {token}", "X-Request-Id": request_id}


def assert_no_raw_embeddings(test_case, value):
    if isinstance(value, dict):
        test_case.assertNotIn("embedding", value)
        test_case.assertNotIn("clipEmbedding", value)
        for nested in value.values():
            assert_no_raw_embeddings(test_case, nested)
    elif isinstance(value, list):
        for nested in value:
            assert_no_raw_embeddings(test_case, nested)


class BackendC01InteroperabilityTests(unittest.TestCase):
    def setUp(self):
        self.auth_patch = patch.object(internal_auth, "INTERNAL_JWT_SECRET", TEST_SECRET)
        self.auth_patch.start()
        self.model_patch = patch.object(
            clip_service,
            "load_model",
            return_value=(ContractModel(), ContractProcessor()),
        )
        self.model_patch.start()
        self.dimension_patch = patch.object(clip_service, "EXPECTED_EMBEDDING_DIMENSION", 3)
        self.dimension_patch.start()
        clip_service.embedding_dimension = 3
        clip_service.preprocessing_version = "fingerprint-c01-test"
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        self.client = TestClient(clip_service.app)

    def tearDown(self):
        self.dimension_patch.stop()
        self.model_patch.stop()
        self.auth_patch.stop()
        clip_service.embedding_dimension = None
        clip_service.preprocessing_version = None
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())

    def create_item(self, item_id, tenant_id, site_id, color):
        request_id = f"create-{tenant_id}-{site_id}-{item_id}"
        return self.client.post(
            "/items",
            headers=service_headers(
                action="corpus:write",
                request_id=request_id,
                tenant_id=tenant_id,
                site_id=site_id,
            ),
            data={"id": item_id, "title": f"Synthetic {item_id}", "description": "Festival synthetic item"},
            files={"file": (f"{item_id}.png", image_bytes(color), "image/png")},
        )

    def release_item(self, item_id, tenant_id, site_id):
        request_id = f"release-{tenant_id}-{site_id}-{item_id}"
        return self.client.post(
            f"/items/{item_id}/release",
            headers=service_headers(
                action="corpus:release",
                request_id=request_id,
                tenant_id=tenant_id,
                site_id=site_id,
            ),
        )

    def test_backend_facing_create_release_analyze_match_sequence(self):
        target = self.create_item("target-wallet", "tenant-festival-a", "site-difc", (255, 0, 0))
        foreign_tenant = self.create_item("foreign-wallet", "tenant-festival-b", "site-difc", (255, 0, 0))
        foreign_site = self.create_item("other-site-wallet", "tenant-festival-a", "site-expo", (255, 0, 0))
        for response in (target, foreign_tenant, foreign_site):
            self.assertEqual(response.status_code, 200)
            assert_no_raw_embeddings(self, response.json())

        target_item = target.json()["item"]
        self.assertEqual(target_item["tenantId"], "tenant-festival-a")
        self.assertEqual(target_item["siteId"], "site-difc")
        self.assertEqual(target_item["createdBy"], "backend-api")
        self.assertEqual(target_item["datasetVersion"], "festival-c01-v1")
        self.assertFalse(target_item["released"])

        for item_id, tenant_id, site_id in (
            ("target-wallet", "tenant-festival-a", "site-difc"),
            ("foreign-wallet", "tenant-festival-b", "site-difc"),
            ("other-site-wallet", "tenant-festival-a", "site-expo"),
        ):
            released = self.release_item(item_id, tenant_id, site_id)
            self.assertEqual(released.status_code, 200)
            self.assertTrue(released.json()["item"]["released"])
            assert_no_raw_embeddings(self, released.json())

        analyze_request_id = "backend-analyze-c01"
        analyzed = self.client.post(
            "/analyze-image",
            headers=service_headers(action="match:execute", request_id=analyze_request_id),
            files={"file": ("query.png", image_bytes((255, 0, 0)), "image/png")},
        )
        self.assertEqual(analyzed.status_code, 200)
        self.assertEqual(analyzed.headers["X-Request-Id"], analyze_request_id)
        self.assertEqual(analyzed.json()["requestId"], analyze_request_id)
        self.assertEqual(analyzed.json()["model"]["modelId"], clip_service.MODEL_NAME)
        self.assertEqual(analyzed.json()["model"]["modelRevision"], clip_service.MODEL_REVISION)
        assert_no_raw_embeddings(self, analyzed.json())

        match_request_id = "backend-match-c01"
        matched = self.client.post(
            "/match",
            headers=service_headers(action="match:execute", request_id=match_request_id),
            data={
                "text": "red synthetic wallet",
                "topK": "5",
                "metadata": json.dumps({"source": "backend-c01-contract"}),
            },
            files={"file": ("query.png", image_bytes((255, 0, 0)), "image/png")},
        )
        self.assertEqual(matched.status_code, 200)
        payload = matched.json()
        self.assertEqual(matched.headers["X-Request-Id"], match_request_id)
        self.assertEqual(payload["requestId"], match_request_id)
        self.assertEqual(payload["governance"]["modelId"], clip_service.MODEL_NAME)
        # P1-7: the response's own governance block now carries the full
        # live provenance -- not just modelId/scoringVersion -- so Backend
        # can persist exactly what produced THIS response as authoritative
        # MatchDecision metadata without a second, potentially-racy call
        # to /health.
        self.assertEqual(payload["governance"]["modelRevision"], clip_service.MODEL_REVISION)
        self.assertEqual(payload["governance"]["embeddingDimension"], 3)
        self.assertEqual(payload["governance"]["preprocessingVersion"], "fingerprint-c01-test")
        self.assertEqual(payload["governance"]["scoringVersion"], clip_service.SCORING_VERSION)
        self.assertEqual(payload["governance"]["thresholds"]["minScore"], clip_service.CONF_MIN_SCORE)
        self.assertTrue(payload["governance"]["releasedOnly"])
        self.assertEqual(payload["governance"]["candidatesEvaluated"], 1)
        self.assertFalse(payload["decision"]["noMatch"])
        self.assertEqual([candidate["candidateId"] for candidate in payload["topK"]], ["target-wallet"])
        candidate = payload["topK"][0]
        self.assertEqual(candidate["item"]["tenantId"], "tenant-festival-a")
        self.assertEqual(candidate["item"]["siteId"], "site-difc")
        self.assertEqual(candidate["item"]["modelId"], clip_service.MODEL_NAME)
        self.assertEqual(candidate["item"]["modelRevision"], clip_service.MODEL_REVISION)
        self.assertEqual(candidate["explanation"]["model"]["scoringVersion"], clip_service.SCORING_VERSION)
        assert_no_raw_embeddings(self, payload)

    def test_signed_identity_claim_contract_and_authorization(self):
        valid = service_headers(action="match:execute", request_id="valid-hs256")
        encoded_header = valid["Authorization"].split(" ", 1)[1].split(".", 1)[0]
        jwt_header = json.loads(base64.urlsafe_b64decode(encoded_header + "=" * (-len(encoded_header) % 4)))
        self.assertEqual(jwt_header, {"alg": "HS256", "typ": "JWT"})

        cases = (
            ("invalid signature", service_headers(action="match:execute", request_id="bad-sig", secret="x" * 32), 401),
            (
                "invalid issuer",
                service_headers(action="match:execute", request_id="bad-iss", issuer="untrusted-backend"),
                401,
            ),
            (
                "invalid audience",
                service_headers(action="match:execute", request_id="bad-aud", audience="other-service"),
                401,
            ),
            ("missing service", service_headers(action="match:execute", request_id="no-svc", omit=("service",)), 401),
            ("missing tenant", service_headers(action="match:execute", request_id="no-tenant", omit=("tenantId",)), 401),
            ("missing site", service_headers(action="match:execute", request_id="no-site", omit=("siteId",)), 401),
            ("missing sites", service_headers(action="match:execute", request_id="no-sites", omit=("siteIds",)), 401),
            (
                "expired",
                service_headers(action="match:execute", request_id="expired", expires_at=int(time.time()) - 1),
                401,
            ),
            (
                "site not permitted",
                service_headers(
                    action="match:execute",
                    request_id="bad-site",
                    site_id="site-difc",
                    site_ids=["site-expo"],
                ),
                403,
            ),
            ("action denied", service_headers(action="corpus:read", request_id="bad-action"), 403),
        )
        for name, headers, expected_status in cases:
            with self.subTest(name=name):
                response = self.client.post("/match", headers=headers, data={"text": "wallet"})
                self.assertEqual(response.status_code, expected_status)

    def test_protected_write_cross_scope_and_request_binding(self):
        unauthorized = self.client.post(
            "/items",
            headers=service_headers(action="match:execute", request_id="write-denied"),
            json={"id": "denied-item", "title": "Denied"},
        )
        self.assertEqual(unauthorized.status_code, 403)
        self.assertEqual(unauthorized.json()["error"]["code"], "action_not_permitted")

        created = self.create_item("scope-item", "tenant-festival-a", "site-difc", (255, 0, 0))
        self.assertEqual(created.status_code, 200)

        cross_tenant = self.client.get(
            "/items/scope-item",
            headers=service_headers(
                action="corpus:read",
                request_id="cross-tenant",
                tenant_id="tenant-festival-b",
                site_id="site-difc",
            ),
        )
        self.assertEqual(cross_tenant.status_code, 404)

        cross_site = self.client.get(
            "/items/scope-item",
            headers=service_headers(
                action="corpus:read",
                request_id="cross-site",
                tenant_id="tenant-festival-a",
                site_id="site-expo",
                site_ids=["site-expo"],
            ),
        )
        self.assertEqual(cross_site.status_code, 404)

        mismatch_headers = service_headers(action="match:execute", request_id="signed-request-id")
        mismatch_headers["X-Request-Id"] = "different-request-id"
        mismatch = self.client.post("/match", headers=mismatch_headers, data={"text": "wallet"})
        self.assertEqual(mismatch.status_code, 401)
        self.assertEqual(mismatch.json()["error"]["code"], "request_id_mismatch")

        missing_headers = service_headers(action="match:execute", request_id="missing-request-id")
        del missing_headers["X-Request-Id"]
        missing = self.client.post("/match", headers=missing_headers, data={"text": "wallet"})
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.json()["error"]["code"], "missing_request_id")

        blank_headers = service_headers(action="match:execute", request_id="blank-request-id")
        blank_headers["X-Request-Id"] = "   "
        blank = self.client.post("/match", headers=blank_headers, data={"text": "wallet"})
        self.assertEqual(blank.status_code, 401)
        self.assertEqual(blank.json()["error"]["code"], "missing_request_id")


class MatchRetrievalContractTests(unittest.TestCase):
    """
    C-01 retrieval-contract tests: CLIP's own no-match decision is advisory
    only and must never suppress ranked, authorized retrieval candidates.
    Backend remains authoritative for the final match decision.
    """

    def setUp(self):
        self.auth_patch = patch.object(internal_auth, "INTERNAL_JWT_SECRET", TEST_SECRET)
        self.auth_patch.start()
        self.model_patch = patch.object(
            clip_service,
            "load_model",
            return_value=(ContractModel(), ContractProcessor()),
        )
        self.model_patch.start()
        self.dimension_patch = patch.object(clip_service, "EXPECTED_EMBEDDING_DIMENSION", 3)
        self.dimension_patch.start()
        clip_service.embedding_dimension = 3
        clip_service.preprocessing_version = "fingerprint-c01-test"
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        self.client = TestClient(clip_service.app)

    def tearDown(self):
        self.dimension_patch.stop()
        self.model_patch.stop()
        self.auth_patch.stop()
        clip_service.embedding_dimension = None
        clip_service.preprocessing_version = None
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())

    def create_item(self, item_id, tenant_id, site_id, color):
        request_id = f"create-{tenant_id}-{site_id}-{item_id}"
        return self.client.post(
            "/items",
            headers=service_headers(
                action="corpus:write",
                request_id=request_id,
                tenant_id=tenant_id,
                site_id=site_id,
            ),
            data={"id": item_id, "title": f"Synthetic {item_id}", "description": "Festival synthetic item"},
            files={"file": (f"{item_id}.png", image_bytes(color), "image/png")},
        )

    def release_item(self, item_id, tenant_id, site_id):
        request_id = f"release-{tenant_id}-{site_id}-{item_id}"
        return self.client.post(
            f"/items/{item_id}/release",
            headers=service_headers(
                action="corpus:release",
                request_id=request_id,
                tenant_id=tenant_id,
                site_id=site_id,
            ),
        )

    def test_a_low_score_no_match_still_returns_ranked_candidates(self):
        # Blue vs red produce near-orthogonal ContractProcessor vectors
        # (cosine well below CONF_MIN_SCORE), so CLIP's own gate says
        # "no match" while a candidate is still available for Backend to
        # evaluate with its own multimodal fusion.
        self.create_item("low-score-item", "tenant-festival-a", "site-difc", (0, 0, 255))
        self.release_item("low-score-item", "tenant-festival-a", "site-difc")

        response = self.client.post(
            "/match",
            headers=service_headers(action="match:execute", request_id="low-score-match"),
            files={"file": ("query.png", image_bytes((255, 0, 0)), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["decision"]["noMatch"])
        self.assertEqual(payload["governance"]["decision"]["noMatch"], True)
        self.assertEqual(
            [candidate["candidateId"] for candidate in payload["topK"]], ["low-score-item"]
        )
        # A returned candidate is retrieval evidence, not an accepted match:
        # nothing in the payload should assert acceptance on CLIP's behalf.
        self.assertNotIn("accepted", payload["topK"][0])
        self.assertNotIn("accepted", payload)
        assert_no_raw_embeddings(self, payload)

    def test_b_client_supplied_thresholds_still_rejected(self):
        for field, value in (("minScore", "0.9"), ("minMargin", "0.5"), ("temperature", "1.0")):
            response = self.client.post(
                "/match",
                headers=service_headers(action="match:execute", request_id=f"forbidden-{field}"),
                data={"text": "red wallet", field: value},
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["error"]["code"], "system_controls_forbidden")

    def test_c_image_only_query_returns_image_cosine(self):
        self.create_item("image-only-item", "tenant-festival-a", "site-difc", (255, 0, 0))
        self.release_item("image-only-item", "tenant-festival-a", "site-difc")

        response = self.client.post(
            "/match",
            headers=service_headers(action="match:execute", request_id="image-only-match"),
            files={"file": ("query.png", image_bytes((255, 0, 0)), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        similarity = response.json()["topK"][0]["explanation"]["similarity"]
        self.assertIsNotNone(similarity["imageCosine"])
        self.assertIsNone(similarity.get("textCosine"))

    def test_d_text_only_query_returns_text_cosine(self):
        self.create_item("text-only-item", "tenant-festival-a", "site-difc", (255, 0, 0))
        self.release_item("text-only-item", "tenant-festival-a", "site-difc")

        response = self.client.post(
            "/match",
            headers=service_headers(action="match:execute", request_id="text-only-match"),
            data={"text": "red synthetic wallet"},
        )
        self.assertEqual(response.status_code, 200)
        similarity = response.json()["topK"][0]["explanation"]["similarity"]
        self.assertIsNotNone(similarity["textCosine"])
        self.assertIsNone(similarity.get("imageCosine"))

    def test_e_image_and_text_query_returns_all_three_cosines(self):
        self.create_item("combined-item", "tenant-festival-a", "site-difc", (255, 0, 0))
        self.release_item("combined-item", "tenant-festival-a", "site-difc")

        response = self.client.post(
            "/match",
            headers=service_headers(action="match:execute", request_id="combined-match"),
            data={"text": "red synthetic wallet"},
            files={"file": ("query.png", image_bytes((255, 0, 0)), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        similarity = response.json()["topK"][0]["explanation"]["similarity"]
        self.assertIsNotNone(similarity["imageCosine"])
        self.assertIsNotNone(similarity["textCosine"])
        self.assertIsNotNone(similarity["combinedCosine"])
        self.assertEqual(similarity["combinedCosine"], similarity["cosine"])

    def test_f_no_response_ever_contains_raw_vectors(self):
        self.create_item("vector-safety-item", "tenant-festival-a", "site-difc", (0, 0, 255))
        self.release_item("vector-safety-item", "tenant-festival-a", "site-difc")

        low_score = self.client.post(
            "/match",
            headers=service_headers(action="match:execute", request_id="vector-safety-low"),
            files={"file": ("query.png", image_bytes((255, 0, 0)), "image/png")},
        )
        matched = self.client.post(
            "/match",
            headers=service_headers(action="match:execute", request_id="vector-safety-match"),
            files={"file": ("query.png", image_bytes((0, 0, 255)), "image/png")},
        )
        for response in (low_score, matched):
            self.assertEqual(response.status_code, 200)
            assert_no_raw_embeddings(self, response.json())

    def test_g_tenant_isolation_enforced_in_retrieval(self):
        self.create_item("tenant-a-item", "tenant-festival-a", "site-difc", (255, 0, 0))
        self.release_item("tenant-a-item", "tenant-festival-a", "site-difc")
        self.create_item("tenant-b-item", "tenant-festival-b", "site-difc", (255, 0, 0))
        self.release_item("tenant-b-item", "tenant-festival-b", "site-difc")

        response = self.client.post(
            "/match",
            headers=service_headers(
                action="match:execute",
                request_id="tenant-isolation-match",
                tenant_id="tenant-festival-a",
                site_id="site-difc",
            ),
            files={"file": ("query.png", image_bytes((255, 0, 0)), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        candidate_ids = [candidate["candidateId"] for candidate in response.json()["topK"]]
        self.assertIn("tenant-a-item", candidate_ids)
        self.assertNotIn("tenant-b-item", candidate_ids)

    def test_h_site_isolation_enforced_in_retrieval(self):
        self.create_item("site-difc-item", "tenant-festival-a", "site-difc", (255, 0, 0))
        self.release_item("site-difc-item", "tenant-festival-a", "site-difc")
        self.create_item("site-expo-item", "tenant-festival-a", "site-expo", (255, 0, 0))
        self.release_item("site-expo-item", "tenant-festival-a", "site-expo")

        response = self.client.post(
            "/match",
            headers=service_headers(
                action="match:execute",
                request_id="site-isolation-match",
                tenant_id="tenant-festival-a",
                site_id="site-difc",
                site_ids=["site-difc"],
            ),
            files={"file": ("query.png", image_bytes((255, 0, 0)), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        candidate_ids = [candidate["candidateId"] for candidate in response.json()["topK"]]
        self.assertIn("site-difc-item", candidate_ids)
        self.assertNotIn("site-expo-item", candidate_ids)

    def test_i_unreleased_candidates_excluded_from_retrieval(self):
        self.create_item("unreleased-item", "tenant-festival-a", "site-difc", (255, 0, 0))
        # Deliberately not released.

        response = self.client.post(
            "/match",
            headers=service_headers(action="match:execute", request_id="unreleased-match"),
            files={"file": ("query.png", image_bytes((255, 0, 0)), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        candidate_ids = [candidate["candidateId"] for candidate in response.json()["topK"]]
        self.assertNotIn("unreleased-item", candidate_ids)

    def test_j_wrong_action_authorization_still_rejected(self):
        response = self.client.post(
            "/match",
            headers=service_headers(action="corpus:write", request_id="wrong-action-match"),
            data={"text": "red wallet"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "action_not_permitted")


if __name__ == "__main__":
    unittest.main()
