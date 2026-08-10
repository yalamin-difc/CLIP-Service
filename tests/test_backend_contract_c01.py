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
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        self.client = TestClient(clip_service.app)

    def tearDown(self):
        self.dimension_patch.stop()
        self.model_patch.stop()
        self.auth_patch.stop()
        clip_service.embedding_dimension = None
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


if __name__ == "__main__":
    unittest.main()
