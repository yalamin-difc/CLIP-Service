"""
F-02 focused tests: internal_auth.require_identity() used to silently
default a missing demoData claim to True and a missing datasetVersion
claim to the hardcoded literal "festival-2026" -- a Backend signing bug
that omitted these claims was invisible, and every item/query it touched
was silently mislabeled as festival demo data. That default is removed
entirely: demoData and datasetVersion are now required, explicitly-typed
signed claims, verified independently of request JSON, headers other than
the identity token, and query parameters -- missing or malformed claims
are rejected outright, contradictory combinations are rejected, and the
already-covered security controls (signature, issuer, audience, service,
expiry, tenant, site membership, action scope, request-ID binding, and
public health) are exercised alongside the new checks to prove none of
them were weakened or reordered.
"""
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as clip_service
import internal_auth
from test_app import FakeModel, FakeProcessor, TEST_SECRET, identity_headers, make_image_bytes


class DemoGovernanceClaimsTests(unittest.TestCase):
    def setUp(self):
        self.auth_patch = patch.object(internal_auth, "INTERNAL_JWT_SECRET", TEST_SECRET)
        self.auth_patch.start()
        self.model_patch = patch.object(clip_service, "load_model", return_value=(FakeModel(), FakeProcessor()))
        self.model_patch.start()
        self.dimension_patch = patch.object(clip_service, "EXPECTED_EMBEDDING_DIMENSION", 3)
        self.dimension_patch.start()
        clip_service.embedding_dimension = 3
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        # Isolate this file's request count from the shared, module-global
        # per-minute rate limiter (app.rate_limit_windows) -- otherwise the
        # cumulative request count from tests that ran earlier in the same
        # `unittest discover`/pytest process can trip 429s here even though
        # nothing in this file made an excessive number of requests itself.
        clip_service.rate_limit_windows.clear()
        self.client = TestClient(clip_service.app)

    def tearDown(self):
        self.dimension_patch.stop()
        self.model_patch.stop()
        self.auth_patch.stop()
        clip_service.embedding_dimension = None
        clip_service.preprocessing_version = None
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())

    def create_item(self, item_id="item-1", **header_kwargs):
        return self.client.post(
            "/items",
            headers=identity_headers(request_id=f"create-{item_id}", **header_kwargs),
            data={"id": item_id, "title": "Festival synthetic wallet"},
            files={"file": ("item.png", make_image_bytes(), "image/png")},
        )

    def release_item(self, item_id="item-1", **header_kwargs):
        return self.client.post(
            f"/items/{item_id}/release",
            headers=identity_headers(request_id=f"release-{item_id}", actions=["corpus:release"], **header_kwargs),
        )

    def reembed_item(self, item_id="item-1", **header_kwargs):
        return self.client.post(
            f"/items/{item_id}/re-embed",
            headers=identity_headers(request_id=f"reembed-{item_id}", actions=["corpus:write"], **header_kwargs),
            files={"file": ("item2.png", make_image_bytes((0, 255, 0)), "image/png")},
        )

    def match(self, **header_kwargs):
        return self.client.post(
            "/match",
            headers=identity_headers(request_id="match-req", actions=["match:execute"], **header_kwargs),
            data={"text": "wallet"},
        )

    # -- Section 7: missing / malformed demoData ---------------------------

    def test_missing_demo_data_is_rejected(self):
        response = self.create_item(omit=["demoData"])
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "missing_demo_data")

    def test_demo_data_as_string_is_rejected(self):
        # A hand-crafted claim set bypassing identity_headers' Python bool
        # typing, to prove the check is on the JSON type, not just Python's.
        claims = {
            "iss": internal_auth.JWT_ISSUER,
            "aud": internal_auth.JWT_AUDIENCE,
            "service": "festival-backend",
            "tenantId": "tenant-a",
            "siteId": "site-1",
            "siteIds": ["site-1"],
            "actions": ["match:execute"],
            "requestId": "req-string-demo",
            "exp": int(time.time()) + 300,
            "demoData": "true",
            "datasetVersion": "festival-v1",
        }
        token = internal_auth.sign_internal_token(claims, TEST_SECRET)
        response = self.client.post(
            "/match",
            headers={"Authorization": f"Bearer {token}", "X-Request-Id": "req-string-demo"},
            data={"text": "wallet"},
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "invalid_demo_data")

    # -- Section 7: datasetVersion contract when demoData=true --------------

    def test_demo_true_missing_dataset_version_is_rejected(self):
        response = self.create_item(demo_data=True, omit=["datasetVersion"])
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "invalid_dataset_version")

    def test_demo_true_blank_dataset_version_is_rejected(self):
        response = self.create_item(demo_data=True, dataset_version="   ")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "invalid_dataset_version")

    def test_demo_true_valid_dataset_version_passes(self):
        response = self.create_item(demo_data=True, dataset_version="festival-2026.1")
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertTrue(item["demoData"])
        self.assertEqual(item["datasetVersion"], "festival-2026.1")

    # -- Section 7: production (demoData=false) semantics -------------------

    def test_demo_false_with_null_dataset_version_passes(self):
        response = self.create_item(demo_data=False, dataset_version=None)
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertFalse(item["demoData"])
        self.assertIsNone(item["datasetVersion"])

    def test_demo_false_never_becomes_festival_2026(self):
        response = self.create_item(demo_data=False, dataset_version=None)
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.json()["item"]["datasetVersion"], "festival-2026")
        self.assertIsNone(response.json()["item"]["datasetVersion"])

    def test_demo_false_with_a_dataset_version_is_rejected(self):
        # Contradictory: production identity that also claims a festival
        # dataset version. Must never be silently accepted (e.g. by
        # ignoring the version) -- that would hide a Backend signing bug.
        response = self.create_item(demo_data=False, dataset_version="festival-2026.1")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "contradictory_demo_claims")

    # -- Section 7: corpus registration fail-closed consistency -------------

    def test_client_payload_cannot_set_demo_data_or_dataset_version(self):
        response = self.client.post(
            "/items",
            headers=identity_headers(demo_data=True, dataset_version="festival-v1"),
            json={"id": "forged-item", "title": "x", "demoData": False, "datasetVersion": "prod"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "system_fields_forbidden")

    def test_release_by_a_contradictory_identity_is_rejected(self):
        created = self.create_item("mismatch-1", demo_data=True, dataset_version="festival-v1")
        self.assertEqual(created.status_code, 200)
        # A production identity (demoData=False) must not be able to
        # release an item that was created as demo data.
        response = self.release_item("mismatch-1", demo_data=False, dataset_version=None)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "governance_mismatch")

    def test_release_by_a_different_dataset_version_identity_is_rejected(self):
        created = self.create_item("mismatch-2", demo_data=True, dataset_version="festival-v1")
        self.assertEqual(created.status_code, 200)
        response = self.release_item("mismatch-2", demo_data=True, dataset_version="festival-v2")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "governance_mismatch")

    def test_release_by_the_matching_identity_succeeds(self):
        created = self.create_item("match-1", demo_data=True, dataset_version="festival-v1")
        self.assertEqual(created.status_code, 200)
        response = self.release_item("match-1", demo_data=True, dataset_version="festival-v1")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["item"]["released"])

    def test_reembed_by_a_contradictory_identity_is_rejected(self):
        created = self.create_item("mismatch-3", demo_data=False, dataset_version=None)
        self.assertEqual(created.status_code, 200)
        response = self.reembed_item("mismatch-3", demo_data=True, dataset_version="festival-v1")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "governance_mismatch")

    def test_update_by_a_contradictory_identity_is_rejected_and_does_not_overwrite(self):
        created = self.create_item("mismatch-4", demo_data=True, dataset_version="festival-v1")
        self.assertEqual(created.status_code, 200)
        # A second POST /items for the same id, signed by a production
        # identity, must not silently reclassify the item's governance.
        response = self.client.post(
            "/items",
            headers=identity_headers(
                request_id="update-mismatch-4", demo_data=False, dataset_version=None
            ),
            data={"id": "mismatch-4", "title": "Updated title"},
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "governance_mismatch")
        # Confirm the stored item's governance was not touched by the
        # rejected attempt.
        stored = self.client.get(
            "/items/mismatch-4",
            headers=identity_headers(request_id="check-mismatch-4", actions=["corpus:read"]),
        )
        self.assertEqual(stored.status_code, 200)
        self.assertTrue(stored.json()["item"]["demoData"])
        self.assertEqual(stored.json()["item"]["datasetVersion"], "festival-v1")

    # -- Section 5: /match candidate dataset-scope filtering -----------------

    def test_demo_query_does_not_match_a_different_demo_dataset(self):
        created = self.create_item("seed-v1-item", demo_data=True, dataset_version="festival-v1")
        self.assertEqual(created.status_code, 200)
        released = self.release_item("seed-v1-item", demo_data=True, dataset_version="festival-v1")
        self.assertEqual(released.status_code, 200)

        # A different demo dataset version must not see v1's candidates.
        response = self.match(demo_data=True, dataset_version="festival-v2")
        self.assertEqual(response.status_code, 200)
        candidate_ids = [result["item"]["id"] for result in response.json()["topK"]]
        self.assertNotIn("seed-v1-item", candidate_ids)

        # The matching dataset version does see it.
        response = self.match(demo_data=True, dataset_version="festival-v1")
        self.assertEqual(response.status_code, 200)
        candidate_ids = [result["item"]["id"] for result in response.json()["topK"]]
        self.assertIn("seed-v1-item", candidate_ids)

    def test_production_query_never_inherits_festival_candidates(self):
        created = self.create_item("seed-festival-item", demo_data=True, dataset_version="festival-v1")
        self.assertEqual(created.status_code, 200)
        released = self.release_item("seed-festival-item", demo_data=True, dataset_version="festival-v1")
        self.assertEqual(released.status_code, 200)

        response = self.match(demo_data=False, dataset_version=None)
        self.assertEqual(response.status_code, 200)
        candidate_ids = [result["item"]["id"] for result in response.json()["topK"]]
        self.assertNotIn("seed-festival-item", candidate_ids)

    def test_demo_query_never_matches_a_production_item(self):
        created = self.create_item("seed-prod-item", demo_data=False, dataset_version=None)
        self.assertEqual(created.status_code, 200)
        released = self.release_item("seed-prod-item", demo_data=False, dataset_version=None)
        self.assertEqual(released.status_code, 200)

        response = self.match(demo_data=True, dataset_version="festival-v1")
        self.assertEqual(response.status_code, 200)
        candidate_ids = [result["item"]["id"] for result in response.json()["topK"]]
        self.assertNotIn("seed-prod-item", candidate_ids)

    # -- Section 6: preserved security controls (regression guard) ----------

    def test_wrong_issuer_still_rejected(self):
        response = self.match(issuer="not-the-real-issuer")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "invalid_identity")

    def test_wrong_audience_still_rejected(self):
        response = self.match(audience="not-clip-service")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "invalid_identity")

    def test_wrong_action_still_rejected(self):
        response = self.client.post(
            "/items",
            headers=identity_headers(actions=["match:execute"]),
            data={"id": "forbidden", "title": "x"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "action_not_permitted")

    def test_wrong_tenant_still_isolates_reads(self):
        created = self.create_item("tenant-scoped", tenant="tenant-a", site="site-1")
        self.assertEqual(created.status_code, 200)
        response = self.client.get(
            "/items/tenant-scoped",
            headers=identity_headers(request_id="wrong-tenant", tenant="tenant-b", actions=["corpus:read"]),
        )
        self.assertEqual(response.status_code, 404)

    def test_wrong_site_still_rejected(self):
        response = self.client.post(
            "/match",
            headers=identity_headers(request_id="wrong-site", site="site-not-permitted", sites=["site-1"]),
            data={"text": "wallet"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "site_not_permitted")

    def test_missing_x_request_id_still_rejected(self):
        headers = identity_headers(request_id="req-no-header")
        del headers["X-Request-Id"]
        response = self.client.post("/match", headers=headers, data={"text": "wallet"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "missing_request_id")

    def test_mismatched_x_request_id_still_rejected(self):
        headers = identity_headers(request_id="req-signed-value")
        headers["X-Request-Id"] = "req-different-value"
        response = self.client.post("/match", headers=headers, data={"text": "wallet"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "request_id_mismatch")

    def test_public_health_endpoints_unchanged(self):
        live = self.client.get("/health/live")
        self.assertEqual(live.status_code, 200)
        self.assertEqual(set(live.json()), {"status", "requestId"})
        ready = self.client.get("/health/ready")
        self.assertIn(ready.status_code, {200, 503})


if __name__ == "__main__":
    unittest.main()
