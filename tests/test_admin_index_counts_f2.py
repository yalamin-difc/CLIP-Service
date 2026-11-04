"""F2: read-only GET /admin/index-counts -- per tenant/site counts of
indexed items, so a pilot index can be checked empty before go-live.

Covers: off by default (404), the dedicated index:count action, tenant
and site isolation (a second tenant can never see or affect the counts),
counts only (no item content), the audit entry, and the MongoDB
aggregation's tenant/site filter.
"""
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as clip_service
import internal_auth
from test_app import TEST_SECRET, identity_headers

COUNT = ["index:count"]


def seed(repository, tenant, site, item_id, *, status="released", released=True, demo=False, title="Black wallet"):
    repository.upsert_item(
        {
            "id": item_id,
            "tenantId": tenant,
            "siteId": site,
            "status": status,
            "released": released,
            "demoData": demo,
            "datasetVersion": "festival-2026.1" if demo else None,
            "title": title,
            "embedding": [0.1, 0.2, 0.3],
        }
    )


class AdminIndexCountsBase(unittest.TestCase):
    enabled = True

    def setUp(self):
        self.auth_patch = patch.object(internal_auth, "INTERNAL_JWT_SECRET", TEST_SECRET)
        self.auth_patch.start()
        self.flag_patch = patch.object(clip_service, "ADMIN_INDEX_COUNTS_ENABLED", self.enabled)
        self.flag_patch.start()
        self.repository = clip_service.InMemoryItemRepository()
        clip_service.set_item_repository(self.repository)
        clip_service.rate_limit_windows.clear()
        self.client = TestClient(clip_service.app)
        # tenant-a: two sites; tenant-b: a site with the SAME name as tenant-a's.
        seed(self.repository, "tenant-a", "site-1", "a1", demo=True, title="Secret title A1")
        seed(self.repository, "tenant-a", "site-1", "a2", status="stored", released=False)
        seed(self.repository, "tenant-a", "site-2", "a3")
        seed(self.repository, "tenant-b", "site-1", "b1", demo=True, title="Other tenant B1")
        seed(self.repository, "tenant-b", "site-1", "b2")
        seed(self.repository, "tenant-b", "site-1", "b3")

    def tearDown(self):
        self.flag_patch.stop()
        self.auth_patch.stop()
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        clip_service.rate_limit_windows.clear()

    def get(self, **identity):
        identity.setdefault("actions", COUNT)
        identity.setdefault("demo_data", False)
        identity.setdefault("dataset_version", None)
        return self.client.get("/admin/index-counts", headers=identity_headers(**identity))


class DisabledByDefaultTests(AdminIndexCountsBase):
    enabled = False

    def test_flag_defaults_to_off(self):
        self.flag_patch.stop()
        try:
            self.assertFalse(clip_service.ADMIN_INDEX_COUNTS_ENABLED)
        finally:
            self.flag_patch.start()

    def test_disabled_returns_404_even_with_a_valid_token(self):
        response = self.get(tenant="tenant-a", site="site-1")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("total", response.text)

    def test_disabled_returns_404_without_a_token(self):
        self.assertEqual(self.client.get("/admin/index-counts").status_code, 404)


class AuthorizationTests(AdminIndexCountsBase):
    def test_requires_a_signed_identity(self):
        self.assertEqual(self.client.get("/admin/index-counts").status_code, 401)

    def test_requires_the_dedicated_index_count_action(self):
        for actions in (["corpus:read"], ["corpus:write", "corpus:release", "match:execute", "health:read", "metrics:read"]):
            with self.subTest(actions=actions):
                response = self.get(tenant="tenant-a", site="site-1", actions=actions)
                self.assertEqual(response.status_code, 403)

    def test_index_count_token_cannot_read_or_change_the_corpus(self):
        headers = identity_headers(tenant="tenant-a", site="site-1", actions=COUNT, demo_data=False, dataset_version=None)
        self.assertEqual(self.client.get("/items", headers=headers).status_code, 403)
        self.assertEqual(self.client.get("/items/a1", headers=headers).status_code, 403)
        self.assertEqual(self.client.get("/audit-logs", headers=headers).status_code, 403)
        headers = identity_headers(tenant="tenant-a", site="site-1", actions=COUNT, demo_data=False, dataset_version=None)
        self.assertEqual(self.client.post("/items/a1/release", headers=headers).status_code, 403)


class CountsTests(AdminIndexCountsBase):
    def test_counts_only_the_selected_tenant_and_site(self):
        response = self.get(tenant="tenant-a", site="site-1")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["tenantId"], "tenant-a")
        self.assertEqual(body["total"], 2)
        self.assertEqual(
            body["sites"],
            [
                {
                    "siteId": "site-1",
                    "total": 2,
                    "released": 1,
                    "demo": 1,
                    "nonDemo": 1,
                    "byStatus": {"released": 1, "stored": 1},
                }
            ],
        )

    def test_counts_every_permitted_site(self):
        body = self.get(tenant="tenant-a", site="site-1", sites=["site-1", "site-2"]).json()
        self.assertEqual(body["total"], 3)
        self.assertEqual([site["siteId"] for site in body["sites"]], ["site-1", "site-2"])

    def test_empty_scope_reports_explicit_zero(self):
        body = self.get(tenant="pilot-tenant", site="pilot-site").json()
        self.assertEqual(body["total"], 0)
        self.assertEqual(
            body["sites"],
            [{"siteId": "pilot-site", "total": 0, "released": 0, "demo": 0, "nonDemo": 0, "byStatus": {}}],
        )

    def test_counts_demo_and_production_items_alike_whatever_the_token_says(self):
        # The point is to see whether ANYTHING is indexed in the scope, so a
        # production (demoData=false) token still counts demo items.
        self.assertEqual(self.get(tenant="tenant-a", site="site-1").json()["sites"][0]["demo"], 1)

    def test_returns_counts_only_never_item_content(self):
        text = self.get(tenant="tenant-a", site="site-1", sites=["site-1", "site-2"]).text
        for leaked in ("Secret title A1", "a1", "a2", "a3", "embedding", "title", "0.1"):
            self.assertNotIn(f'"{leaked}"' if leaked in {"a1", "a2", "a3"} else leaked, text)

    def test_read_only_and_audited(self):
        before = {key: dict(value) for key, value in self.repository._items.items()}
        self.get(tenant="tenant-a", site="site-1")
        self.assertEqual({key: dict(value) for key, value in self.repository._items.items()}, before)
        logs = [log for log in self.repository._logs if log["eventType"] == "admin_index_counts_read"]
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["tenantId"], "tenant-a")
        self.assertEqual(logs[0]["payload"]["siteIds"], ["site-1"])
        self.assertEqual(logs[0]["payload"]["total"], 2)
        self.assertNotIn("Secret title A1", json.dumps(logs[0]))


class TenantIsolationSecurityTests(AdminIndexCountsBase):
    def test_second_tenant_cannot_see_first_tenants_counts(self):
        # Same site name, different tenant: only tenant-b's own 3 items.
        body = self.get(tenant="tenant-b", site="site-1").json()
        self.assertEqual(body["tenantId"], "tenant-b")
        self.assertEqual(body["total"], 3)

    def test_second_tenant_cannot_reach_a_site_it_was_not_granted(self):
        # tenant-b asks for site-2 (which only tenant-a uses): explicit zero,
        # never tenant-a's item.
        body = self.get(tenant="tenant-b", site="site-2").json()
        self.assertEqual(body["total"], 0)

    def test_query_parameters_cannot_widen_the_scope(self):
        headers = identity_headers(tenant="tenant-b", site="site-1", actions=COUNT, demo_data=False, dataset_version=None)
        response = self.client.get("/admin/index-counts?tenantId=tenant-a&siteId=site-2", headers=headers)
        self.assertEqual(response.json()["tenantId"], "tenant-b")
        self.assertEqual(response.json()["total"], 3)

    def test_unauthorised_counting_does_not_change_the_other_tenants_data(self):
        before = {key: dict(value) for key, value in self.repository._items.items()}
        self.get(tenant="tenant-b", site="site-1")
        self.get(tenant="tenant-b", site="site-1", actions=["corpus:read"])
        self.assertEqual({key: dict(value) for key, value in self.repository._items.items()}, before)


class FakeAggregateCollection:
    def __init__(self, rows):
        self.rows = rows
        self.pipelines = []

    def aggregate(self, pipeline):
        self.pipelines.append(pipeline)
        return iter(self.rows)


class MongoRepositoryCountTests(unittest.TestCase):
    def make_repository(self, rows):
        repository = object.__new__(clip_service.MongoItemRepository)
        repository._items = FakeAggregateCollection(rows)
        return repository

    def identity(self, tenant="tenant-a", sites=("site-1",)):
        return internal_auth.ServiceIdentity(
            service_name="backend-api",
            tenant_id=tenant,
            site_id=sites[0],
            permitted_site_ids=frozenset(sites),
            actions=frozenset(COUNT),
            request_id="req-1",
            dataset_version=None,
            demo_data=False,
        )

    def test_aggregation_is_filtered_by_tenant_and_permitted_sites(self):
        repository = self.make_repository([])
        repository.count_items_by_site(self.identity(sites=("site-2", "site-1")))
        match = repository._items.pipelines[0][0]["$match"]
        self.assertEqual(match, {"tenantId": "tenant-a", "siteId": {"$in": ["site-1", "site-2"]}})

    def test_folds_grouped_rows_and_ignores_rows_outside_the_scope(self):
        rows = [
            {"_id": {"siteId": "site-1", "status": "released", "released": True, "demo": True}, "count": 4},
            {"_id": {"siteId": "site-1", "status": None, "released": False, "demo": False}, "count": 1},
            # A row for a site outside the identity's scope must never be counted.
            {"_id": {"siteId": "site-9", "status": "released", "released": True, "demo": False}, "count": 50},
        ]
        result = self.make_repository(rows).count_items_by_site(self.identity())
        self.assertEqual(
            result,
            [
                {
                    "siteId": "site-1",
                    "total": 5,
                    "released": 4,
                    "demo": 4,
                    "nonDemo": 1,
                    "byStatus": {"released": 4, "unknown": 1},
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
