"""
F-07 focused tests: a lost report with no photo (the frontend allows
this, requiring a longer description instead) must still be able to
become a real, matchable corpus item -- before this fix, POST /items
only ever computed an embedding when an image was uploaded, so a
text-only item could never become eligibleForMatching regardless of its
released state, and there was no way to tell "no evidence yet" apart
from "this candidate just doesn't visually resemble anything".

These tests prove: a text-only item gets a real text embedding tagged
with embeddingModality "text"; an item with neither an image nor usable
text stays unmatchable without fabricating anything; a text-only
candidate is actually findable by both an image query and a text query
(CLIP's joint embedding space makes a text embedding directly comparable
to an image query's embedding, not just other text); the match
explanation reflects "text similarity (no image evidence)" rather than
misrepresenting it as ordinary visual similarity; and an empty/whitespace
description never fabricates an embedding.
"""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as clip_service
import internal_auth
from test_app import FakeModel, FakeProcessor, TEST_SECRET, identity_headers, make_image_bytes


class TextOnlyCorpusTests(unittest.TestCase):
    def setUp(self):
        self.auth_patch = patch.object(internal_auth, "INTERNAL_JWT_SECRET", TEST_SECRET)
        self.auth_patch.start()
        self.model_patch = patch.object(clip_service, "load_model", return_value=(FakeModel(), FakeProcessor()))
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

    def create_text_only_item(self, item_id, title=None, description=None):
        data = {"id": item_id}
        if title is not None:
            data["title"] = title
        if description is not None:
            data["description"] = description
        return self.client.post("/items", headers=identity_headers(), data=data)

    def create_image_item(self, item_id, color=(255, 0, 0)):
        return self.client.post(
            "/items",
            headers=identity_headers(),
            data={"id": item_id, "title": "Image item"},
            files={"file": ("item.png", make_image_bytes(color), "image/png")},
        )

    def release(self, item_id):
        return self.client.post(f"/items/{item_id}/release", headers=identity_headers(actions=["corpus:release"]))

    def match(self, **kwargs):
        return self.client.post("/match", headers=identity_headers(actions=["match:execute"]), **kwargs)

    def test_text_only_item_gets_a_real_text_embedding(self):
        response = self.create_text_only_item("lost-1", title="Red wallet", description="Lost near the north gate")
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertEqual(item["embeddingModality"], "text")
        self.assertEqual(item["embeddingDimension"], 3)
        self.assertFalse(item["eligibleForMatching"], "not released yet")

        release_response = self.release("lost-1")
        self.assertEqual(release_response.status_code, 200)
        released_item = release_response.json()["item"]
        self.assertEqual(released_item["status"], "released")
        self.assertTrue(released_item["eligibleForMatching"])
        self.assertEqual(released_item["embeddingModality"], "text")

    def test_no_image_and_no_usable_text_stays_unmatchable_without_fabricating_an_embedding(self):
        response = self.create_text_only_item("lost-2")
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertIsNone(item["embeddingModality"])
        self.assertFalse(item["eligibleForMatching"])
        self.assertEqual(item["status"], "stored")

    def test_whitespace_only_description_does_not_fabricate_an_embedding(self):
        response = self.create_text_only_item("lost-3", title="   ", description="   \n\t  ")
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertIsNone(item["embeddingModality"])
        self.assertFalse(item["eligibleForMatching"])

    def test_text_only_candidate_is_findable_by_an_image_query(self):
        create_response = self.create_text_only_item("lost-red-1", title="Red wallet", description="red leather wallet")
        self.assertEqual(create_response.status_code, 200)
        self.assertEqual(self.release("lost-red-1").status_code, 200)

        match_response = self.match(files={"file": ("query.png", make_image_bytes((255, 0, 0)), "image/png")})
        self.assertEqual(match_response.status_code, 200)
        body = match_response.json()
        top_ids = [candidate["item"]["id"] for candidate in body["topK"]]
        self.assertIn("lost-red-1", top_ids)
        matched = next(c for c in body["topK"] if c["item"]["id"] == "lost-red-1")
        self.assertEqual(matched["explanation"]["candidateEmbeddingModality"], "text")
        self.assertIn("text similarity", matched["explanation"]["reason"])

    def test_text_only_candidate_is_findable_by_a_text_query(self):
        self.assertEqual(
            self.create_text_only_item("lost-red-2", title="Red wallet", description="red leather wallet").status_code, 200
        )
        self.assertEqual(self.release("lost-red-2").status_code, 200)

        match_response = self.match(data={"text": "red"})
        self.assertEqual(match_response.status_code, 200)
        top_ids = [candidate["item"]["id"] for candidate in match_response.json()["topK"]]
        self.assertIn("lost-red-2", top_ids)

    def test_multilingual_description_is_handled_without_error(self):
        response = self.create_text_only_item("lost-ar-1", title="محفظة حمراء", description="فقدت بالقرب من البوابة الشمالية")
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertEqual(item["embeddingModality"], "text")
        self.assertEqual(item["embeddingDimension"], 3)

    def test_image_backed_candidate_still_reports_image_modality_and_visual_similarity_reason(self):
        self.assertEqual(self.create_image_item("found-1", color=(0, 0, 255)).status_code, 200)
        self.assertEqual(self.release("found-1").status_code, 200)

        match_response = self.match(files={"file": ("query.png", make_image_bytes((0, 0, 255)), "image/png")})
        self.assertEqual(match_response.status_code, 200)
        matched = next(c for c in match_response.json()["topK"] if c["item"]["id"] == "found-1")
        self.assertEqual(matched["explanation"]["candidateEmbeddingModality"], "image")
        self.assertIn("visual similarity", matched["explanation"]["reason"])

    def test_embeddingModality_is_a_system_field_callers_cannot_set_directly(self):
        response = self.client.post(
            "/items",
            headers=identity_headers(),
            data={"id": "lost-4", "title": "x", "embeddingModality": "text"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "system_fields_forbidden")


if __name__ == "__main__":
    unittest.main()
