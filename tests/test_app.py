import io
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app as clip_service


def make_image_bytes(color):
    image = Image.new("RGB", (8, 8), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_vector(image):
    array = np.asarray(image, dtype=float)
    mean_rgb = array.reshape(-1, 3).mean(axis=0)
    return np.asarray(mean_rgb, dtype=float) + 1.0


def text_vector(text):
    lowered = text.lower()
    if "red" in lowered:
        return np.asarray([255.0, 1.0, 1.0], dtype=float)
    if "green" in lowered:
        return np.asarray([1.0, 255.0, 1.0], dtype=float)
    if "blue" in lowered:
        return np.asarray([1.0, 1.0, 255.0], dtype=float)
    return np.asarray([32.0, 32.0, 32.0], dtype=float)


class FakeProcessor:
    def __call__(self, images=None, text=None, **kwargs):
        if images is not None:
            batch = images if isinstance(images, list) else [images]
            return {"image_features": np.asarray([image_vector(image) for image in batch], dtype=float)}
        if text is not None:
            batch = text if isinstance(text, list) else [text]
            return {"text_features": np.asarray([text_vector(value) for value in batch], dtype=float)}
        raise AssertionError("Processor expected images or text")


class FakeModel:
    def get_image_features(self, **inputs):
        return inputs["image_features"]

    def get_text_features(self, **inputs):
        return inputs["text_features"]


class ClipServiceTests(unittest.TestCase):
    def setUp(self):
        clip_service.model = None
        clip_service.processor = None
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        self.model_patch = patch.object(clip_service, "load_model", return_value=(FakeModel(), FakeProcessor()))
        self.model_patch.start()
        self.client = TestClient(clip_service.app)

    def tearDown(self):
        self.model_patch.stop()
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        clip_service.model = None
        clip_service.processor = None

    def upsert_item_with_image(self, item_id, color, **fields):
        payload = {"id": item_id, **fields}
        return self.client.post(
            "/items",
            data=payload,
            files={"file": (f"{item_id}.png", make_image_bytes(color), "image/png")},
        )

    def upsert_item_with_embedding(self, item_id, embedding, **fields):
        payload = {"id": item_id, "clipEmbedding": embedding, **fields}
        return self.client.post("/items", json=payload)

    def test_health_reports_dependencies_and_request_id(self):
        response = self.client.get("/health", headers={"X-Request-Id": "req-health"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-Id"], "req-health")
        payload = response.json()
        self.assertEqual(payload["requestId"], "req-health")
        self.assertEqual(payload["status"], "ok")
        self.assertFalse(payload["dependencies"]["model"]["loaded"])
        self.assertEqual(payload["dependencies"]["store"]["backend"], "memory")

    def test_encode_image_returns_embedding_and_request_id(self):
        response = self.client.post(
            "/encode-image",
            headers={"X-Request-Id": "req-image"},
            files={"file": ("red.png", make_image_bytes((255, 0, 0)), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-Id"], "req-image")
        payload = response.json()
        self.assertEqual(payload["requestId"], "req-image")
        self.assertEqual(len(payload["embedding"]), 3)
        self.assertGreater(payload["embedding"][0], payload["embedding"][1])

    def test_encode_text_returns_embedding_and_request_id(self):
        response = self.client.post(
            "/encode-text",
            headers={"X-Request-Id": "req-text"},
            data={"text": "red wallet"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-Id"], "req-text")
        payload = response.json()
        self.assertEqual(payload["requestId"], "req-text")
        self.assertEqual(len(payload["embedding"]), 3)
        self.assertGreater(payload["embedding"][0], payload["embedding"][2])

    def test_analyze_image_returns_compact_signals(self):
        response = self.client.post(
            "/analyze-image",
            headers={"X-Request-Id": "req-analyze"},
            data={
                "ocrText": "  wallet receipt ABC123  ",
                "barcodeValues": "ABC123, XYZ999",
                "labels": "wallet, receipt",
            },
            files={"file": ("green.png", make_image_bytes((0, 255, 0)), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["requestId"], "req-analyze")
        self.assertEqual(payload["image"]["width"], 8)
        self.assertEqual(payload["signals"]["barcode"]["count"], 2)
        self.assertEqual(payload["signals"]["labels"], ["wallet", "receipt"])
        self.assertIn("ABC123", payload["signals"]["ocr"]["excerpt"])

    def test_items_upsert_and_release_behavior(self):
        upsert = self.client.post(
            "/items",
            data={
                "id": "item-123",
                "title": "Red wallet",
                "barcodeValues": "ABC123",
                "ocrText": "Wallet ABC123",
            },
            files={"file": ("wallet.png", make_image_bytes((255, 0, 0)), "image/png")},
        )

        self.assertEqual(upsert.status_code, 200)
        upsert_payload = upsert.json()
        self.assertEqual(upsert_payload["item"]["id"], "item-123")
        self.assertFalse(upsert_payload["item"]["released"])
        self.assertFalse(upsert_payload["item"]["eligibleForMatching"])

        release = self.client.post("/items/item-123/release", headers={"X-Request-Id": "req-release"})

        self.assertEqual(release.status_code, 200)
        self.assertEqual(release.headers["X-Request-Id"], "req-release")
        release_payload = release.json()
        self.assertTrue(release_payload["item"]["released"])
        self.assertTrue(release_payload["item"]["eligibleForMatching"])
        self.assertEqual(release_payload["item"]["status"], "released")

    def test_match_returns_released_candidates_and_signal_explanations(self):
        self.upsert_item_with_image(
            "item-red",
            (255, 0, 0),
            title="Red wallet",
            barcodeValues="ABC123",
            ocrText="Wallet ABC123",
            labels="wallet,red",
        )
        self.client.post("/items/item-red/release")

        self.upsert_item_with_image(
            "item-blue",
            (0, 0, 255),
            title="Blue bag",
            barcodeValues="XYZ999",
            ocrText="Bag XYZ999",
            labels="bag,blue",
        )

        response = self.client.post(
            "/match",
            headers={"X-Request-Id": "req-match"},
            data={
                "topK": "3",
                "barcodeValues": "ABC123",
                "ocrText": "found wallet abc123",
                "labels": "wallet",
            },
            files={"file": ("query.png", make_image_bytes((255, 0, 0)), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-Id"], "req-match")
        payload = response.json()
        self.assertEqual(payload["requestId"], "req-match")
        self.assertEqual(payload["governance"]["candidatesEvaluated"], 1)
        self.assertEqual(payload["governance"]["scoringVersion"], "clip-match-v1")
        self.assertFalse(payload["decision"]["noMatch"])
        self.assertEqual(payload["decision"]["reason"], "OK")
        self.assertEqual(payload["topK"][0]["item"]["id"], "item-red")
        self.assertEqual(payload["topK"][0]["candidateId"], "item-red")
        self.assertEqual(payload["topK"][0]["explanation"]["signals"]["barcode"], ["ABC123"])
        self.assertIn("wallet", payload["topK"][0]["explanation"]["ocr"]["matchedTerms"])
        self.assertEqual(payload["query"]["signals"]["barcode"]["values"], ["ABC123"])
        self.assertEqual(payload["query"]["modalities"], ["image"])
        self.assertEqual(payload["query"]["thresholds"]["minScore"], clip_service.CONF_MIN_SCORE)

    def test_match_text_only_uses_stable_contract(self):
        self.upsert_item_with_embedding(
            "item-red-text",
            [255.0, 1.0, 1.0],
            title="Red text wallet",
            ocrText="Wallet receipt",
            barcodes=[{"text": "TXT123"}],
            labels=["wallet", "text"],
        )
        self.client.post("/items/item-red-text/release")

        response = self.client.post(
            "/match",
            headers={"X-Request-Id": "req-text-match"},
            data={"text": "red wallet", "topK": "2", "metadata": '{"source":"backend"}'},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query"]["modalities"], ["text"])
        self.assertEqual(payload["query"]["metadata"], {"source": "backend"})
        self.assertEqual(payload["topK"][0]["candidateId"], "item-red-text")
        self.assertEqual(payload["topK"][0]["explanation"]["model"]["scoringVersion"], "clip-match-v1")
        self.assertIn("similarity", payload["topK"][0]["explanation"])

    def test_match_image_and_text_combines_modalities(self):
        self.upsert_item_with_image(
            "item-combo",
            (255, 0, 0),
            title="Combo wallet",
            ocrText="Wallet combo",
            labels="wallet,combo",
        )
        self.client.post("/items/item-combo/release")

        response = self.client.post(
            "/match",
            data={"text": "red wallet", "topK": "1"},
            files={"file": ("combo.png", make_image_bytes((255, 0, 0)), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query"]["modalities"], ["image", "text"])
        self.assertEqual(payload["query"]["type"], "image+text")
        self.assertEqual(payload["topK"][0]["item"]["id"], "item-combo")

    def test_match_returns_explicit_no_match_decision(self):
        self.upsert_item_with_embedding(
            "item-blue-text",
            [1.0, 1.0, 255.0],
            title="Blue bag",
            labels=["bag", "blue"],
        )
        self.client.post("/items/item-blue-text/release")

        response = self.client.post("/match", data={"text": "red wallet", "topK": "1"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["decision"]["noMatch"])
        self.assertEqual(payload["topK"], [])
        self.assertIn(payload["decision"]["reason"], {"LOW_TOP_SCORE", "LOW_MARGIN"})
        self.assertTrue(payload["governance"]["decision"]["noMatch"])

    def test_match_rejects_malformed_request(self):
        response = self.client.post("/match", data={"topK": "oops", "text": "wallet"})

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["success"], False)
        self.assertEqual(payload["error"]["code"], "invalid_top_k")

    def test_match_accepts_legacy_k_alias(self):
        self.upsert_item_with_embedding(
            "item-k-alias",
            [255.0, 1.0, 1.0],
            title="Alias wallet",
            labels=["wallet"],
        )
        self.client.post("/items/item-k-alias/release")

        response = self.client.post("/match", data={"text": "red wallet", "k": "1"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["governance"]["topKRequested"], 1)
        self.assertEqual(len(payload["topK"]), 1)

    def test_match_enforces_bearer_auth_when_configured(self):
        with patch.object(clip_service, "CLIP_API_KEY", "secret-token"):
            unauthorized = self.client.post("/match", data={"text": "wallet"})
            self.assertEqual(unauthorized.status_code, 401)
            self.assertEqual(unauthorized.json()["error"]["code"], "unauthorized")

            authorized = self.client.post(
                "/match",
                headers={"Authorization": "Bearer secret-token"},
                data={"text": "wallet"},
            )
            self.assertEqual(authorized.status_code, 200)

    def test_non_dev_environment_requires_auth_configuration(self):
        with patch.object(clip_service, "ENVIRONMENT", "production"), patch.object(clip_service, "CLIP_API_KEY", ""):
            response = self.client.post("/match", data={"text": "wallet"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "auth_misconfigured")

    def test_match_schema_is_consistent(self):
        self.upsert_item_with_embedding(
            "item-schema",
            [255.0, 1.0, 1.0],
            title="Schema wallet",
            labels=["wallet"],
        )
        self.client.post("/items/item-schema/release")

        response = self.client.post("/match", data={"text": "red wallet", "topK": "1"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload.keys()), {"requestId", "governance", "query", "decision", "topK"})
        self.assertEqual(
            set(payload["topK"][0].keys()),
            {"candidateId", "item", "score", "confidence", "explanation"},
        )
        self.assertEqual(
            set(payload["topK"][0]["explanation"].keys()),
            {"reason", "summary", "scoreBand", "signals", "similarity", "ocr", "barcode", "labels", "model"},
        )

    def test_items_upsert_accepts_clip_embedding_alias(self):
        response = self.client.post(
            "/items",
            json={
                "id": "item-clip-embedding",
                "title": "Embedding alias",
                "clipEmbedding": [255.0, 1.0, 1.0],
            },
        )

        self.assertEqual(response.status_code, 200)
        release = self.client.post("/items/item-clip-embedding/release")
        self.assertEqual(release.status_code, 200)
        self.assertTrue(release.json()["item"]["eligibleForMatching"])

    def test_errors_use_structured_contract(self):
        response = self.client.post("/encode-text", headers={"X-Request-Id": "req-error"}, data={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.headers["X-Request-Id"], "req-error")
        payload = response.json()
        self.assertEqual(payload["success"], False)
        self.assertEqual(payload["error"]["requestId"], "req-error")
        self.assertEqual(payload["error"]["code"], "missing_text")


if __name__ == "__main__":
    unittest.main()
