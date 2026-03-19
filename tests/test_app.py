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
        self.client.post(
            "/items",
            data={
                "id": "item-red",
                "title": "Red wallet",
                "barcodeValues": "ABC123",
                "ocrText": "Wallet ABC123",
                "labels": "wallet,red",
            },
            files={"file": ("red.png", make_image_bytes((255, 0, 0)), "image/png")},
        )
        self.client.post("/items/item-red/release")

        self.client.post(
            "/items",
            data={
                "id": "item-blue",
                "title": "Blue bag",
                "barcodeValues": "XYZ999",
                "ocrText": "Bag XYZ999",
                "labels": "bag,blue",
            },
            files={"file": ("blue.png", make_image_bytes((0, 0, 255)), "image/png")},
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
        self.assertEqual(payload["topK"][0]["item"]["id"], "item-red")
        self.assertEqual(payload["topK"][0]["explanation"]["signals"]["barcode"], ["ABC123"])
        self.assertIn("wallet", payload["topK"][0]["explanation"]["signals"]["ocr"])
        self.assertEqual(payload["query"]["signals"]["barcode"]["values"], ["ABC123"])

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
