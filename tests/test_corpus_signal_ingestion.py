"""
F-03 focused tests: corpus ingestion (POST /items, POST /items/{id}/re-embed)
must analyze an uploaded image for OCR/barcode signals itself, using the
existing bounded run_ocr/run_barcode_scan services -- before this fix,
/items only ever stored whatever ocrText/ocrWords/barcodes/barcodeValues
the caller happened to supply manually; an uploaded image's actual OCR
text and barcodes were silently discarded, permanently, since nothing
ever re-analyzed the image later.

These tests prove: automatic analysis runs and is persisted with
provenance (source/engine/timestamp/error) on both /items and re-embed;
a manually supplied signal in the same request takes precedence and
skips automatic analysis (the documented precedence policy); a disabled
or missing OCR dependency degrades to an "unavailable" provenance
without failing item creation; a scan timeout degrades the same way;
re-embed reprocessing the same image twice is idempotent; and a
corpus item's auto-detected barcode is actually usable end-to-end by
/match's retrieval explanation (a distinct, testable scalar reason),
not just a manually-supplied one.
"""
import io
import time
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app as clip_service
import internal_auth
from ocr_service import ocr_dependency_ready
from test_app import FakeModel, FakeProcessor, TEST_SECRET, identity_headers


def blank_image_bytes(color=(240, 240, 240), size=(64, 64)):
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def barcode_image_bytes(text="DXB-CORPUS-2026-0001"):
    import zxingcpp

    barcode = zxingcpp.create_barcode(text, zxingcpp.BarcodeFormat.Code128)
    generated = zxingcpp.write_barcode_to_image(barcode, scale=4, add_hrt=True)
    image = Image.fromarray(np.asarray(generated)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), text


class CorpusSignalIngestionTests(unittest.TestCase):
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

    def create_item(self, item_id, image_bytes, *, extra_data=None, filename="item.png"):
        data = {"id": item_id, "title": "Corpus signal test item"}
        if extra_data:
            data.update(extra_data)
        return self.client.post(
            "/items",
            headers=identity_headers(),
            data=data,
            files={"file": (filename, image_bytes, "image/png")},
        )

    def reembed(self, item_id, image_bytes):
        return self.client.post(
            f"/items/{item_id}/re-embed",
            headers=identity_headers(),
            files={"file": ("item.png", image_bytes, "image/png")},
        )

    # -- barcode auto-detection: real end-to-end, not mocked (zxing-cpp is
    # a pure-Python wheel, no external binary needed) --------------------

    def test_barcode_is_auto_detected_and_persisted_with_provenance(self):
        image_bytes, expected_text = barcode_image_bytes()
        response = self.create_item("item-barcode-1", image_bytes)
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertIn(expected_text, item["barcodeValues"])
        self.assertEqual(item["barcodeProvenance"]["source"], "auto")
        self.assertIsNone(item["barcodeProvenance"]["error"])
        self.assertEqual(item["barcodeProvenance"]["engine"], "zxing-cpp")

    def test_blank_image_yields_no_barcode_but_no_failure(self):
        response = self.create_item("item-blank-1", blank_image_bytes())
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertEqual(item["barcodeValues"], [])
        self.assertEqual(item["barcodeProvenance"]["source"], "auto")
        self.assertIsNone(item["barcodeProvenance"]["error"])

    def test_manual_signals_take_precedence_over_automatic_analysis(self):
        image_bytes, detected_text = barcode_image_bytes("DXB-AUTO-VALUE-0001")
        response = self.create_item(
            "item-manual-1",
            image_bytes,
            extra_data={"barcodeValues": "MANUAL-OVERRIDE-VALUE", "ocrText": "manual ocr text"},
        )
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        # The manually supplied value wins outright -- the real barcode
        # baked into the uploaded image must NOT appear, proving automatic
        # analysis was skipped rather than merged/appended.
        self.assertEqual(item["barcodeValues"], ["MANUAL-OVERRIDE-VALUE"])
        self.assertNotIn(detected_text, item["barcodeValues"])
        self.assertEqual(item["barcodeProvenance"]["source"], "manual")
        self.assertEqual(item["ocrText"], "manual ocr text")
        self.assertEqual(item["ocrProvenance"]["source"], "manual")

    def test_ocr_disabled_degrades_to_unavailable_without_failing_ingestion(self):
        with patch.object(clip_service, "OCR_ENABLED", False):
            response = self.create_item("item-ocr-disabled-1", blank_image_bytes())
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertIsNone(item["ocrText"])
        self.assertEqual(item["ocrProvenance"]["source"], "unavailable")
        self.assertEqual(item["ocrProvenance"]["error"], "OCR is disabled by service configuration.")

    def test_missing_ocr_dependency_degrades_to_unavailable(self):
        with patch.object(clip_service, "_extract_ocr", None):
            response = self.create_item("item-ocr-missing-1", blank_image_bytes())
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertEqual(item["ocrProvenance"]["source"], "unavailable")
        self.assertEqual(item["ocrProvenance"]["error"], "OCR service is not available.")

    def test_barcode_scan_timeout_degrades_gracefully(self):
        def slow_scan(image):
            time.sleep(0.2)
            return {"barcodes": [], "meta": {}}

        with patch.object(clip_service, "_scan_barcodes", slow_scan), patch.object(
            clip_service, "BARCODE_TIMEOUT_MS", 20
        ):
            response = self.create_item("item-barcode-timeout-1", blank_image_bytes())
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertEqual(item["barcodeValues"], [])
        self.assertEqual(item["barcodeProvenance"]["source"], "auto")
        self.assertEqual(item["barcodeProvenance"]["error"], "Barcode scan timed out.")

    def test_reembed_reprocesses_signals_and_is_idempotent(self):
        image_bytes, expected_text = barcode_image_bytes("DXB-REEMBED-0001")
        create_response = self.create_item("item-reembed-1", blank_image_bytes())
        self.assertEqual(create_response.status_code, 200)
        self.assertEqual(create_response.json()["item"]["barcodeValues"], [])

        first = self.reembed("item-reembed-1", image_bytes)
        self.assertEqual(first.status_code, 200)
        first_item = first.json()["item"]
        self.assertEqual(first_item["barcodeValues"], [expected_text])
        self.assertEqual(first_item["barcodeProvenance"]["source"], "auto")

        second = self.reembed("item-reembed-1", image_bytes)
        self.assertEqual(second.status_code, 200)
        second_item = second.json()["item"]
        self.assertEqual(second_item["barcodeValues"], first_item["barcodeValues"])
        self.assertEqual(second_item["ocrText"], first_item["ocrText"])

    def test_reembed_with_new_image_replaces_stale_signals(self):
        old_bytes, old_text = barcode_image_bytes("DXB-STALE-0001")
        new_bytes, new_text = barcode_image_bytes("DXB-FRESH-0002")
        self.create_item("item-reembed-2", old_bytes)
        response = self.reembed("item-reembed-2", new_bytes)
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertIn(new_text, item["barcodeValues"])
        self.assertNotIn(old_text, item["barcodeValues"])

    def test_auto_detected_barcode_is_usable_in_match_explanation(self):
        image_bytes, shared_text = barcode_image_bytes("DXB-EXPLAIN-0001")
        create_response = self.create_item("item-explain-1", image_bytes)
        self.assertEqual(create_response.status_code, 200)
        release_response = self.client.post(
            "/items/item-explain-1/release",
            headers=identity_headers(actions=["corpus:release"]),
        )
        self.assertEqual(release_response.status_code, 200)

        match_response = self.client.post(
            "/match",
            headers=identity_headers(actions=["match:execute"]),
            data={"barcodeValues": shared_text},
            files={"file": ("query.png", blank_image_bytes(), "image/png")},
        )
        self.assertEqual(match_response.status_code, 200)
        body = match_response.json()
        self.assertGreaterEqual(len(body["topK"]), 1)
        top = body["topK"][0]
        # Distinct, testable scalar reasons -- barcode agreement is its
        # own counted signal, separate from OCR/labels/visual similarity.
        self.assertEqual(top["explanation"]["barcode"]["matchCount"], 1)
        self.assertIn(shared_text, top["explanation"]["barcode"]["matchedValues"])
        self.assertIn("barcode overlap", top["explanation"]["reason"])


@unittest.skipUnless(ocr_dependency_ready("eng+ara"), "Tesseract English and Arabic data are required")
class CorpusSignalIngestionRealOcrTests(unittest.TestCase):
    """
    Gated exactly like tests/test_ocr_integration.py's real-OCR classes --
    skipped where the tesseract binary/language data isn't installed
    (this sandbox), but real end-to-end proof wherever it is (CI).
    """

    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

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

    def _text_image_bytes(self, text, direction=None):
        from PIL import ImageDraw, ImageFont

        image = Image.new("RGB", (1400, 260), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(self.FONT_PATH, 92)
        kwargs = {"direction": direction} if direction else {}
        draw.text((40, 60), text, fill="black", font=font, **kwargs)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_english_text_is_auto_extracted_at_ingestion(self):
        response = self.client.post(
            "/items",
            headers=identity_headers(),
            data={"id": "item-ocr-en-1", "title": "English OCR corpus item"},
            files={"file": ("item.png", self._text_image_bytes("DUBAI FESTIVAL"), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertIn("DUBAI", (item["ocrText"] or "").upper())
        self.assertEqual(item["ocrProvenance"]["source"], "auto")
        self.assertEqual(item["ocrProvenance"]["lang"], "eng+ara")

    def test_arabic_text_is_auto_extracted_at_ingestion(self):
        response = self.client.post(
            "/items",
            headers=identity_headers(),
            data={"id": "item-ocr-ar-1", "title": "Arabic OCR corpus item"},
            files={
                "file": (
                    "item.png",
                    self._text_image_bytes("مهرجان دبي", direction="rtl"),
                    "image/png",
                )
            },
        )
        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        normalized = (item["ocrText"] or "").replace(" ", "")
        self.assertIn("دبي", normalized)


if __name__ == "__main__":
    unittest.main()
