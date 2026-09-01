import io
import os
import time
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

import app as clip_service
import internal_auth
from barcode_service import scan_barcodes
from ocr_service import extract_ocr, ocr_dependency_ready
from test_app import FakeModel, FakeProcessor, TEST_SECRET, identity_headers


# A real TTF (not PIL's built-in default bitmap font) is required here: the
# Arabic test below renders with direction="rtl", which needs a FreeType
# font loaded from a file for Raqm text-shaping to run at all. The font is
# bundled in this repo (tests/assets/fonts/DejaVuSans.ttf) specifically so
# these tests don't depend on a system font installed at some fixed,
# Linux-distro-specific path -- a bare `/usr/share/fonts/...` path breaks on
# macOS, Windows, and minimal/non-Debian Linux images. The system paths
# below are kept only as a defensive fallback for a checkout where the
# bundled asset is somehow missing (e.g. a broken sparse checkout).
_BUNDLED_FONT_PATH = os.path.join(os.path.dirname(__file__), "assets", "fonts", "DejaVuSans.ttf")
_FALLBACK_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",  # Fedora/RHEL/CentOS
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",  # openSUSE
    "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Arch
    "/opt/homebrew/share/fonts/DejaVuSans.ttf",  # Homebrew, Apple Silicon
    "/usr/local/share/fonts/DejaVuSans.ttf",  # Homebrew, Intel / manual installs
    "C:\\Windows\\Fonts\\DejaVuSans.ttf",
]


def _resolve_font_path():
    for candidate in [_BUNDLED_FONT_PATH, *_FALLBACK_FONT_PATHS]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "No DejaVu Sans font found for OCR test image generation. Expected "
        f"the bundled copy at {_BUNDLED_FONT_PATH} -- check the repository "
        "checkout is complete."
    )


FONT_PATH = _resolve_font_path()


def text_image(text, *, direction=None):
    image = Image.new("RGB", (1400, 260), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, 92)
    kwargs = {"direction": direction} if direction else {}
    draw.text((40, 60), text, fill="black", font=font, **kwargs)
    return image


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@unittest.skipUnless(ocr_dependency_ready("eng+ara"), "Tesseract English and Arabic data are required")
class RealOcrIntegrationTests(unittest.TestCase):
    def test_english_ocr_from_real_image(self):
        result = extract_ocr(text_image("DUBAI FESTIVAL"), lang="eng", psm=6)
        self.assertIn("DUBAI", result["fullText"].upper())
        self.assertIn("FESTIVAL", result["fullText"].upper())

    def test_arabic_ocr_from_real_image(self):
        result = extract_ocr(text_image("مهرجان دبي", direction="rtl"), lang="ara", psm=6)
        normalized = result["fullText"].replace(" ", "")
        self.assertIn("دبي", normalized)

    def test_mixed_arabic_english_ocr_from_real_image(self):
        result = extract_ocr(text_image("DUBAI 2026 دبي"), lang="eng+ara", psm=6)
        compact = result["fullText"].replace(" ", "")
        self.assertIn("DUBAI", compact.upper())
        self.assertIn("2026", compact)
        self.assertIn("دبي", compact)

    def test_numeric_identifier_extraction_from_real_image(self):
        result = extract_ocr(text_image("DXB 2026 0417"), lang="eng", psm=6)
        compact = result["fullText"].replace(" ", "")
        self.assertIn("2026", compact)
        self.assertIn("0417", compact)


class RealBarcodeIntegrationTests(unittest.TestCase):
    def test_barcode_extraction_from_real_generated_image(self):
        import zxingcpp

        barcode = zxingcpp.create_barcode("DXB-FESTIVAL-2026-0417", zxingcpp.BarcodeFormat.Code128)
        generated = zxingcpp.write_barcode_to_image(barcode, scale=4, add_hrt=True)
        image = Image.fromarray(np.asarray(generated)).convert("RGB")
        result = scan_barcodes(image)
        self.assertIn("DXB-FESTIVAL-2026-0417", [entry["text"] for entry in result["barcodes"]])


class OcrFailureIntegrationTests(unittest.TestCase):
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
        self.actual_image = png_bytes(text_image("DUBAI 2026"))

    def tearDown(self):
        self.dimension_patch.stop()
        self.model_patch.stop()
        self.auth_patch.stop()
        clip_service.embedding_dimension = None

    def analyze(self):
        return self.client.post(
            "/analyze-image",
            headers=identity_headers(actions=["match:execute"]),
            files={"file": ("ocr.png", self.actual_image, "image/png")},
        )

    def analyze_bytes(self, payload, filename="image.png"):
        return self.client.post(
            "/analyze-image",
            headers=identity_headers(actions=["match:execute"]),
            files={"file": (filename, payload, "image/png")},
        )

    @unittest.skipUnless(ocr_dependency_ready("eng+ara"), "Tesseract English and Arabic data are required")
    def test_http_path_executes_real_ocr(self):
        response = self.analyze()
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["ocrError"])
        self.assertIn("DUBAI", response.json()["ocr"]["fullText"].upper())

    def test_http_path_extracts_real_barcode(self):
        import zxingcpp

        barcode = zxingcpp.create_barcode("DXB-HTTP-2026", zxingcpp.BarcodeFormat.Code128)
        generated = zxingcpp.write_barcode_to_image(barcode, scale=4, add_hrt=True)
        payload = png_bytes(Image.fromarray(np.asarray(generated)).convert("RGB"))
        response = self.analyze_bytes(payload, "barcode.png")
        self.assertEqual(response.status_code, 200)
        self.assertIn("DXB-HTTP-2026", [entry["text"] for entry in response.json()["barcode"]["barcodes"]])

    def test_invalid_image(self):
        response = self.client.post(
            "/analyze-image",
            headers=identity_headers(actions=["match:execute"]),
            files={"file": ("invalid.png", b"\x89PNG invalid", "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_image")

    def test_oversized_image(self):
        with patch.object(clip_service, "MAX_UPLOAD_BYTES", len(self.actual_image) - 1):
            response = self.analyze()
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "image_too_large")

    def test_ocr_timeout_uses_actual_image_request(self):
        def slow_ocr(*args, **kwargs):
            time.sleep(0.2)
            return extract_ocr(*args, **kwargs)

        with patch.object(clip_service, "_extract_ocr", slow_ocr), patch.object(clip_service, "OCR_TIMEOUT_MS", 20):
            response = self.analyze()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ocrError"], "OCR timed out.")

    def test_ocr_disabled(self):
        with patch.object(clip_service, "OCR_ENABLED", False):
            response = self.analyze()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["governance"]["ocrEnabled"])
        self.assertIsNone(response.json()["ocr"])

    def test_missing_ocr_dependency(self):
        with patch.object(clip_service, "_extract_ocr", None):
            response = self.analyze()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ocrError"], "OCR service is not available.")


class FontResolutionPortabilityTests(unittest.TestCase):
    """
    P2-11: text_image() must not depend on a font installed at a fixed,
    Linux-distro-specific system path -- it should resolve to the font
    bundled in this repo on any platform, including one with none of the
    Linux/macOS/Windows system font paths present at all.
    """

    def test_bundled_font_is_present_and_loadable(self):
        self.assertTrue(
            os.path.isfile(_BUNDLED_FONT_PATH),
            "the bundled DejaVuSans.ttf is missing from the checkout",
        )
        # Loading it (not just checking existence) proves it's a real,
        # parseable TTF -- not just a placeholder file.
        ImageFont.truetype(_BUNDLED_FONT_PATH, 92)

    def test_resolves_to_bundled_font_even_when_no_system_font_exists(self):
        real_isfile = os.path.isfile

        def only_bundled_exists(path):
            if path == _BUNDLED_FONT_PATH:
                return real_isfile(path)
            return False

        with patch.object(os.path, "isfile", side_effect=only_bundled_exists):
            resolved = _resolve_font_path()
        self.assertEqual(resolved, _BUNDLED_FONT_PATH)

    def test_raises_a_clear_error_when_nothing_is_found(self):
        with patch.object(os.path, "isfile", return_value=False):
            with self.assertRaises(FileNotFoundError):
                _resolve_font_path()

    def test_falls_back_to_a_system_path_if_the_bundled_font_is_missing(self):
        real_isfile = os.path.isfile
        fallback = _FALLBACK_FONT_PATHS[0]

        def bundled_missing(path):
            if path == _BUNDLED_FONT_PATH:
                return False
            if path == fallback:
                return True
            return real_isfile(path)

        with patch.object(os.path, "isfile", side_effect=bundled_missing):
            resolved = _resolve_font_path()
        self.assertEqual(resolved, fallback)


if __name__ == "__main__":
    unittest.main()
