#!/usr/bin/env python3
"""CPU image smoke checks for CLIP-Service CI.

Validates the running container: liveness, Mongo-backed readiness, English
OCR, Arabic OCR, a generated barcode fixture, and the absence of raw
embeddings on protected HTTP responses.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import time
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import internal_auth
from barcode_service import scan_barcodes
from ocr_service import extract_ocr

BASE_URL = os.environ.get("CLIP_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
FONT_PATH = pathlib.Path(__file__).resolve().parents[1] / "tests" / "assets" / "fonts" / "DejaVuSans.ttf"
BARCODE_VALUE = "DXB-FESTIVAL-2026-0417"
FORBIDDEN_VECTOR_KEYS = {"embedding", "clipEmbedding"}


def fail(message: str) -> None:
    print(f"SMOKE_FAIL {message}", file=sys.stderr)
    raise SystemExit(1)


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def text_image(text: str, *, direction: str | None = None) -> Image.Image:
    image = Image.new("RGB", (1400, 260), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT_PATH), 92)
    kwargs = {"direction": direction} if direction else {}
    draw.text((40, 60), text, fill="black", font=font, **kwargs)
    return image


def barcode_image(value: str = BARCODE_VALUE) -> Image.Image:
    import zxingcpp

    barcode = zxingcpp.create_barcode(value, zxingcpp.BarcodeFormat.Code128)
    generated = zxingcpp.write_barcode_to_image(barcode, scale=4, add_hrt=True)
    return Image.fromarray(np.asarray(generated)).convert("RGB")


def service_headers(action: str, request_id: str) -> dict[str, str]:
    claims = {
        "iss": internal_auth.JWT_ISSUER,
        "aud": internal_auth.JWT_AUDIENCE,
        "service": "clip-ci",
        "tenantId": "tenant-ci",
        "siteId": "site-ci",
        "siteIds": ["site-ci"],
        "actions": [action],
        "requestId": request_id,
        "datasetVersion": "clip-ci",
        "demoData": True,
        "exp": int(time.time()) + 300,
    }
    token = internal_auth.sign_internal_token(claims)
    return {"Authorization": f"Bearer {token}", "X-Request-Id": request_id}


def assert_no_raw_embeddings(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_VECTOR_KEYS:
                fail(f"raw embedding key '{key}' present at {path}")
            assert_no_raw_embeddings(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_no_raw_embeddings(nested, f"{path}[{index}]")


def get_json(path: str, expected_status: int = 200) -> dict[str, Any]:
    response = httpx.get(f"{BASE_URL}{path}", timeout=30.0)
    if response.status_code != expected_status:
        fail(f"GET {path} returned {response.status_code}: {response.text[:500]}")
    payload = response.json()
    if not isinstance(payload, dict):
        fail(f"GET {path} did not return a JSON object")
    return payload


def analyze_image(image: Image.Image, request_id: str) -> dict[str, Any]:
    response = httpx.post(
        f"{BASE_URL}/analyze-image",
        headers=service_headers("match:execute", request_id),
        files={"file": (f"{request_id}.png", png_bytes(image), "image/png")},
        timeout=120.0,
    )
    if response.status_code != 200:
        fail(f"POST /analyze-image {request_id} returned {response.status_code}: {response.text[:800]}")
    payload = response.json()
    assert_no_raw_embeddings(payload)
    return payload


def check_live() -> None:
    payload = get_json("/health/live")
    if payload.get("status") != "ok":
        fail(f"/health/live status={payload.get('status')!r}")
    print("SMOKE_OK /health/live")


def check_ready() -> None:
    payload = get_json("/health/ready")
    if payload.get("status") != "ready":
        fail(f"/health/ready not ready: {json.dumps(payload.get('checks'))}")
    checks = payload.get("checks") or {}
    if checks.get("databaseReady") is not True:
        fail(f"/health/ready databaseReady={checks.get('databaseReady')!r}")
    if checks.get("ocrDependencyReady") is not True:
        fail(f"/health/ready ocrDependencyReady={checks.get('ocrDependencyReady')!r}")
    if checks.get("warmupCompleted") is not True:
        fail("CLIP warmup did not complete")
    print("SMOKE_OK /health/ready mongodb+ocr+warmup")


def check_english_ocr() -> None:
    result = extract_ocr(text_image("DUBAI FESTIVAL"), lang="eng", psm=6)
    text = (result.get("fullText") or "").upper()
    if "DUBAI" not in text or "FESTIVAL" not in text:
        fail(f"English OCR missed expected tokens: {result.get('fullText')!r}")
    payload = analyze_image(text_image("DUBAI FESTIVAL"), "ci-ocr-en")
    http_text = ((payload.get("ocr") or {}).get("fullText") or "").upper()
    if "DUBAI" not in http_text:
        fail(f"HTTP English OCR missed DUBAI: {http_text!r} error={payload.get('ocrError')!r}")
    print("SMOKE_OK english_ocr")


def check_arabic_ocr() -> None:
    result = extract_ocr(text_image("مهرجان دبي", direction="rtl"), lang="ara", psm=6)
    normalized = (result.get("fullText") or "").replace(" ", "")
    if "دبي" not in normalized:
        fail(f"Arabic OCR missed expected token: {result.get('fullText')!r}")
    payload = analyze_image(text_image("مهرجان دبي", direction="rtl"), "ci-ocr-ar")
    http_text = ((payload.get("ocr") or {}).get("fullText") or "").replace(" ", "")
    if "دبي" not in http_text:
        fail(f"HTTP Arabic OCR missed دبي: {http_text!r} error={payload.get('ocrError')!r}")
    print("SMOKE_OK arabic_ocr")


def check_barcode_fixture() -> None:
    image = barcode_image()
    scanned = scan_barcodes(image)
    values = [entry.get("text") for entry in scanned.get("barcodes") or []]
    if BARCODE_VALUE not in values:
        fail(f"barcode fixture scan missed {BARCODE_VALUE}: {values!r}")
    payload = analyze_image(image, "ci-barcode")
    http_values = [entry.get("text") for entry in ((payload.get("barcode") or {}).get("barcodes") or [])]
    if BARCODE_VALUE not in http_values:
        fail(f"HTTP barcode scan missed {BARCODE_VALUE}: {http_values!r} error={payload.get('barcodeError')!r}")
    print("SMOKE_OK barcode_fixture")


def check_protected_responses_hide_embeddings() -> None:
    image = text_image("DUBAI FESTIVAL")
    created = httpx.post(
        f"{BASE_URL}/items",
        headers=service_headers("corpus:write", "ci-item-create"),
        data={"id": "ci-item-1", "title": "CI OCR item", "description": "Festival CI item"},
        files={"file": ("item.png", png_bytes(image), "image/png")},
        timeout=120.0,
    )
    if created.status_code != 200:
        fail(f"POST /items returned {created.status_code}: {created.text[:800]}")
    created_payload = created.json()
    assert_no_raw_embeddings(created_payload)

    fetched = httpx.get(
        f"{BASE_URL}/items/ci-item-1",
        headers=service_headers("corpus:read", "ci-item-read"),
        timeout=30.0,
    )
    if fetched.status_code != 200:
        fail(f"GET /items/ci-item-1 returned {fetched.status_code}: {fetched.text[:800]}")
    fetched_payload = fetched.json()
    assert_no_raw_embeddings(fetched_payload)

    encoded = httpx.post(
        f"{BASE_URL}/encode-image",
        headers=service_headers("match:execute", "ci-encode-image"),
        files={"file": ("encode.png", png_bytes(image), "image/png")},
        timeout=120.0,
    )
    if encoded.status_code != 200:
        fail(f"POST /encode-image returned {encoded.status_code}: {encoded.text[:800]}")
    encoded_payload = encoded.json()
    assert_no_raw_embeddings(encoded_payload)
    print("SMOKE_OK protected_responses_hide_embeddings")


def main() -> None:
    if len((os.environ.get("INTERNAL_JWT_SECRET") or "").encode("utf-8")) < 32:
        fail("INTERNAL_JWT_SECRET must contain at least 32 bytes")
    if not FONT_PATH.is_file():
        fail(f"bundled OCR font missing at {FONT_PATH}")
    check_live()
    check_ready()
    check_english_ocr()
    check_arabic_ocr()
    check_barcode_fixture()
    check_protected_responses_hide_embeddings()
    print("SMOKE_OK live ready english_ocr arabic_ocr barcode embeddings")


if __name__ == "__main__":
    main()
