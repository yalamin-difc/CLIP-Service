from typing import Optional

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
import uuid
from datetime import datetime, timezone

import hashlib
import time
import os
import anyio

from barcode_service import scan_barcodes
from match_service import build_explanation, cosine_similarity, should_return_no_match, softmax_confidences
from ocr_service import extract_ocr

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = Gauge = Histogram = None
    generate_latest = None

app = FastAPI(title="CLIP Service")

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://ailostfound.al-amentech.io",
    "https://openailostfound-fccdb129f869.herokuapp.com",
    "http://localhost:3000",
]
ALLOW_ORIGIN_REGEX = r"^https:\/\/ai-lost-and-found-ver-2-.*\.vercel\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Environment / Auth
# ---------------------------------------------------------
ENV = os.environ.get("ENV", os.environ.get("NODE_ENV", "dev")).strip().lower() or "dev"
IS_PRODUCTION = ENV in {"prod", "production"}

CLIP_API_KEY = os.environ.get("CLIP_API_KEY", "").strip()
METRICS_API_KEY = os.environ.get("METRICS_API_KEY", "").strip()

# Confidence calibration / decisioning controls
CONF_TEMPERATURE = float(os.environ.get("CONF_TEMPERATURE", "0.07"))
CONF_MIN_SCORE = float(os.environ.get("CONF_MIN_SCORE", "0.22"))
CONF_MIN_MARGIN = float(os.environ.get("CONF_MIN_MARGIN", "0.03"))

# Service/model governance metadata
SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "dev").strip() or "dev"
MODEL_ID = os.environ.get("MODEL_ID", "openai/clip-vit-base-patch32").strip() or "openai/clip-vit-base-patch32"

# Upload validation (defense-in-depth)
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10MB
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(20_000_000)))  # e.g. 4000x5000
MAX_IMAGE_DIM = int(os.environ.get("MAX_IMAGE_DIM", "6000"))

OCR_TIMEOUT_MS = int(os.environ.get("OCR_TIMEOUT_MS", "4000"))
BARCODE_TIMEOUT_MS = int(os.environ.get("BARCODE_TIMEOUT_MS", "2500"))

_ALLOWED_ITEM_STATUSES = {"draft", "released", "archived"}

# ---------------------------------------------------------
# Metrics (Prometheus)
# ---------------------------------------------------------
if Counter is not None:
    METRIC_REQUESTS = Counter("clip_service_requests_total", "Requests", ["endpoint", "status"])
    METRIC_LATENCY = Histogram("clip_service_request_latency_seconds", "Request latency", ["endpoint"])
    METRIC_MODEL_LOADED = Gauge("clip_service_model_loaded", "Model loaded (1/0)")
    METRIC_OCR_OK = Counter("clip_service_ocr_ok_total", "OCR successes", ["lang"])
    METRIC_OCR_FAIL = Counter("clip_service_ocr_fail_total", "OCR failures", ["lang"])
    METRIC_BARCODE_OK = Counter("clip_service_barcode_ok_total", "Barcode scan successes")
    METRIC_BARCODE_FAIL = Counter("clip_service_barcode_fail_total", "Barcode scan failures")
    METRIC_MATCH_NO_MATCH = Counter("clip_service_match_no_match_total", "No-match decisions")
else:
    METRIC_REQUESTS = METRIC_LATENCY = METRIC_MODEL_LOADED = None
    METRIC_OCR_OK = METRIC_OCR_FAIL = None
    METRIC_BARCODE_OK = METRIC_BARCODE_FAIL = None
    METRIC_MATCH_NO_MATCH = None


def _get_request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    return (rid or "").strip() or str(uuid.uuid4())


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = (request.headers.get("X-Request-Id") or "").strip() or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    try:
        response.headers["X-Request-Id"] = rid
    except Exception:
        pass
    return response


def _error_payload(*, code: str, message: str, request_id: str, details: Optional[dict] = None) -> dict:
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "requestId": request_id,
            **({"details": details} if details else {}),
        },
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    rid = _get_request_id(request)
    detail = exc.detail
    if isinstance(detail, dict):
        code = str(detail.get("code") or f"HTTP_{exc.status_code}")
        message = str(detail.get("message") or detail.get("detail") or "Request failed")
        details = detail.get("details") if isinstance(detail.get("details"), dict) else None
    else:
        code = f"HTTP_{exc.status_code}"
        message = str(detail) if detail else "Request failed"
        details = None
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(code=code, message=message, request_id=rid, details=details),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    rid = _get_request_id(request)
    return JSONResponse(
        status_code=422,
        content=_error_payload(
            code="VALIDATION_ERROR",
            message="Validation failed",
            request_id=rid,
            details={"errors": exc.errors()},
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = _get_request_id(request)
    return JSONResponse(status_code=500, content=_error_payload(code="INTERNAL_ERROR", message="Server error", request_id=rid))


def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    # In production, auth must be enforced even if CLIP_API_KEY is missing.
    if IS_PRODUCTION and not CLIP_API_KEY:
        raise HTTPException(status_code=503, detail="CLIP auth misconfigured (missing CLIP_API_KEY)")
    if not CLIP_API_KEY:
        return  # auth disabled (dev)
    if authorization != f"Bearer {CLIP_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def require_non_production(feature: str) -> None:
    if IS_PRODUCTION:
        raise HTTPException(status_code=404, detail=f"{feature} is not available in production")


# ---------------------------------------------------------
# LAZY LOADING
# ---------------------------------------------------------
model = None
processor = None


def load_model():
    global model, processor
    if model is None:
        print("⏳ Loading CLIP model on demand...")
        model = CLIPModel.from_pretrained(MODEL_ID)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        print("✅ CLIP model ready!")
    return model, processor


@app.get("/metrics")
def metrics(request: Request, authorization: Optional[str] = Header(default=None)):
    if IS_PRODUCTION:
        if METRICS_API_KEY:
            if authorization != f"Bearer {METRICS_API_KEY}":
                raise HTTPException(status_code=401, detail="Unauthorized")
        else:
            require_auth(authorization)

    if generate_latest is None:
        raise HTTPException(status_code=503, detail={"code": "METRICS_UNAVAILABLE", "message": "Metrics not available"})
    if METRIC_MODEL_LOADED is not None:
        METRIC_MODEL_LOADED.set(1 if model is not None else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _governance_meta() -> dict:
    return {
        "serviceVersion": SERVICE_VERSION,
        "modelId": MODEL_ID,
        "confidence": {"temperature": CONF_TEMPERATURE, "minScore": CONF_MIN_SCORE, "minMargin": CONF_MIN_MARGIN},
    }


async def _read_upload_bytes(upload: UploadFile) -> bytes:
    ct = (upload.content_type or "").lower().strip()
    if ct and not ct.startswith("image/"):
        raise HTTPException(status_code=400, detail={"code": "BAD_CONTENT_TYPE", "message": "Invalid content-type for image upload"})
    raw = await upload.read()
    if raw is None:
        raise HTTPException(status_code=400, detail={"code": "READ_FAILED", "message": "Failed to read upload"})
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail={"code": "UPLOAD_TOO_LARGE", "message": "Upload too large"})
    return raw


def _validate_pil_image(pil: Image.Image) -> None:
    w, h = pil.size
    if w <= 0 or h <= 0:
        raise HTTPException(status_code=400, detail={"code": "BAD_DIMENSIONS", "message": "Invalid image dimensions"})
    if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
        raise HTTPException(status_code=413, detail={"code": "IMAGE_TOO_LARGE", "message": "Image dimensions too large"})
    if (w * h) > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail={"code": "IMAGE_TOO_LARGE", "message": "Image too large"})


async def _upload_to_pil(upload: UploadFile) -> Image.Image:
    raw = await _read_upload_bytes(upload)
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.verify()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        _validate_pil_image(pil)
        return pil
    except Exception as e:
        raise HTTPException(status_code=400, detail={"code": "INVALID_IMAGE", "message": "Invalid image upload"}) from e


def _redact_ocr_payload(ocr: Optional[dict]) -> dict:
    if not isinstance(ocr, dict):
        return {"present": False}
    full_text = ocr.get("fullText") if isinstance(ocr.get("fullText"), str) else ""
    words = ocr.get("words") if isinstance(ocr.get("words"), list) else []
    return {
        "present": True,
        "wordCount": len(words),
        "fullTextSha256": _sha256_hex(full_text.encode("utf-8")) if full_text else None,
        "meta": ocr.get("meta") if isinstance(ocr.get("meta"), dict) else None,
    }


def _redact_barcode_payload(barcode: Optional[dict]) -> dict:
    if not isinstance(barcode, dict):
        return {"present": False}
    barcodes = barcode.get("barcodes") if isinstance(barcode.get("barcodes"), list) else []
    texts = []
    for b in barcodes:
        if isinstance(b, dict) and isinstance(b.get("text"), str):
            texts.append(b["text"])
    joined = "\n".join(texts)
    return {
        "present": True,
        "count": len(barcodes),
        "textsSha256": _sha256_hex(joined.encode("utf-8")) if joined else None,
        "meta": barcode.get("meta") if isinstance(barcode.get("meta"), dict) else None,
    }


def _normalize_embedding(emb: torch.Tensor) -> torch.Tensor:
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb


# ---------------------------------------------------------
# Health / Readiness
# ---------------------------------------------------------
@app.get("/")
def home(request: Request):
    return {"requestId": _get_request_id(request), "status": "running", "model_loaded": model is not None, "env": ENV}


@app.get("/health")
def health(request: Request):
    return {
        "requestId": _get_request_id(request),
        "status": "running",
        "env": ENV,
        "model_loaded": model is not None,
        "ready": model is not None,
        "governance": _governance_meta(),
    }


@app.get("/ready")
def ready(request: Request, authorization: Optional[str] = Header(default=None)):
    if IS_PRODUCTION:
        require_auth(authorization)
    try:
        load_model()
        return {"requestId": _get_request_id(request), "ready": True, "model_loaded": True, "env": ENV}
    except Exception as e:
        raise HTTPException(status_code=503, detail={"code": "NOT_READY", "message": f"Model not ready: {e}"}) from e


# ---------------------------------------------------------
# UI (disabled in prod)
# ---------------------------------------------------------
UI_HTML = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CLIP Service UI</title>
  </head>
  <body>
    <h1>CLIP Service UI</h1>
    <p>UI disabled in production.</p>
  </body>
</html>
"""


@app.get("/ui", response_class=HTMLResponse)
def ui():
    require_non_production("UI")
    return HTMLResponse(UI_HTML)


# ---------------------------------------------------------
# Encode Image / Text
# ---------------------------------------------------------
@app.post("/encode-image")
async def encode_image(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    model, processor = load_model()

    upload = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail={"code": "MISSING_IMAGE", "message": "No file uploaded. Use form field 'file' (or 'image')."})

    pil_image = await _upload_to_pil(upload)
    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    embedding = _normalize_embedding(embedding)
    return {"requestId": _get_request_id(request), "governance": _governance_meta(), "embedding": embedding.squeeze().tolist()}


@app.post("/encode-text")
async def encode_text(
    request: Request,
    text: Optional[str] = Form(default=None),
    queryText: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    model, processor = load_model()

    value = (text or queryText or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail={"code": "MISSING_TEXT", "message": "No text provided. Use form field 'text' (or 'queryText')."})

    inputs = processor(text=[value], return_tensors="pt", padding=True)

    with torch.no_grad():
        embedding = model.get_text_features(**inputs)

    embedding = _normalize_embedding(embedding)
    return {"requestId": _get_request_id(request), "governance": _governance_meta(), "embedding": embedding.squeeze().tolist()}


# ---------------------------------------------------------
# Analyze Image (CLIP + OCR + Barcode)
# ---------------------------------------------------------
@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    doOcr: bool = Form(default=True),
    doBarcode: bool = Form(default=True),
    ocrLang: str = Form(default="eng"),
    ocrPsm: int = Form(default=6),
    authorization: Optional[str] = Header(default=None),
):
    t0 = time.time()
    endpoint = "/analyze-image"

    require_auth(authorization)
    model, processor = load_model()

    upload = file or image
    if upload is None:
        if METRIC_REQUESTS is not None:
            METRIC_REQUESTS.labels(endpoint=endpoint, status="400").inc()
        raise HTTPException(status_code=400, detail={"code": "MISSING_IMAGE", "message": "No file uploaded. Use form field 'file' (or 'image')."})

    request_id = _get_request_id(request)

    raw = await _read_upload_bytes(upload)
    pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    _validate_pil_image(pil_image)

    input_hash = _sha256_hex(raw)
    input_meta = {"sha256": input_hash, "bytes": len(raw), "contentType": upload.content_type}

    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    embedding = _normalize_embedding(embedding).squeeze().tolist()

    ocr = None
    ocr_error = None
    ocr_error_code = None
    if doOcr:
        try:
            timeout_s = max(0.1, float(OCR_TIMEOUT_MS) / 1000.0)
            with anyio.fail_after(timeout_s):
                ocr = await anyio.to_thread.run_sync(extract_ocr, pil_image, lang=ocrLang, psm=int(ocrPsm))
            if METRIC_OCR_OK is not None:
                METRIC_OCR_OK.labels(lang=str(ocrLang)).inc()
        except TimeoutError:
            ocr_error = "OCR timed out"
            ocr_error_code = "OCR_TIMEOUT"
            if METRIC_OCR_FAIL is not None:
                METRIC_OCR_FAIL.labels(lang=str(ocrLang)).inc()
        except Exception as e:
            ocr_error = str(e)
            msg = (ocr_error or "").lower()
            ocr_error_code = "OCR_DEPENDENCY_MISSING" if ("tesseract" in msg or "pytesseract" in msg or "not installed" in msg or "binary not found" in msg) else "OCR_FAILED"
            if METRIC_OCR_FAIL is not None:
                METRIC_OCR_FAIL.labels(lang=str(ocrLang)).inc()

    barcode = None
    barcode_error = None
    barcode_error_code = None
    if doBarcode:
        try:
            timeout_s = max(0.1, float(BARCODE_TIMEOUT_MS) / 1000.0)
            with anyio.fail_after(timeout_s):
                barcode = await anyio.to_thread.run_sync(scan_barcodes, pil_image)
            if METRIC_BARCODE_OK is not None:
                METRIC_BARCODE_OK.inc()
        except TimeoutError:
            barcode_error = "Barcode scan timed out"
            barcode_error_code = "BARCODE_TIMEOUT"
            if METRIC_BARCODE_FAIL is not None:
                METRIC_BARCODE_FAIL.inc()
        except Exception as e:
            barcode_error = str(e)
            msg = (barcode_error or "").lower()
            barcode_error_code = "BARCODE_DEPENDENCY_MISSING" if ("zxing" in msg or "zxingcpp" in msg or "not installed" in msg) else "BARCODE_FAILED"
            if METRIC_BARCODE_FAIL is not None:
                METRIC_BARCODE_FAIL.inc()

    if METRIC_LATENCY is not None:
        METRIC_LATENCY.labels(endpoint=endpoint).observe(max(0.0, time.time() - t0))
    if METRIC_REQUESTS is not None:
        METRIC_REQUESTS.labels(endpoint=endpoint, status="200").inc()

    return {
        "requestId": request_id,
        "governance": _governance_meta(),
        "input": input_meta,
        "embedding": embedding,
        "ocr": ocr,
        "ocrError": ocr_error,
        "ocrErrorCode": ocr_error_code,
        "barcode": barcode,
        "barcodeError": barcode_error,
        "barcodeErrorCode": barcode_error_code,
    }


# ---------------------------------------------------------
# Disabled in prod: /items*, /match, /audit-logs
# ---------------------------------------------------------
@app.post("/items", status_code=201)
async def create_item(payload: dict = Body(...), authorization: Optional[str] = Header(default=None)):
    require_non_production("Item persistence")
    require_auth(authorization)
    return {"ok": True}


@app.get("/items")
def list_items(authorization: Optional[str] = Header(default=None)):
    require_non_production("Item persistence")
    require_auth(authorization)
    return {"items": []}


@app.get("/items/{item_id}")
def get_item(item_id: str, authorization: Optional[str] = Header(default=None)):
    require_non_production("Item persistence")
    require_auth(authorization)
    return {"item": {"id": item_id}}


@app.post("/items/{item_id}/release")
def release_item(item_id: str, authorization: Optional[str] = Header(default=None)):
    require_non_production("Item persistence")
    require_auth(authorization)
    return {"item": {"id": item_id, "status": "released"}}


@app.get("/audit-logs")
def list_audit_logs(authorization: Optional[str] = Header(default=None)):
    require_non_production("Audit logs")
    require_auth(authorization)
    return {"logs": []}


@app.post("/match")
async def match_top_k(authorization: Optional[str] = Header(default=None)):
    require_non_production("Match endpoint")
    require_auth(authorization)
    return {"topK": []}
