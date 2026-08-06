from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import threading
import time
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Mapping, Optional

import anyio
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from internal_auth import ServiceIdentity, require_identity, validate_auth_configuration

try:  # pragma: no cover - optional runtime dependency
    from barcode_service import scan_barcodes as _scan_barcodes
except Exception:  # pragma: no cover
    _scan_barcodes = None

try:  # pragma: no cover - optional runtime dependency
    from ocr_service import extract_ocr as _extract_ocr, ocr_dependency_ready
except Exception:  # pragma: no cover
    _extract_ocr = None

    def ocr_dependency_ready(required_languages: str = "eng+ara") -> bool:
        return False

try:  # pragma: no cover - optional runtime dependency
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = Gauge = Histogram = None
    generate_latest = None

app = FastAPI(title="CLIP Service")
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = [
    "https://ailostfound.al-amentech.io",
    "https://openailostfound-fccdb129f869.herokuapp.com",
    "http://localhost:3000",
]
ALLOW_ORIGIN_REGEX = r"^https:\/\/ai-lost-and-found-ver-2-.*\.vercel\.app$"
REQUEST_ID_HEADER = "X-Request-Id"
MODEL_NAME = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "dev").strip() or "dev"
ENVIRONMENT = os.environ.get("ENV", os.environ.get("NODE_ENV", "dev")).strip().lower() or "dev"
STORAGE_MODE = os.environ.get("STORAGE_MODE", "").strip().lower()
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "cpu").strip().lower()
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
MONGODB_URI = (os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI") or "").strip()
MONGODB_DB = os.environ.get("MONGODB_DB", "clip_service").strip() or "clip_service"
MONGODB_ITEMS_COLLECTION = os.environ.get("MONGODB_COLLECTION", "items").strip() or "items"
MONGODB_AUDIT_COLLECTION = os.environ.get("MONGODB_AUDIT_COLLECTION", "audit_logs").strip() or "audit_logs"
CONF_TEMPERATURE = float(os.environ.get("CONF_TEMPERATURE", "0.07"))
CONF_MIN_SCORE = float(os.environ.get("CONF_MIN_SCORE", "0.22"))
CONF_MIN_MARGIN = float(os.environ.get("CONF_MIN_MARGIN", "0.03"))
OCR_TIMEOUT_MS = int(os.environ.get("OCR_TIMEOUT_MS", "4000"))
BARCODE_TIMEOUT_MS = int(os.environ.get("BARCODE_TIMEOUT_MS", "2500"))
REQUEST_TIMEOUT_MS = int(os.environ.get("REQUEST_TIMEOUT_MS", "15000"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", "25000000"))
REQUEST_CONCURRENCY = int(os.environ.get("REQUEST_CONCURRENCY", "32"))
OCR_CONCURRENCY = int(os.environ.get("OCR_CONCURRENCY", "2"))
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
OCR_ENABLED = os.environ.get("OCR_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "eng+ara").strip() or "eng+ara"
OCR_PSM = int(os.environ.get("OCR_PSM", "6"))
APPROVED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
SCORING_VERSION = "clip-match-v1"
CORPUS_LIMIT = int(os.environ.get("CORPUS_LIMIT", "50000"))
EXPECTED_EMBEDDING_DIMENSION = int(os.environ.get("EXPECTED_EMBEDDING_DIMENSION", "512"))

request_limiter = anyio.CapacityLimiter(max(1, REQUEST_CONCURRENCY))
ocr_limiter = anyio.CapacityLimiter(max(1, OCR_CONCURRENCY))
rate_limit_lock = threading.Lock()
rate_limit_windows: Dict[str, tuple[int, int]] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if Counter is not None:
    METRIC_REQUESTS = Counter("clip_service_requests_total", "Requests", ["endpoint", "status"])
    METRIC_LATENCY = Histogram("clip_service_request_latency_seconds", "Request latency", ["endpoint"])
    METRIC_MODEL_LOADED = Gauge("clip_service_model_loaded", "Model loaded (1/0)")
    METRIC_MATCH_RESULTS = Counter("clip_service_match_results_total", "Match outcomes", ["outcome"])
    METRIC_MATCH_FAILURES = Counter("clip_service_match_failures_total", "Match failures", ["code"])
else:  # pragma: no cover
    METRIC_REQUESTS = None
    METRIC_LATENCY = None
    METRIC_MODEL_LOADED = None
    METRIC_MATCH_RESULTS = None
    METRIC_MATCH_FAILURES = None


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ThresholdsModel(StrictBaseModel):
    minScore: float = Field(default=CONF_MIN_SCORE, ge=-1.0, le=1.0)
    minMargin: float = Field(default=CONF_MIN_MARGIN, ge=0.0, le=2.0)
    temperature: float = Field(default=CONF_TEMPERATURE, gt=0.0, le=100.0)


class MatchRequestModel(StrictBaseModel):
    text: Optional[str] = None
    topK: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    status: str = "released"
    thresholds: ThresholdsModel = Field(default_factory=ThresholdsModel)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ocrText: Optional[str] = None
    barcodeValues: List[str] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)
    doOcr: bool = True
    doBarcode: bool = True
    ocrLang: str = "eng"
    ocrPsm: int = Field(default=6, ge=1, le=13)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = (value or "released").strip().lower()
        if normalized not in {"released", "stored", "draft", "archived"}:
            raise ValueError("status must be one of: released, stored, draft, archived")
        return normalized


class QueryImageModel(StrictBaseModel):
    filename: Optional[str] = None
    width: int
    height: int
    mode: str


class QueryBarcodeModel(StrictBaseModel):
    barcodes: List[Dict[str, Any]] = Field(default_factory=list)
    meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class QueryOcrModel(StrictBaseModel):
    fullText: Optional[str] = None
    words: List[Dict[str, Any]] = Field(default_factory=list)
    meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class QuerySignalsModel(StrictBaseModel):
    ocr: Optional[Dict[str, Any]] = None
    barcode: Optional[Dict[str, Any]] = None
    labels: List[str] = Field(default_factory=list)


class MatchQueryModel(StrictBaseModel):
    type: str
    modalities: List[str] = Field(default_factory=list)
    text: Optional[str] = None
    image: Optional[QueryImageModel] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    thresholds: ThresholdsModel
    ocrText: Optional[str] = None
    ocr: QueryOcrModel = Field(default_factory=QueryOcrModel)
    barcodeValues: List[str] = Field(default_factory=list)
    barcode: QueryBarcodeModel = Field(default_factory=QueryBarcodeModel)
    labels: List[str] = Field(default_factory=list)
    signals: QuerySignalsModel


class MatchDecisionModel(StrictBaseModel):
    noMatch: bool
    reason: str
    details: Dict[str, Any] = Field(default_factory=dict)


class MatchGovernanceModel(StrictBaseModel):
    engine: str = "clip"
    serviceVersion: str
    modelId: str
    confidence: Dict[str, float]
    scoringVersion: str
    thresholds: ThresholdsModel
    releasedOnly: bool
    topKRequested: int
    candidatesEvaluated: int
    latencyMs: int
    decision: MatchDecisionModel


class MatchExplanationSimilarityModel(StrictBaseModel):
    cosine: float
    band: str


class MatchExplanationOverlapModel(StrictBaseModel):
    matchedTerms: List[str] = Field(default_factory=list)
    matchCount: int = 0


class MatchExplanationValueMatchModel(StrictBaseModel):
    matchedValues: List[str] = Field(default_factory=list)
    matchCount: int = 0


class MatchModelMetadataModel(StrictBaseModel):
    engine: str = "clip"
    modelId: str
    serviceVersion: str
    scoringVersion: str


class MatchExplanationModel(StrictBaseModel):
    reason: str
    summary: str
    scoreBand: str
    signals: Dict[str, Any] = Field(default_factory=dict)
    similarity: MatchExplanationSimilarityModel
    ocr: MatchExplanationOverlapModel
    barcode: MatchExplanationValueMatchModel
    labels: MatchExplanationValueMatchModel
    model: MatchModelMetadataModel


class MatchItemModel(StrictBaseModel):
    id: str
    title: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    released: bool = False
    eligibleForMatching: bool = False
    ocrText: Optional[str] = None
    ocr: Optional[Dict[str, Any]] = None
    ocrWords: List[Dict[str, Any]] = Field(default_factory=list)
    barcodes: List[Dict[str, Any]] = Field(default_factory=list)
    barcodeValues: List[str] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    image: Optional[Dict[str, Any]] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    releasedAt: Optional[str] = None
    tenantId: str
    siteId: str
    datasetVersion: str
    demoData: bool
    createdBy: str
    modelId: str
    modelRevision: str
    embeddingDimension: int


class MatchCandidateModel(StrictBaseModel):
    candidateId: str
    item: MatchItemModel
    score: float
    confidence: float
    explanation: MatchExplanationModel


class MatchResponseModel(StrictBaseModel):
    requestId: str
    governance: MatchGovernanceModel
    query: MatchQueryModel
    decision: MatchDecisionModel
    topK: List[MatchCandidateModel] = Field(default_factory=list)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def get_request_id(request: Request) -> Optional[str]:
    return getattr(request.state, "request_id", None)


def with_request_id(request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = dict(payload)
    body.setdefault("requestId", get_request_id(request))
    return body


def problem_detail(code: str, message: str, details: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    return payload


def default_error_code(status_code: int) -> str:
    return {
        400: "invalid_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        422: "validation_error",
        500: "internal_error",
        503: "service_unavailable",
    }.get(status_code, "http_error")


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
) -> JSONResponse:
    request_id = get_request_id(request)
    payload: Dict[str, Any] = {
        "success": False,
        "error": {
            "status": status_code,
            "code": code,
            "message": message,
            "requestId": request_id,
        },
    }
    if details is not None:
        payload["error"]["details"] = details
    headers = {REQUEST_ID_HEADER: request_id} if request_id else None
    return JSONResponse(status_code=status_code, content=payload, headers=headers)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get(REQUEST_ID_HEADER, "").strip() or str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.request_started_at = time.time()
    if request.url.path not in {"/health/live", "/health/ready"}:
        client_key = request.client.host if request.client else "unknown"
        window = int(time.time() // 60)
        with rate_limit_lock:
            stored_window, count = rate_limit_windows.get(client_key, (window, 0))
            if stored_window != window:
                stored_window, count = window, 0
            count += 1
            rate_limit_windows[client_key] = (stored_window, count)
        if count > RATE_LIMIT_PER_MINUTE:
            return error_response(request, 429, "rate_limit_exceeded", "Request rate limit exceeded.")
    try:
        async with request_limiter:
            with anyio.fail_after(max(0.1, REQUEST_TIMEOUT_MS / 1000.0)):
                response = await call_next(request)
    except TimeoutError:
        return error_response(request, 504, "request_timeout", "Request processing timed out.")
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    details = None
    code = default_error_code(exc.status_code)
    message = str(exc.detail) if exc.detail is not None else "Request failed."
    if isinstance(exc.detail, dict):
        code = str(exc.detail.get("code") or code)
        message = str(exc.detail.get("message") or message)
        details = exc.detail.get("details")
    elif isinstance(exc.detail, list):
        message = "Request failed."
        details = exc.detail
    record_request_metric(request.url.path, exc.status_code)
    if request.url.path == "/match":
        logger.warning("Match request failed [%s]: %s", code, message)
        record_match_failure(code)
    return error_response(request, exc.status_code, code, message, details)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    record_request_metric(request.url.path, 422)
    if request.url.path == "/match":
        record_match_failure("validation_error")
    return error_response(request, 422, "validation_error", "Request validation failed.", exc.errors())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    record_request_metric(request.url.path, 500)
    logger.exception("Unhandled error on %s", request.url.path)
    if request.url.path == "/match":
        record_match_failure("internal_error")
    return error_response(request, 500, "internal_error", "An unexpected error occurred.")


def record_request_metric(endpoint: str, status: int) -> None:
    if METRIC_REQUESTS is not None:
        METRIC_REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()


def record_latency_metric(endpoint: str, started_at: float) -> None:
    if METRIC_LATENCY is not None:
        METRIC_LATENCY.labels(endpoint=endpoint).observe(max(0.0, time.time() - started_at))


def record_match_outcome(outcome: str) -> None:
    if METRIC_MATCH_RESULTS is not None:
        METRIC_MATCH_RESULTS.labels(outcome=outcome).inc()


def record_match_failure(code: str) -> None:
    if METRIC_MATCH_FAILURES is not None:
        METRIC_MATCH_FAILURES.labels(code=code).inc()


def request_latency_ms(request: Request) -> int:
    started_at = getattr(request.state, "request_started_at", None)
    if started_at is None:
        return 0
    return int(max(0.0, (time.time() - started_at) * 1000))


def governance_meta(**extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "engine": "clip",
        "serviceVersion": SERVICE_VERSION,
        "modelId": MODEL_NAME,
        "confidence": {
            "temperature": CONF_TEMPERATURE,
            "minScore": CONF_MIN_SCORE,
            "minMargin": CONF_MIN_MARGIN,
        },
    }
    payload.update(extra)
    return payload


def authorize(request: Request, action: str) -> ServiceIdentity:
    identity = require_identity(request, action)
    request.state.identity = identity
    return identity


def compact_text(value: Optional[str], limit: int = 160) -> Optional[str]:
    if value is None:
        return None
    collapsed = " ".join(str(value).split())
    if not collapsed:
        return None
    return collapsed[: limit - 3] + "..." if len(collapsed) > limit else collapsed


def is_upload(value: Any) -> bool:
    return hasattr(value, "filename") and hasattr(value, "read")


def first_value(container: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = container.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise HTTPException(status_code=400, detail=problem_detail("invalid_boolean", "Boolean field contains an invalid value."))


def normalize_string_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    raw_items: List[Any]
    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                raw_items = parsed
            else:
                raw_items = re.split(r"[\n,]+", stripped)
        else:
            raw_items = re.split(r"[\n,]+", stripped)
    else:
        raw_items = [value]

    normalized: List[str] = []
    seen = set()
    for item in raw_items:
        text = compact_text(str(item), limit=120)
        key = (text or "").lower()
        if text and key not in seen:
            normalized.append(text)
            seen.add(key)
    return normalized


def normalize_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=problem_detail("invalid_json", f"Field '{field_name}' must contain valid JSON."),
            ) from exc
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise HTTPException(status_code=400, detail=problem_detail("invalid_payload", f"Field '{field_name}' must be an object."))


def parse_optional_object(value: Any, field_name: str) -> Dict[str, Any]:
    if value in (None, ""):
        return {}
    return normalize_mapping(value, field_name)


def parse_top_k(value: Any) -> int:
    if value is None or value == "":
        return DEFAULT_TOP_K
    try:
        top_k = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=problem_detail("invalid_top_k", "Field 'topK' must be an integer.")) from exc
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("invalid_top_k", f"Field 'topK' must be between 1 and {MAX_TOP_K}."),
        )
    return top_k


def tokenize(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return sorted({token.lower() for token in re.findall(r"[A-Za-z0-9]{3,}", value)})


def normalize_barcodes(value: Any) -> List[Dict[str, Any]]:
    if value is None or value == "":
        return []
    raw_items: List[Any]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            raw_items = parsed if isinstance(parsed, list) else re.split(r"[\n,]+", stripped)
        else:
            raw_items = re.split(r"[\n,]+", stripped)
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raw_items = [value]

    normalized: List[Dict[str, Any]] = []
    seen = set()
    for item in raw_items:
        if isinstance(item, Mapping):
            text = compact_text(str(first_value(item, "text", "value", "code") or ""), limit=160)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            barcode = dict(item)
            barcode["text"] = text
        else:
            text = compact_text(str(item), limit=160)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            barcode = {"text": text}
        seen.add(key)
        normalized.append(barcode)
    return normalized


def barcode_overlap_values(query_barcodes: List[Dict[str, Any]], item_barcodes: List[Dict[str, Any]]) -> List[str]:
    query_barcode_map = {value.lower(): value for value in barcode_values(query_barcodes)}
    item_barcode_map = {value.lower(): value for value in barcode_values(item_barcodes)}
    return [item_barcode_map[key] for key in sorted(query_barcode_map.keys() & item_barcode_map.keys())][:3]


def barcode_values(barcodes: List[Dict[str, Any]]) -> List[str]:
    return [barcode.get("text") for barcode in barcodes if barcode.get("text")]


def merge_barcodes(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for group in groups:
        for item in group or []:
            text = compact_text(str(item.get("text") or ""), limit=160)
            key = (text or "").lower()
            if text and key not in seen:
                normalized = dict(item)
                normalized["text"] = text
                merged.append(normalized)
                seen.add(key)
    return merged


def compact_signals(ocr_text: Optional[str], barcode_texts: List[str], labels: List[str]) -> Dict[str, Any]:
    signals: Dict[str, Any] = {}
    excerpt = compact_text(ocr_text, limit=160)
    if excerpt:
        signals["ocr"] = {"excerpt": excerpt}
    if barcode_texts:
        signals["barcode"] = {"count": len(barcode_texts), "values": barcode_texts[:3]}
    if labels:
        signals["labels"] = labels[:5]
    return signals


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def normalize_vectors(value: Any) -> np.ndarray:
    array = to_numpy(value)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("invalid_embedding", "Embedding must be a one- or two-dimensional numeric vector."),
        )
    if not np.isfinite(array).all():
        raise HTTPException(
            status_code=400,
            detail=problem_detail("invalid_embedding", "Embedding must contain only finite numeric values."),
        )
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise HTTPException(status_code=400, detail=problem_detail("invalid_embedding", "Embedding norm must be greater than zero."))
    return array / norms


def normalize_embedding(value: Any) -> Optional[List[float]]:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=problem_detail("invalid_embedding", "Embedding must be valid JSON.")) from exc
    return [float(component) for component in normalize_vectors(value)[0].tolist()]


def validate_stored_embedding(value: Any) -> List[float]:
    normalized = normalize_embedding(value)
    if normalized is None:
        raise HTTPException(status_code=409, detail=problem_detail("missing_embedding", "An internal embedding is required."))
    expected = embedding_dimension or EXPECTED_EMBEDDING_DIMENSION
    if len(normalized) != expected:
        raise HTTPException(
            status_code=409,
            detail=problem_detail(
                "embedding_dimension_mismatch",
                f"Embedding dimension {len(normalized)} is incompatible with expected dimension {expected}.",
            ),
        )
    norm = float(np.linalg.norm(np.asarray(normalized, dtype=float)))
    if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-6):
        raise HTTPException(status_code=409, detail=problem_detail("embedding_not_normalized", "Embedding is not normalized."))
    return normalized


@contextmanager
def inference_mode():
    try:  # pragma: no cover - import path tested via monkeypatching
        import torch
    except ImportError:  # pragma: no cover
        yield
        return
    with torch.no_grad():
        yield


model = None
processor = None
model_lock = threading.Lock()
embedding_dimension: Optional[int] = None
model_warmup_completed = False
model_load_error: Optional[str] = None


def configured_device() -> str:
    if INFERENCE_DEVICE not in {"cpu", "cuda"}:
        raise RuntimeError("INFERENCE_DEVICE must be explicitly set to 'cpu' or 'cuda'")
    if INFERENCE_DEVICE == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA was configured but PyTorch is unavailable") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was configured but no CUDA device is available")
    return INFERENCE_DEVICE


def _move_inputs_to_device(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    device = configured_device()
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _feature_tensor(value: Any) -> Any:
    """Normalize Transformers 4.x tensor and 5.x model-output return shapes."""
    pooled = getattr(value, "pooler_output", None)
    return pooled if pooled is not None else value


def load_model():
    global model, processor, model_load_error
    if model is None or processor is None:
        with model_lock:
            if model is None or processor is None:
                try:
                    from transformers import CLIPModel, CLIPProcessor
                except ImportError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=problem_detail("model_unavailable", "CLIP model dependencies are not installed."),
                    ) from exc
                try:
                    device = configured_device()
                    model = CLIPModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
                    processor = CLIPProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
                    if hasattr(model, "to"):
                        model = model.to(device)
                    if hasattr(model, "eval"):
                        model.eval()
                    model_load_error = None
                    logger.info(
                        json.dumps(
                            {
                                "event": "model_load",
                                "modelId": MODEL_NAME,
                                "modelRevision": MODEL_REVISION,
                                "device": device,
                            }
                        )
                    )
                except Exception as exc:
                    model_load_error = type(exc).__name__
                    logger.error(
                        json.dumps(
                            {
                                "event": "model_failure",
                                "modelId": MODEL_NAME,
                                "modelRevision": MODEL_REVISION,
                                "errorType": type(exc).__name__,
                            }
                        )
                    )
                    raise
    return model, processor


def model_health() -> Dict[str, Any]:
    return {
        "modelId": MODEL_NAME,
        "modelRevision": MODEL_REVISION,
        "loaded": model is not None and processor is not None,
        "embeddingDimension": embedding_dimension,
        "device": INFERENCE_DEVICE,
        "warmupCompleted": model_warmup_completed,
    }


async def read_image_upload(upload: Optional[UploadFile], field_name: str) -> tuple[bytes, Image.Image]:
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("missing_image", f"Missing image upload in field '{field_name}'."),
        )
    content_type = (upload.content_type or "").lower()
    if content_type not in APPROVED_IMAGE_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=problem_detail("unsupported_image_type", "Image MIME type is not approved."),
        )
    try:
        payload = await upload.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await upload.close()
    if not payload:
        raise HTTPException(status_code=400, detail=problem_detail("invalid_image", "Uploaded image is empty."))
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=problem_detail("image_too_large", "Uploaded image exceeds the size limit."))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(payload))
            image.verify()
            image = Image.open(io.BytesIO(payload))
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=problem_detail("image_too_many_pixels", "Decoded image exceeds the pixel limit."),
                )
            image = image.convert("RGB")
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(
            status_code=413,
            detail=problem_detail("image_too_many_pixels", "Decoded image exceeds the pixel limit."),
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=problem_detail("invalid_image", "Uploaded file is not a valid image.")) from exc
    return payload, image


def image_info(image: Image.Image, upload: Optional[UploadFile] = None) -> Dict[str, Any]:
    return {
        "filename": getattr(upload, "filename", None),
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
    }


def image_embedding_for(image: Image.Image) -> List[float]:
    global embedding_dimension
    model_instance, processor_instance = load_model()
    try:
        inputs = _move_inputs_to_device(processor_instance(images=image, return_tensors="pt"))
        with inference_mode():
            features = _feature_tensor(model_instance.get_image_features(**inputs))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=problem_detail("model_inference_failed", "Failed to encode image.")) from exc
    embedding = [float(value) for value in normalize_vectors(features)[0].tolist()]
    embedding_dimension = len(embedding)
    return embedding


def text_embedding_for(text: str) -> List[float]:
    global embedding_dimension
    model_instance, processor_instance = load_model()
    try:
        inputs = _move_inputs_to_device(processor_instance(text=[text], return_tensors="pt", padding=True))
        with inference_mode():
            features = _feature_tensor(model_instance.get_text_features(**inputs))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=problem_detail("model_inference_failed", "Failed to encode text.")) from exc
    embedding = [float(value) for value in normalize_vectors(features)[0].tolist()]
    embedding_dimension = len(embedding)
    return embedding


def warmup_model() -> None:
    global model_warmup_completed
    model_warmup_completed = False
    image_embedding_for(Image.new("RGB", (32, 32), "white"))
    model_warmup_completed = True


async def run_ocr(image: Image.Image, lang: str, psm: int) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not OCR_ENABLED:
        return None, "OCR is disabled by service configuration."
    if _extract_ocr is None:
        return None, "OCR service is not available."
    try:
        with anyio.fail_after(max(0.1, float(OCR_TIMEOUT_MS) / 1000.0)):
            invocation = partial(_extract_ocr, image, lang=lang, psm=int(psm))
            result = await anyio.to_thread.run_sync(invocation, limiter=ocr_limiter, abandon_on_cancel=True)
        return result, None
    except TimeoutError:
        logger.warning(json.dumps({"event": "ocr_failure", "reason": "timeout"}))
        return None, "OCR timed out."
    except Exception as exc:  # pragma: no cover - depends on optional runtime tools
        logger.warning(json.dumps({"event": "ocr_failure", "reason": type(exc).__name__}))
        return None, "OCR failed."


async def run_barcode_scan(image: Image.Image) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    if _scan_barcodes is None:
        return None, "Barcode service is not available."
    try:
        with anyio.fail_after(max(0.1, float(BARCODE_TIMEOUT_MS) / 1000.0)):
            result = await anyio.to_thread.run_sync(_scan_barcodes, image)
        return result, None
    except TimeoutError:
        return None, "Barcode scan timed out."
    except Exception as exc:  # pragma: no cover - depends on optional runtime tools
        return None, str(exc)


def softmax_confidences(scores: List[float], temperature: float = 0.07) -> List[float]:
    if not scores:
        return []
    t = temperature if temperature > 0 else 0.07
    scaled = np.asarray(scores, dtype=float) / t
    shifted = scaled - np.max(scaled)
    exps = np.exp(shifted)
    total = float(np.sum(exps)) or 1.0
    return [float(value / total) for value in exps]


def should_return_no_match(top_score: float, second_score: Optional[float], min_score: float, min_margin: float) -> tuple[bool, Dict[str, Any]]:
    if float(top_score) < float(min_score):
        return True, {"reason": "LOW_TOP_SCORE", "topScore": float(top_score), "minScore": float(min_score)}
    if second_score is not None and (float(top_score) - float(second_score)) < float(min_margin):
        return True, {
            "reason": "LOW_MARGIN",
            "topScore": float(top_score),
            "secondScore": float(second_score),
            "minMargin": float(min_margin),
        }
    return False, {
        "reason": "OK",
        "topScore": float(top_score),
        "secondScore": None if second_score is None else float(second_score),
        "minScore": float(min_score),
        "minMargin": float(min_margin),
    }


def match_explanation(item: Mapping[str, Any], query_ocr_text: Optional[str], query_barcodes: List[Dict[str, Any]], query_labels: List[str], score: float) -> Dict[str, Any]:
    item_barcodes = normalize_barcodes(item.get("barcodes") or item.get("barcodeValues"))
    barcode_overlap = barcode_overlap_values(query_barcodes, item_barcodes)

    query_ocr_tokens = set(tokenize(query_ocr_text))
    item_ocr_tokens = set(tokenize(item.get("ocrText")))
    ocr_overlap = sorted(query_ocr_tokens & item_ocr_tokens)[:5]

    query_label_map = {value.lower(): value for value in query_labels}
    item_label_map = {value.lower(): value for value in normalize_string_list(item.get("labels"))}
    label_overlap = [item_label_map[key] for key in sorted(query_label_map.keys() & item_label_map.keys())][:5]

    signals: Dict[str, Any] = {"clipCosine": round(float(score), 6)}
    reasons: List[str] = []
    if barcode_overlap:
        signals["barcode"] = barcode_overlap
        reasons.append("barcode overlap")
    if ocr_overlap:
        signals["ocr"] = ocr_overlap
        reasons.append("OCR overlap")
    if label_overlap:
        signals["labels"] = label_overlap
        reasons.append("shared labels")
    if not reasons:
        reasons.append("visual similarity")
    similarity_band = "high" if score >= 0.85 else "medium" if score >= 0.65 else "low"
    reason = ", ".join(reasons)
    return {
        "reason": reason,
        "summary": reason,
        "signals": signals,
        "scoreBand": similarity_band,
        "similarity": {"cosine": round(float(score), 6), "band": similarity_band},
        "ocr": {"matchedTerms": ocr_overlap, "matchCount": len(ocr_overlap)},
        "barcode": {"matchedValues": barcode_overlap, "matchCount": len(barcode_overlap)},
        "labels": {"matchedValues": label_overlap, "matchCount": len(label_overlap)},
        "model": {
            "engine": "clip",
            "modelId": MODEL_NAME,
            "serviceVersion": SERVICE_VERSION,
            "scoringVersion": SCORING_VERSION,
        },
    }


class ItemRepository:
    def health(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_item(self, item_id: str, identity: ServiceIdentity) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def delete_item(self, item_id: str, identity: ServiceIdentity) -> bool:
        raise NotImplementedError

    def list_items(self, identity: ServiceIdentity, status: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def list_released_items(self, identity: ServiceIdentity, limit: int = 5000) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def add_audit_log(
        self,
        event_type: str,
        payload: Dict[str, Any],
        item_id: Optional[str] = None,
        request_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def list_audit_logs(
        self, identity: ServiceIdentity, item_id: Optional[str] = None, limit: int = 200, offset: int = 0
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class InMemoryItemRepository(ItemRepository):
    def __init__(self):
        self._items: Dict[str, Dict[str, Any]] = {}
        self._logs: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    @staticmethod
    def _key(tenant_id: str, site_id: str, item_id: str) -> str:
        return f"{tenant_id}\x1f{site_id}\x1f{item_id}"

    def health(self) -> Dict[str, Any]:
        return {"backend": "memory", "configured": False, "ok": True, "items": len(self._items)}

    def get_item(self, item_id: str, identity: ServiceIdentity) -> Optional[Dict[str, Any]]:
        with self._lock:
            for site_id in identity.permitted_site_ids:
                item = self._items.get(self._key(identity.tenant_id, site_id, item_id))
                if item:
                    return dict(item)
            return None

    def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            stored = dict(item)
            self._items[self._key(item["tenantId"], item["siteId"], item["id"])] = stored
            return dict(stored)

    def delete_item(self, item_id: str, identity: ServiceIdentity) -> bool:
        with self._lock:
            for site_id in identity.permitted_site_ids:
                key = self._key(identity.tenant_id, site_id, item_id)
                if key in self._items:
                    del self._items[key]
                    return True
            return False

    def list_items(self, identity: ServiceIdentity, status: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        with self._lock:
            items = [
                dict(value)
                for value in self._items.values()
                if value.get("tenantId") == identity.tenant_id and value.get("siteId") in identity.permitted_site_ids
            ]
        if status:
            items = [item for item in items if item.get("status") == status]
        items.sort(key=lambda item: item.get("updatedAt") or "", reverse=True)
        return items[:limit]

    def list_released_items(self, identity: ServiceIdentity, limit: int = 5000) -> List[Dict[str, Any]]:
        with self._lock:
            items = [
                dict(value)
                for value in self._items.values()
                if value.get("tenantId") == identity.tenant_id
                and value.get("siteId") in identity.permitted_site_ids
                and value.get("released")
                and value.get("eligibleForMatching")
                and value.get("modelId") == MODEL_NAME
                and value.get("modelRevision") == MODEL_REVISION
                and value.get("embeddingDimension") == (embedding_dimension or EXPECTED_EMBEDDING_DIMENSION)
            ]
        items.sort(key=lambda item: item.get("updatedAt") or "", reverse=True)
        return items[:limit]

    def add_audit_log(
        self,
        event_type: str,
        payload: Dict[str, Any],
        item_id: Optional[str] = None,
        request_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        with self._lock:
            log_id = str(uuid.uuid4())
            self._logs.append(
                {
                    "id": log_id,
                    "ts": now_iso(),
                    "eventType": event_type,
                    "itemId": item_id,
                    "requestId": request_id,
                    "tenantId": tenant_id,
                    "payload": payload or {},
                }
            )
            return log_id

    def list_audit_logs(
        self, identity: ServiceIdentity, item_id: Optional[str] = None, limit: int = 200, offset: int = 0
    ) -> List[Dict[str, Any]]:
        with self._lock:
            logs = [log for log in reversed(self._logs) if log.get("tenantId") == identity.tenant_id]
        if item_id:
            logs = [log for log in logs if log.get("itemId") == item_id]
        return logs[offset : offset + limit]


class UnavailableItemRepository(ItemRepository):
    def __init__(self, reason: str):
        self.reason = reason

    def health(self) -> Dict[str, Any]:
        return {"backend": "mongo", "configured": True, "ok": False, "error": self.reason}

    def _raise(self) -> None:
        raise HTTPException(status_code=503, detail=problem_detail("store_unavailable", self.reason))

    def get_item(self, item_id: str, identity: ServiceIdentity) -> Optional[Dict[str, Any]]:
        self._raise()

    def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        self._raise()

    def delete_item(self, item_id: str, identity: ServiceIdentity) -> bool:
        self._raise()

    def list_items(self, identity: ServiceIdentity, status: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        self._raise()

    def list_released_items(self, identity: ServiceIdentity, limit: int = 5000) -> List[Dict[str, Any]]:
        self._raise()

    def add_audit_log(
        self,
        event_type: str,
        payload: Dict[str, Any],
        item_id: Optional[str] = None,
        request_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        self._raise()

    def list_audit_logs(
        self, identity: ServiceIdentity, item_id: Optional[str] = None, limit: int = 200, offset: int = 0
    ) -> List[Dict[str, Any]]:
        self._raise()


class MongoItemRepository(ItemRepository):
    def __init__(self, uri: str, database: str, items_collection: str, audit_collection: str):
        from pymongo import MongoClient

        self._client = MongoClient(uri, serverSelectionTimeoutMS=1000)
        self._items = self._client[database][items_collection]
        self._audit = self._client[database][audit_collection]
        self._items.create_index([("tenantId", 1), ("siteId", 1), ("id", 1)], unique=True)
        self._items.create_index([("tenantId", 1), ("siteId", 1), ("status", 1), ("modelId", 1), ("modelRevision", 1)])
        self._items.create_index([("status", 1), ("updatedAt", -1)])
        self._audit.create_index([("ts", -1)])
        self._audit.create_index([("itemId", 1), ("ts", -1)])

    def health(self) -> Dict[str, Any]:
        try:
            self._client.admin.command("ping")
            return {"backend": "mongo", "configured": True, "ok": True}
        except Exception as exc:  # pragma: no cover - depends on live mongo
            return {"backend": "mongo", "configured": True, "ok": False, "error": str(exc)}

    def _clean_item(self, item: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if item is None:
            return None
        cleaned = dict(item)
        cleaned.pop("_id", None)
        return cleaned

    def get_item(self, item_id: str, identity: ServiceIdentity) -> Optional[Dict[str, Any]]:
        return self._clean_item(
            self._items.find_one(
                {"id": item_id, "tenantId": identity.tenant_id, "siteId": {"$in": list(identity.permitted_site_ids)}}
            )
        )

    def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        selector = {"id": item["id"], "tenantId": item["tenantId"], "siteId": item["siteId"]}
        self._items.update_one(selector, {"$set": item}, upsert=True)
        stored = self._clean_item(self._items.find_one(selector))
        if stored is None:
            raise HTTPException(status_code=503, detail=problem_detail("store_unavailable", "Failed to read item after upsert."))
        return stored

    def delete_item(self, item_id: str, identity: ServiceIdentity) -> bool:
        result = self._items.delete_one(
            {"id": item_id, "tenantId": identity.tenant_id, "siteId": {"$in": list(identity.permitted_site_ids)}}
        )
        return result.deleted_count == 1

    def list_items(self, identity: ServiceIdentity, status: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {
            "tenantId": identity.tenant_id,
            "siteId": {"$in": list(identity.permitted_site_ids)},
        }
        if status:
            query["status"] = status
        cursor = self._items.find(query).sort("updatedAt", -1).limit(limit)
        return [self._clean_item(item) for item in cursor if item is not None]

    def list_released_items(self, identity: ServiceIdentity, limit: int = 5000) -> List[Dict[str, Any]]:
        cursor = (
            self._items.find(
                {
                    "tenantId": identity.tenant_id,
                    "siteId": {"$in": list(identity.permitted_site_ids)},
                    "released": True,
                    "eligibleForMatching": True,
                    "modelId": MODEL_NAME,
                    "modelRevision": MODEL_REVISION,
                    "embeddingDimension": embedding_dimension or EXPECTED_EMBEDDING_DIMENSION,
                }
            )
            .sort("updatedAt", -1)
            .limit(limit)
        )
        return [self._clean_item(item) for item in cursor if item is not None]

    def add_audit_log(
        self,
        event_type: str,
        payload: Dict[str, Any],
        item_id: Optional[str] = None,
        request_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        document = {
            "ts": now_iso(),
            "eventType": event_type,
            "itemId": item_id,
            "requestId": request_id,
            "tenantId": tenant_id,
            "payload": payload or {},
        }
        return str(self._audit.insert_one(document).inserted_id)

    def list_audit_logs(
        self, identity: ServiceIdentity, item_id: Optional[str] = None, limit: int = 200, offset: int = 0
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"tenantId": identity.tenant_id}
        if item_id:
            query["itemId"] = item_id
        cursor = self._audit.find(query).sort("ts", -1).skip(offset).limit(limit)
        logs: List[Dict[str, Any]] = []
        for log in cursor:
            cleaned = dict(log)
            cleaned["id"] = str(cleaned.pop("_id"))
            logs.append(cleaned)
        return logs


repository_lock = threading.Lock()
repository: Optional[ItemRepository] = None


def build_repository() -> ItemRepository:
    strict_environment = ENVIRONMENT in {"festival", "staging", "prod", "production"}
    if STORAGE_MODE == "memory":
        if strict_environment:
            raise RuntimeError("In-memory storage is forbidden in festival, staging, and production environments")
        return InMemoryItemRepository()
    if STORAGE_MODE != "mongodb":
        raise RuntimeError("STORAGE_MODE must be explicitly configured as 'memory' or 'mongodb'")
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is required when STORAGE_MODE=mongodb")
    mongo_repository = MongoItemRepository(MONGODB_URI, MONGODB_DB, MONGODB_ITEMS_COLLECTION, MONGODB_AUDIT_COLLECTION)
    health = mongo_repository.health()
    if not health.get("ok"):
        raise RuntimeError("MongoDB is unavailable")
    return mongo_repository


def get_repository() -> ItemRepository:
    global repository
    if repository is None:
        with repository_lock:
            if repository is None:
                repository = build_repository()
    return repository


def set_item_repository(item_repository: ItemRepository) -> None:
    global repository
    repository = item_repository


def serialize_item(item: Mapping[str, Any], include_embedding: bool = False) -> Dict[str, Any]:
    payload = {
        "id": item.get("id"),
        "title": item.get("title"),
        "name": item.get("name"),
        "description": item.get("description"),
        "status": item.get("status"),
        "released": bool(item.get("released")),
        "eligibleForMatching": bool(item.get("eligibleForMatching")),
        "ocrText": item.get("ocrText"),
        "ocr": item.get("ocr"),
        "ocrWords": item.get("ocrWords"),
        "barcodes": item.get("barcodes") or [],
        "barcodeValues": item.get("barcodeValues") or [],
        "labels": item.get("labels") or [],
        "metadata": item.get("metadata") or {},
        "image": item.get("image"),
        "createdAt": item.get("createdAt"),
        "updatedAt": item.get("updatedAt"),
        "releasedAt": item.get("releasedAt"),
        "tenantId": item.get("tenantId"),
        "siteId": item.get("siteId"),
        "datasetVersion": item.get("datasetVersion"),
        "demoData": bool(item.get("demoData")),
        "createdBy": item.get("createdBy"),
        "modelId": item.get("modelId"),
        "modelRevision": item.get("modelRevision"),
        "embeddingDimension": item.get("embeddingDimension"),
    }
    return payload


def parse_item_payload(raw_payload: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(raw_payload)
    nested = payload.get("item")
    if isinstance(nested, Mapping):
        payload = {**nested, **{key: value for key, value in payload.items() if key != "item"}}
    forbidden = {
        "tenantId",
        "siteId",
        "datasetVersion",
        "demoData",
        "createdBy",
        "createdAt",
        "status",
        "released",
        "eligibleForMatching",
        "embedding",
        "clipEmbedding",
        "modelId",
        "modelRevision",
        "embeddingDimension",
    }
    supplied_forbidden = sorted(forbidden.intersection(payload))
    if supplied_forbidden:
        raise HTTPException(
            status_code=400,
            detail=problem_detail(
                "system_fields_forbidden",
                "Corpus system fields are controlled by the trusted service.",
                {"fields": supplied_forbidden},
            ),
        )

    item_id = compact_text(str(first_value(payload, "id", "itemId") or str(uuid.uuid4())), limit=120)
    if not item_id:
        raise HTTPException(status_code=400, detail=problem_detail("missing_item_id", "Item id is required."))

    item: Dict[str, Any] = {"id": item_id}
    name = compact_text(first_value(payload, "title", "name"), limit=240)
    if name is not None:
        item["title"] = name
        item["name"] = name
    if "description" in payload:
        item["description"] = compact_text(payload.get("description"), limit=500)
    if "metadata" in payload:
        item["metadata"] = normalize_mapping(payload.get("metadata"), "metadata")
    if "labels" in payload or "tags" in payload:
        item["labels"] = normalize_string_list(first_value(payload, "labels", "tags"))
    ocr_payload = payload.get("ocr")
    ocr_text = None
    ocr_words = None
    ocr_meta = {}
    if isinstance(ocr_payload, Mapping):
        ocr_text = compact_text(ocr_payload.get("fullText"), limit=500)
        words_value = ocr_payload.get("words")
        ocr_words = list(words_value) if isinstance(words_value, list) else None
        meta_value = ocr_payload.get("meta")
        ocr_meta = dict(meta_value) if isinstance(meta_value, Mapping) else {}
    else:
        if "ocrText" in payload or "ocr" in payload:
            ocr_text = compact_text(first_value(payload, "ocrText", "ocr"), limit=500)
        if "ocrWords" in payload and isinstance(payload.get("ocrWords"), list):
            ocr_words = list(payload.get("ocrWords"))
    if ocr_text is not None or ocr_words is not None:
        item["ocrText"] = ocr_text
        item["ocrWords"] = ocr_words or []
        item["ocr"] = {"fullText": ocr_text or "", "words": ocr_words or [], "meta": ocr_meta}

    if "barcodes" in payload or "barcodeValues" in payload:
        barcodes = normalize_barcodes(first_value(payload, "barcodes", "barcodeValues"))
        item["barcodes"] = barcodes
        item["barcodeValues"] = barcode_values(barcodes)

    extras = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "item",
            "id",
            "itemId",
            "title",
            "name",
            "description",
            "metadata",
            "labels",
            "tags",
            "status",
            "released",
            "embedding",
            "clipEmbedding",
            "ocr",
            "ocrText",
            "ocrWords",
            "barcodes",
            "barcodeValues",
        }
    }
    if extras:
        item["metadata"] = {**item.get("metadata", {}), "extra": extras}
    return item


async def parse_items_request(request: Request) -> Dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    upload: Optional[UploadFile] = None
    if "application/json" in content_type:
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=problem_detail("invalid_json", "Request body must contain valid JSON.")) from exc
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=400, detail=problem_detail("invalid_payload", "Request body must be an object."))
        return dict(body)

    form = await request.form()
    parsed = {key: value for key, value in form.items() if not is_upload(value)}
    item_blob = form.get("item")
    if isinstance(item_blob, str) and item_blob.strip():
        try:
            nested = json.loads(item_blob)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=problem_detail("invalid_json", "Field 'item' must contain valid JSON.")) from exc
        if not isinstance(nested, Mapping):
            raise HTTPException(status_code=400, detail=problem_detail("invalid_payload", "Field 'item' must contain an object."))
        parsed = {**nested, **parsed}
    upload_value = first_value(form, "file", "image")
    upload = upload_value if is_upload(upload_value) else None
    if upload is not None:
        parsed["upload"] = upload
    return parsed


def prepare_item_for_storage(
    existing: Optional[Dict[str, Any]], patch: Dict[str, Any], identity: ServiceIdentity
) -> Dict[str, Any]:
    current = dict(existing or {})
    current.update(patch)
    current["id"] = patch["id"]
    current["tenantId"] = identity.tenant_id
    current["siteId"] = identity.site_id
    current["datasetVersion"] = identity.dataset_version
    current["demoData"] = identity.demo_data
    current["createdBy"] = current.get("createdBy") or identity.service_name
    current["modelId"] = MODEL_NAME
    current["modelRevision"] = MODEL_REVISION
    title = current.get("title") or current.get("name")
    current["title"] = title
    current["name"] = title
    current.setdefault("description", None)
    current.setdefault("metadata", {})
    current.setdefault("labels", [])
    current.setdefault("barcodes", [])
    current.setdefault("barcodeValues", barcode_values(normalize_barcodes(current.get("barcodes") or current.get("barcodeValues"))))
    current.setdefault("ocrWords", [])
    if current.get("ocrText") is not None:
        current["ocr"] = {
            "fullText": current.get("ocrText") or "",
            "words": current.get("ocrWords") or [],
            "meta": (current.get("ocr") or {}).get("meta", {}) if isinstance(current.get("ocr"), Mapping) else {},
        }
    else:
        current.setdefault("ocr", None)
    if current.get("embedding") is not None:
        current["embedding"] = validate_stored_embedding(current["embedding"])
        current["clipEmbedding"] = current["embedding"]
        current["embeddingDimension"] = len(current["embedding"])
    else:
        current["embeddingDimension"] = embedding_dimension or EXPECTED_EMBEDDING_DIMENSION
    current.setdefault("released", False)
    has_embedding = bool(current.get("embedding"))
    current["eligibleForMatching"] = bool(current.get("released")) and has_embedding
    explicit_status = compact_text(str(current.get("status")), limit=32) if current.get("status") not in (None, "") else None
    if explicit_status and explicit_status not in {"stored", "released", "archived", "draft"}:
        explicit_status = "stored"
    current["status"] = "released" if current["eligibleForMatching"] else (explicit_status or "stored")
    current["createdAt"] = current.get("createdAt") or now_iso()
    current["updatedAt"] = now_iso()
    if current.get("released") and has_embedding:
        current["releasedAt"] = current.get("releasedAt") or now_iso()
    return current


def parse_match_request(form: Mapping[str, Any]) -> MatchRequestModel:
    forbidden_controls = {
        "status",
        "thresholds",
        "minScore",
        "minMargin",
        "temperature",
        "doOcr",
        "doBarcode",
        "ocrLang",
        "ocrPsm",
        "modelId",
        "modelRevision",
        "embeddingDimension",
    }
    supplied = sorted(forbidden_controls.intersection(form))
    if supplied:
        raise HTTPException(
            status_code=400,
            detail=problem_detail(
                "system_controls_forbidden",
                "Matching policy and model controls are configured by the service.",
                {"fields": supplied},
            ),
        )
    metadata_input = parse_optional_object(first_value(form, "metadata"), "metadata")
    payload = {
        "text": compact_text(first_value(form, "text", "queryText", "query"), limit=2000),
        "topK": parse_top_k(first_value(form, "topK", "k", "limit")),
        "status": "released",
        "thresholds": {
            "minScore": CONF_MIN_SCORE,
            "minMargin": CONF_MIN_MARGIN,
            "temperature": CONF_TEMPERATURE,
        },
        "metadata": metadata_input,
        "ocrText": compact_text(first_value(form, "ocrText", "ocr"), limit=500),
        "barcodeValues": barcode_values(normalize_barcodes(first_value(form, "barcodeValues", "barcodes"))),
        "labels": normalize_string_list(first_value(form, "labels", "tags")),
        "doOcr": OCR_ENABLED,
        "doBarcode": True,
        "ocrLang": OCR_LANGUAGES,
        "ocrPsm": OCR_PSM,
    }
    try:
        return MatchRequestModel.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=problem_detail("validation_error", "Request validation failed.", exc.errors()),
        ) from exc


UI_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CLIP Service</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 900px; line-height: 1.5; padding: 0 1rem; }
      code { background: #f4f4f4; padding: 0.15rem 0.35rem; border-radius: 4px; }
      pre { background: #111827; color: #f9fafb; padding: 1rem; border-radius: 8px; overflow: auto; }
    </style>
  </head>
  <body>
    <h1>CLIP Service</h1>
    <p>This service exposes CLIP encoding plus backend-compatible analyze, item, and match endpoints.</p>
    <ul>
      <li><code>GET /health</code> - dependency health</li>
      <li><code>POST /encode-image</code> - image embedding</li>
      <li><code>POST /encode-text</code> - text embedding</li>
      <li><code>POST /analyze-image</code> - compact image analysis</li>
      <li><code>POST /items</code> - upsert stored item</li>
      <li><code>POST /items/{id}/release</code> - release item for matching</li>
      <li><code>POST /match</code> - match against released items</li>
    </ul>
  </body>
</html>
"""


@app.get("/metrics")
def metrics(request: Request):
    authorize(request, "metrics:read")
    if generate_latest is None:
        raise HTTPException(status_code=503, detail=problem_detail("metrics_unavailable", "Metrics not available."))
    if METRIC_MODEL_LOADED is not None:
        METRIC_MODEL_LOADED.set(1 if model is not None and processor is not None else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def home(request: Request):
    return with_request_id(request, {"status": "ok"})


@app.get("/health")
def health(request: Request):
    authorize(request, "health:read")
    store_health = get_repository().health()
    overall_status = "ok" if store_health.get("ok", False) else "degraded"
    return with_request_id(
        request,
        {
            "status": overall_status,
            "dependencies": {
                "model": model_health(),
                "store": store_health,
            },
        },
    )


@app.get("/health/live")
def health_live(request: Request):
    return with_request_id(request, {"status": "ok"})


@app.get("/health/ready")
def health_ready(request: Request):
    database_ready = False
    try:
        database_ready = bool(get_repository().health().get("ok"))
    except Exception:
        database_ready = False
    checks = {
        "modelLoaded": model is not None and processor is not None,
        "embeddingDimensionKnown": isinstance(embedding_dimension, int) and embedding_dimension > 0,
        "ocrDependencyReady": (not OCR_ENABLED) or ocr_dependency_ready(OCR_LANGUAGES),
        "databaseReady": database_ready,
        "device": INFERENCE_DEVICE,
        "warmupCompleted": model_warmup_completed,
    }
    ready = all(value for key, value in checks.items() if key != "device")
    payload = with_request_id(request, {"status": "ready" if ready else "not_ready", "checks": checks})
    return JSONResponse(status_code=200 if ready else 503, content=payload)


def initialize_runtime() -> None:
    validate_auth_configuration()
    configured_device()
    store = get_repository()
    if not store.health().get("ok"):
        raise RuntimeError("Configured database is unavailable")
    if OCR_ENABLED and not ocr_dependency_ready(OCR_LANGUAGES):
        raise RuntimeError("Configured OCR dependency or language data is unavailable")
    warmup_model()


@app.on_event("startup")
def startup_runtime() -> None:
    initialize_runtime()


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(UI_HTML)


@app.post("/encode-image")
async def encode_image(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
):
    started = time.time()
    endpoint = "/encode-image"
    authorize(request, "match:execute")
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail=problem_detail("missing_image", "No image uploaded. Use form field 'file' or 'image'."))
    _, pil_image = await read_image_upload(upload, "file")
    encoded = image_embedding_for(pil_image)
    response = with_request_id(
        request,
        {
            "encoded": True,
            "modelId": MODEL_NAME,
            "modelRevision": MODEL_REVISION,
            "embeddingDimension": len(encoded),
        },
    )
    record_latency_metric(endpoint, started)
    record_request_metric(endpoint, 200)
    return response


@app.post("/encode-text")
async def encode_text(
    request: Request,
    text: Optional[str] = Form(default=None),
    queryText: Optional[str] = Form(default=None),
):
    started = time.time()
    endpoint = "/encode-text"
    authorize(request, "match:execute")
    value = compact_text(text or queryText, limit=2000)
    if not value:
        raise HTTPException(status_code=400, detail=problem_detail("missing_text", "No text provided. Use form field 'text' or 'queryText'."))
    encoded = text_embedding_for(value)
    response = with_request_id(
        request,
        {
            "encoded": True,
            "modelId": MODEL_NAME,
            "modelRevision": MODEL_REVISION,
            "embeddingDimension": len(encoded),
        },
    )
    record_latency_metric(endpoint, started)
    record_request_metric(endpoint, 200)
    return response


@app.post("/similarity")
async def similarity(
    request: Request,
    file1: Optional[UploadFile] = File(default=None),
    file2: Optional[UploadFile] = File(default=None),
    image1: Optional[UploadFile] = File(default=None),
    image2: Optional[UploadFile] = File(default=None),
):
    started = time.time()
    endpoint = "/similarity"
    authorize(request, "match:execute")
    upload_1 = file1 or image1
    upload_2 = file2 or image2
    if upload_1 is None or upload_2 is None:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("missing_image", "Two images are required. Use fields file1/file2 or image1/image2."),
        )
    _, image_1 = await read_image_upload(upload_1, "file1")
    _, image_2 = await read_image_upload(upload_2, "file2")
    embedding_1 = normalize_vectors(image_embedding_for(image_1))[0]
    embedding_2 = normalize_vectors(image_embedding_for(image_2))[0]
    response = with_request_id(request, {"similarity": float(np.dot(embedding_1, embedding_2))})
    record_latency_metric(endpoint, started)
    record_request_metric(endpoint, 200)
    return response


@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
):
    started = time.time()
    endpoint = "/analyze-image"
    identity = authorize(request, "match:execute")
    do_ocr = OCR_ENABLED
    do_barcode = True
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail=problem_detail("missing_image", "No image uploaded. Use form field 'file' or 'image'."))

    raw, pil_image = await read_image_upload(upload, "file")
    request_id = get_request_id(request)
    manual_ocr_text = compact_text(request.query_params.get("ocrText"), limit=500)
    manual_barcodes = []
    manual_labels: List[str] = []

    try:
        form = await request.form()
        manual_ocr_text = compact_text(first_value(form, "ocrText", "ocr"), limit=500) or manual_ocr_text
        manual_barcodes = normalize_barcodes(first_value(form, "barcodeValues", "barcodes"))
        manual_labels = normalize_string_list(first_value(form, "labels", "tags"))
    except Exception as exc:
        logger.warning("Failed to read optional analyze-image form signals: %s", exc)

    ocr_payload = None
    ocr_error = None
    if do_ocr:
        ocr_payload, ocr_error = await run_ocr(pil_image, OCR_LANGUAGES, OCR_PSM)

    barcode_payload = None
    barcode_error = None
    if do_barcode:
        barcode_payload, barcode_error = await run_barcode_scan(pil_image)

    scanned_barcodes = normalize_barcodes((barcode_payload or {}).get("barcodes"))
    combined_barcodes = merge_barcodes(manual_barcodes, scanned_barcodes)
    effective_ocr_text = manual_ocr_text or ((ocr_payload or {}).get("fullText"))
    effective_barcode_values = barcode_values(combined_barcodes)

    get_repository().add_audit_log(
        "IMAGE_ANALYZE",
        {
            "input": {"sha256": sha256_hex(raw), "bytes": len(raw), "contentType": getattr(upload, "content_type", None)},
            "ocrEnabled": do_ocr,
            "barcodeEnabled": do_barcode,
            "ocrError": ocr_error,
            "barcodeError": barcode_error,
        },
        request_id=request_id,
        tenant_id=identity.tenant_id,
    )
    if ocr_error:
        get_repository().add_audit_log(
            "ocr_failure",
            {"reason": ocr_error, "inputSha256": sha256_hex(raw)},
            request_id=request_id,
            tenant_id=identity.tenant_id,
        )

    response = with_request_id(
        request,
        {
            "governance": governance_meta(ocrEnabled=do_ocr, barcodeEnabled=do_barcode),
            "image": image_info(pil_image, upload),
            "input": {"sha256": sha256_hex(raw), "bytes": len(raw), "contentType": getattr(upload, "content_type", None)},
            "model": {
                "modelId": MODEL_NAME,
                "modelRevision": MODEL_REVISION,
                "embeddingDimension": len(image_embedding_for(pil_image)),
                "device": INFERENCE_DEVICE,
            },
            "ocr": ocr_payload,
            "ocrError": ocr_error,
            "barcode": {"barcodes": combined_barcodes, "meta": (barcode_payload or {}).get("meta")} if combined_barcodes or barcode_payload else None,
            "barcodeError": barcode_error,
            "signals": compact_signals(effective_ocr_text, effective_barcode_values, manual_labels),
        },
    )
    record_latency_metric(endpoint, started)
    record_request_metric(endpoint, 200)
    return response


@app.post("/items")
async def upsert_item(request: Request):
    started = time.time()
    endpoint = "/items"
    identity = authorize(request, "corpus:write")
    parsed_payload = await parse_items_request(request)
    upload = parsed_payload.pop("upload", None)
    normalized_patch = parse_item_payload(parsed_payload)

    existing = get_repository().get_item(normalized_patch["id"], identity)
    stored_item = prepare_item_for_storage(existing, normalized_patch, identity)

    if upload is not None:
        _, pil_image = await read_image_upload(upload, "file")
        stored_item["embedding"] = image_embedding_for(pil_image)
        stored_item["clipEmbedding"] = stored_item["embedding"]
        stored_item["embeddingDimension"] = len(stored_item["embedding"])
        stored_item["image"] = image_info(pil_image, upload)
        stored_item["eligibleForMatching"] = bool(stored_item.get("released")) and True
        stored_item["status"] = "released" if stored_item["eligibleForMatching"] else "stored"

    saved_item = get_repository().upsert_item(stored_item)
    get_repository().add_audit_log(
        "corpus_creation" if existing is None else "corpus_update",
        {"status": saved_item.get("status"), "released": saved_item.get("released"), "hasEmbedding": bool(saved_item.get("embedding"))},
        item_id=saved_item["id"],
        request_id=get_request_id(request),
        tenant_id=identity.tenant_id,
    )

    response = with_request_id(request, {"governance": governance_meta(), "item": serialize_item(saved_item)})
    record_latency_metric(endpoint, started)
    record_request_metric(endpoint, 200)
    return response


@app.get("/items")
def list_items(
    request: Request,
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
):
    identity = authorize(request, "corpus:read")
    if status not in {None, "stored", "released", "archived", "draft"}:
        raise HTTPException(status_code=400, detail=problem_detail("invalid_status", "Invalid corpus status filter."))
    items = [serialize_item(item) for item in get_repository().list_items(identity, status=status, limit=limit)]
    return with_request_id(request, {"items": items})


@app.get("/items/{item_id}")
def get_item(request: Request, item_id: str):
    identity = authorize(request, "corpus:read")
    item = get_repository().get_item(item_id, identity)
    if item is None:
        raise HTTPException(status_code=404, detail=problem_detail("item_not_found", f"Item '{item_id}' was not found."))
    return with_request_id(request, {"item": serialize_item(item)})


@app.post("/items/{item_id}/release")
def release_item(request: Request, item_id: str):
    started = time.time()
    endpoint = "/items/release"
    identity = authorize(request, "corpus:release")
    existing = get_repository().get_item(item_id, identity)
    if existing is None:
        raise HTTPException(status_code=404, detail=problem_detail("item_not_found", f"Item '{item_id}' was not found."))
    if not existing.get("embedding"):
        raise HTTPException(
            status_code=409,
            detail=problem_detail("item_not_matchable", f"Item '{item_id}' cannot be released without an embedding or image."),
        )
    validate_stored_embedding(existing["embedding"])
    if existing.get("modelId") != MODEL_NAME or existing.get("modelRevision") != MODEL_REVISION:
        raise HTTPException(status_code=409, detail=problem_detail("model_incompatible", "Corpus item model is incompatible."))
    existing["released"] = True
    existing["eligibleForMatching"] = True
    existing["status"] = "released"
    existing["releasedAt"] = existing.get("releasedAt") or now_iso()
    existing["updatedAt"] = now_iso()
    saved_item = get_repository().upsert_item(existing)
    get_repository().add_audit_log(
        "corpus_release",
        {"status": "released", "releasedAt": saved_item.get("releasedAt")},
        item_id=item_id,
        request_id=get_request_id(request),
        tenant_id=identity.tenant_id,
    )
    response = with_request_id(request, {"item": serialize_item(saved_item)})
    record_latency_metric(endpoint, started)
    record_request_metric(endpoint, 200)
    return response


@app.delete("/items/{item_id}")
def remove_item(request: Request, item_id: str):
    identity = authorize(request, "corpus:write")
    if not get_repository().delete_item(item_id, identity):
        raise HTTPException(status_code=404, detail=problem_detail("item_not_found", f"Item '{item_id}' was not found."))
    get_repository().add_audit_log(
        "corpus_removal",
        {},
        item_id=item_id,
        request_id=get_request_id(request),
        tenant_id=identity.tenant_id,
    )
    return with_request_id(request, {"removed": True, "itemId": item_id})


@app.post("/items/{item_id}/re-embed")
async def reembed_item(request: Request, item_id: str, file: UploadFile = File(...)):
    identity = authorize(request, "corpus:write")
    existing = get_repository().get_item(item_id, identity)
    if existing is None:
        raise HTTPException(status_code=404, detail=problem_detail("item_not_found", f"Item '{item_id}' was not found."))
    _, image = await read_image_upload(file, "file")
    existing["embedding"] = image_embedding_for(image)
    existing["clipEmbedding"] = existing["embedding"]
    existing["embeddingDimension"] = len(existing["embedding"])
    existing["modelId"] = MODEL_NAME
    existing["modelRevision"] = MODEL_REVISION
    existing["updatedAt"] = now_iso()
    saved = get_repository().upsert_item(existing)
    get_repository().add_audit_log(
        "re_embedding",
        {"modelId": MODEL_NAME, "modelRevision": MODEL_REVISION},
        item_id=item_id,
        request_id=get_request_id(request),
        tenant_id=identity.tenant_id,
    )
    return with_request_id(request, {"item": serialize_item(saved)})


@app.get("/audit-logs")
def list_audit_logs(
    request: Request,
    itemId: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    identity = authorize(request, "corpus:read")
    logs = get_repository().list_audit_logs(identity, item_id=itemId, limit=limit, offset=offset)
    return with_request_id(request, {"logs": logs})


@app.post("/match", response_model=MatchResponseModel)
async def match(request: Request):
    started = time.time()
    endpoint = "/match"
    identity = authorize(request, "match:execute")
    form = await request.form()

    upload_value = first_value(form, "file", "image")
    upload = upload_value if is_upload(upload_value) else None
    parsed_request = parse_match_request(form)
    if upload is None and not parsed_request.text:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("invalid_match_request", "Provide an image file, text query, or both."),
        )

    query_embeddings: List[List[float]] = []
    query_image_info = None
    raw_bytes = None
    query_ocr_payload = None
    query_barcode_payload = None
    query_ocr_error = None
    query_barcode_error = None

    if upload is not None:
        raw_bytes, pil_image = await read_image_upload(upload, "file")
        query_embeddings.append(image_embedding_for(pil_image))
        query_image_info = image_info(pil_image, upload)
        if parsed_request.doOcr:
            query_ocr_payload, query_ocr_error = await run_ocr(pil_image, parsed_request.ocrLang, parsed_request.ocrPsm)
        if parsed_request.doBarcode:
            query_barcode_payload, query_barcode_error = await run_barcode_scan(pil_image)

    if parsed_request.text:
        query_embeddings.append(text_embedding_for(parsed_request.text))

    query_vector = normalize_vectors(np.mean(np.asarray(query_embeddings, dtype=float), axis=0))[0]
    scanned_barcodes = normalize_barcodes((query_barcode_payload or {}).get("barcodes"))
    combined_barcodes = merge_barcodes(normalize_barcodes(parsed_request.barcodeValues), scanned_barcodes)
    query_barcode_values = barcode_values(combined_barcodes)
    query_ocr_text = parsed_request.ocrText or ((query_ocr_payload or {}).get("fullText"))

    if parsed_request.status == "released":
        candidates = get_repository().list_released_items(identity, limit=CORPUS_LIMIT)
    else:
        candidates = []

    ranked: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.get("embedding"):
            continue
        candidate_vector = normalize_vectors(candidate["embedding"])[0]
        score = float(np.dot(query_vector, candidate_vector))
        ranked.append({"candidate": candidate, "score": score})

    ranked.sort(key=lambda row: row["score"], reverse=True)
    top_rows = ranked[: parsed_request.topK]
    scores = [row["score"] for row in top_rows]
    confidences = softmax_confidences(scores, temperature=parsed_request.thresholds.temperature)
    top_score = scores[0] if scores else 0.0
    second_score = scores[1] if len(scores) > 1 else None
    no_match, no_match_meta = should_return_no_match(
        top_score,
        second_score,
        parsed_request.thresholds.minScore,
        parsed_request.thresholds.minMargin,
    )

    results: List[Dict[str, Any]] = []
    if not no_match:
        for row, confidence in zip(top_rows, confidences):
            candidate = row["candidate"]
            score = row["score"]
            results.append(
                {
                    "candidateId": candidate["id"],
                    "item": serialize_item(candidate, include_embedding=False),
                    "score": round(score, 6),
                    "confidence": round(float(confidence), 6),
                    "explanation": match_explanation(candidate, query_ocr_text, combined_barcodes, parsed_request.labels, score),
                }
            )

    get_repository().add_audit_log(
        "match_execution",
        {
            "topK": parsed_request.topK,
            "statusFilter": "released",
            "queryTextPresent": bool(parsed_request.text),
            "resultIds": [result["item"]["id"] for result in results],
            "decisioning": {"noMatch": no_match, "meta": no_match_meta},
            "input": None if raw_bytes is None else {"sha256": sha256_hex(raw_bytes), "bytes": len(raw_bytes)},
        },
        request_id=get_request_id(request),
        tenant_id=identity.tenant_id,
    )
    if query_ocr_error:
        get_repository().add_audit_log(
            "ocr_failure",
            {"reason": query_ocr_error, "inputSha256": sha256_hex(raw_bytes or b"")},
            request_id=get_request_id(request),
            tenant_id=identity.tenant_id,
        )

    query_type_parts: List[str] = []
    if upload is not None:
        query_type_parts.append("image")
    if parsed_request.text:
        query_type_parts.append("text")

    decision = {
        "noMatch": no_match,
        "reason": str(no_match_meta.get("reason") or ("OK" if not no_match else "NO_MATCH")),
        "details": no_match_meta,
    }
    match_outcome = "no_match" if no_match else "matched"
    record_match_outcome(match_outcome)
    latency_ms = request_latency_ms(request)
    logger.info(
        "Match completed request_id=%s outcome=%s candidates=%s top_score=%.6f latency_ms=%s",
        get_request_id(request),
        match_outcome,
        len(results),
        top_score,
        latency_ms,
    )

    response_data = with_request_id(
        request,
        {
            "governance": {
                **governance_meta(),
                "scoringVersion": SCORING_VERSION,
                "thresholds": parsed_request.thresholds.model_dump(mode="json"),
                "releasedOnly": parsed_request.status == "released",
                "topKRequested": parsed_request.topK,
                "candidatesEvaluated": len(ranked),
                "latencyMs": latency_ms,
                "decision": decision,
            },
            "query": {
                "type": "+".join(query_type_parts),
                "modalities": query_type_parts,
                "text": parsed_request.text,
                "image": query_image_info,
                "metadata": parsed_request.metadata,
                "thresholds": parsed_request.thresholds.model_dump(mode="json"),
                "ocrText": query_ocr_text,
                "ocr": {
                    "fullText": (query_ocr_payload or {}).get("fullText"),
                    "words": (query_ocr_payload or {}).get("words") or [],
                    "meta": (query_ocr_payload or {}).get("meta"),
                    "error": query_ocr_error,
                },
                "barcodeValues": query_barcode_values,
                "barcode": {
                    "barcodes": combined_barcodes,
                    "meta": (query_barcode_payload or {}).get("meta"),
                    "error": query_barcode_error,
                },
                "labels": parsed_request.labels,
                "signals": compact_signals(query_ocr_text, query_barcode_values, parsed_request.labels),
            },
            "decision": decision,
            "topK": results,
        },
    )
    try:
        response = MatchResponseModel.model_validate(response_data)
    except ValidationError as exc:
        logger.exception("Match response schema validation failed")
        record_match_failure("response_schema_invalid")
        raise HTTPException(
            status_code=500,
            detail=problem_detail("response_schema_invalid", "Match response schema validation failed.", exc.errors()),
        ) from exc
    record_latency_metric(endpoint, started)
    record_request_metric(endpoint, 200)
    return response
