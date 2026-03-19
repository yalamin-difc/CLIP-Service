from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError

try:  # pragma: no cover - optional runtime dependency
    from barcode_service import scan_barcodes as _scan_barcodes
except Exception:  # pragma: no cover
    _scan_barcodes = None

try:  # pragma: no cover - optional runtime dependency
    from ocr_service import extract_ocr as _extract_ocr
except Exception:  # pragma: no cover
    _extract_ocr = None

try:  # pragma: no cover - optional runtime dependency
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = Gauge = Histogram = None
    generate_latest = None

app = FastAPI(title="CLIP Service")

ALLOWED_ORIGINS = [
    "https://ailostfound.al-amentech.io",
    "https://openailostfound-fccdb129f869.herokuapp.com",
    "http://localhost:3000",
]
ALLOW_ORIGIN_REGEX = r"^https:\/\/ai-lost-and-found-ver-2-.*\.vercel\.app$"
REQUEST_ID_HEADER = "X-Request-Id"
MODEL_NAME = os.environ.get("MODEL_ID", "openai/clip-vit-base-patch32").strip() or "openai/clip-vit-base-patch32"
SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "dev").strip() or "dev"
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
CLIP_API_KEY = os.environ.get("CLIP_API_KEY", "").strip()
MONGODB_URI = (os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI") or "").strip()
MONGODB_DB = os.environ.get("MONGODB_DB", "clip_service").strip() or "clip_service"
MONGODB_ITEMS_COLLECTION = os.environ.get("MONGODB_COLLECTION", "items").strip() or "items"
MONGODB_AUDIT_COLLECTION = os.environ.get("MONGODB_AUDIT_COLLECTION", "audit_logs").strip() or "audit_logs"
CONF_TEMPERATURE = float(os.environ.get("CONF_TEMPERATURE", "0.07"))
CONF_MIN_SCORE = float(os.environ.get("CONF_MIN_SCORE", "0.22"))
CONF_MIN_MARGIN = float(os.environ.get("CONF_MIN_MARGIN", "0.03"))

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
else:  # pragma: no cover
    METRIC_REQUESTS = None
    METRIC_LATENCY = None
    METRIC_MODEL_LOADED = None


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
    response = await call_next(request)
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
    return error_response(request, exc.status_code, code, message, details)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return error_response(request, 422, "validation_error", "Request validation failed.", exc.errors())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return error_response(request, 500, "internal_error", "An unexpected error occurred.")


def record_request_metric(endpoint: str, status: int) -> None:
    if METRIC_REQUESTS is not None:
        METRIC_REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()


def record_latency_metric(endpoint: str, started_at: float) -> None:
    if METRIC_LATENCY is not None:
        METRIC_LATENCY.labels(endpoint=endpoint).observe(max(0.0, time.time() - started_at))


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


def require_auth(authorization: Optional[str]) -> None:
    if not CLIP_API_KEY:
        return
    if authorization != f"Bearer {CLIP_API_KEY}":
        raise HTTPException(status_code=401, detail=problem_detail("unauthorized", "Unauthorized"))


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
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
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


def load_model():
    global model, processor
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
                model = CLIPModel.from_pretrained(MODEL_NAME)
                processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def model_health() -> Dict[str, Any]:
    return {"name": MODEL_NAME, "loaded": model is not None and processor is not None}


async def read_image_upload(upload: Optional[UploadFile], field_name: str) -> tuple[bytes, Image.Image]:
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("missing_image", f"Missing image upload in field '{field_name}'."),
        )
    try:
        payload = await upload.read()
    finally:
        await upload.close()
    if not payload:
        raise HTTPException(status_code=400, detail=problem_detail("invalid_image", "Uploaded image is empty."))
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
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
    model_instance, processor_instance = load_model()
    try:
        inputs = processor_instance(images=image, return_tensors="pt")
        with inference_mode():
            features = model_instance.get_image_features(**inputs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=problem_detail("model_inference_failed", "Failed to encode image.")) from exc
    return [float(value) for value in normalize_vectors(features)[0].tolist()]


def text_embedding_for(text: str) -> List[float]:
    model_instance, processor_instance = load_model()
    try:
        inputs = processor_instance(text=[text], return_tensors="pt", padding=True)
        with inference_mode():
            features = model_instance.get_text_features(**inputs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=problem_detail("model_inference_failed", "Failed to encode text.")) from exc
    return [float(value) for value in normalize_vectors(features)[0].tolist()]


def run_ocr(image: Image.Image, lang: str, psm: int) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    if _extract_ocr is None:
        return None, "OCR service is not available."
    try:
        return _extract_ocr(image, lang=lang, psm=int(psm)), None
    except Exception as exc:  # pragma: no cover - depends on optional runtime tools
        return None, str(exc)


def run_barcode_scan(image: Image.Image) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    if _scan_barcodes is None:
        return None, "Barcode service is not available."
    try:
        return _scan_barcodes(image), None
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
    query_barcode_map = {value.lower(): value for value in barcode_values(query_barcodes)}
    item_barcode_map = {value.lower(): value for value in barcode_values(normalize_barcodes(item.get("barcodes") or item.get("barcodeValues")))}
    barcode_overlap = [item_barcode_map[key] for key in sorted(query_barcode_map.keys() & item_barcode_map.keys())][:3]

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
    return {
        "reason": ", ".join(reasons),
        "signals": signals,
        "scoreBand": "high" if score >= 0.85 else "medium" if score >= 0.65 else "low",
    }


class ItemRepository:
    def health(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def list_items(self, status: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def list_released_items(self, limit: int = 5000) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def add_audit_log(self, event_type: str, payload: Dict[str, Any], item_id: Optional[str] = None, request_id: Optional[str] = None) -> str:
        raise NotImplementedError

    def list_audit_logs(self, item_id: Optional[str] = None, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
        raise NotImplementedError


class InMemoryItemRepository(ItemRepository):
    def __init__(self):
        self._items: Dict[str, Dict[str, Any]] = {}
        self._logs: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def health(self) -> Dict[str, Any]:
        return {"backend": "memory", "configured": False, "ok": True, "items": len(self._items)}

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._items.get(item_id)
            return dict(item) if item else None

    def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            stored = dict(item)
            self._items[item["id"]] = stored
            return dict(stored)

    def list_items(self, status: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        with self._lock:
            items = [dict(value) for value in self._items.values()]
        if status:
            items = [item for item in items if item.get("status") == status]
        items.sort(key=lambda item: item.get("updatedAt") or "", reverse=True)
        return items[:limit]

    def list_released_items(self, limit: int = 5000) -> List[Dict[str, Any]]:
        with self._lock:
            items = [
                dict(value)
                for value in self._items.values()
                if value.get("released") and value.get("eligibleForMatching")
            ]
        items.sort(key=lambda item: item.get("updatedAt") or "", reverse=True)
        return items[:limit]

    def add_audit_log(self, event_type: str, payload: Dict[str, Any], item_id: Optional[str] = None, request_id: Optional[str] = None) -> str:
        with self._lock:
            log_id = str(uuid.uuid4())
            self._logs.append(
                {
                    "id": log_id,
                    "ts": now_iso(),
                    "eventType": event_type,
                    "itemId": item_id,
                    "requestId": request_id,
                    "payload": payload or {},
                }
            )
            return log_id

    def list_audit_logs(self, item_id: Optional[str] = None, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            logs = list(reversed(self._logs))
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

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        self._raise()

    def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        self._raise()

    def list_items(self, status: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        self._raise()

    def list_released_items(self, limit: int = 5000) -> List[Dict[str, Any]]:
        self._raise()

    def add_audit_log(self, event_type: str, payload: Dict[str, Any], item_id: Optional[str] = None, request_id: Optional[str] = None) -> str:
        self._raise()

    def list_audit_logs(self, item_id: Optional[str] = None, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
        self._raise()


class MongoItemRepository(ItemRepository):
    def __init__(self, uri: str, database: str, items_collection: str, audit_collection: str):
        from pymongo import MongoClient

        self._client = MongoClient(uri, serverSelectionTimeoutMS=1000)
        self._items = self._client[database][items_collection]
        self._audit = self._client[database][audit_collection]
        self._items.create_index("id", unique=True)
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

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        return self._clean_item(self._items.find_one({"id": item_id}))

    def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        self._items.update_one({"id": item["id"]}, {"$set": item}, upsert=True)
        stored = self.get_item(item["id"])
        if stored is None:
            raise HTTPException(status_code=503, detail=problem_detail("store_unavailable", "Failed to read item after upsert."))
        return stored

    def list_items(self, status: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if status:
            query["status"] = status
        cursor = self._items.find(query).sort("updatedAt", -1).limit(limit)
        return [self._clean_item(item) for item in cursor if item is not None]

    def list_released_items(self, limit: int = 5000) -> List[Dict[str, Any]]:
        cursor = self._items.find({"released": True, "eligibleForMatching": True}).sort("updatedAt", -1).limit(limit)
        return [self._clean_item(item) for item in cursor if item is not None]

    def add_audit_log(self, event_type: str, payload: Dict[str, Any], item_id: Optional[str] = None, request_id: Optional[str] = None) -> str:
        document = {
            "ts": now_iso(),
            "eventType": event_type,
            "itemId": item_id,
            "requestId": request_id,
            "payload": payload or {},
        }
        return str(self._audit.insert_one(document).inserted_id)

    def list_audit_logs(self, item_id: Optional[str] = None, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
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
    if not MONGODB_URI:
        return InMemoryItemRepository()
    try:
        return MongoItemRepository(MONGODB_URI, MONGODB_DB, MONGODB_ITEMS_COLLECTION, MONGODB_AUDIT_COLLECTION)
    except Exception as exc:  # pragma: no cover - depends on live mongo
        return UnavailableItemRepository(f"Mongo repository unavailable: {exc}")


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
    }
    if include_embedding and item.get("embedding") is not None:
        payload["clipEmbedding"] = item.get("embedding")
    return payload


def parse_item_payload(raw_payload: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(raw_payload)
    nested = payload.get("item")
    if isinstance(nested, Mapping):
        payload = {**nested, **{key: value for key, value in payload.items() if key != "item"}}

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
    if "status" in payload:
        item["status"] = compact_text(str(payload.get("status")), limit=32)
    if "released" in payload:
        item["released"] = normalize_bool(payload.get("released"), default=False)
    if "embedding" in payload or "clipEmbedding" in payload:
        embedding = normalize_embedding(first_value(payload, "embedding", "clipEmbedding"))
        item["embedding"] = embedding
        item["clipEmbedding"] = embedding

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


def prepare_item_for_storage(existing: Optional[Dict[str, Any]], patch: Dict[str, Any]) -> Dict[str, Any]:
    current = dict(existing or {})
    current.update(patch)
    current["id"] = patch["id"]
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
        current["clipEmbedding"] = current["embedding"]
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
def metrics():
    if generate_latest is None:
        raise HTTPException(status_code=503, detail=problem_detail("metrics_unavailable", "Metrics not available."))
    if METRIC_MODEL_LOADED is not None:
        METRIC_MODEL_LOADED.set(1 if model is not None and processor is not None else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def home(request: Request):
    return with_request_id(request, {"status": "running", "modelLoaded": model is not None and processor is not None})


@app.get("/health")
def health(request: Request):
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


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(UI_HTML)


@app.post("/encode-image")
async def encode_image(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    authorization: Optional[str] = Header(default=None),
):
    started = time.time()
    endpoint = "/encode-image"
    require_auth(authorization)
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail=problem_detail("missing_image", "No image uploaded. Use form field 'file' or 'image'."))
    _, pil_image = await read_image_upload(upload, "file")
    response = with_request_id(request, {"embedding": image_embedding_for(pil_image)})
    record_latency_metric(endpoint, started)
    record_request_metric(endpoint, 200)
    return response


@app.post("/encode-text")
async def encode_text(
    request: Request,
    text: Optional[str] = Form(default=None),
    queryText: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
):
    started = time.time()
    endpoint = "/encode-text"
    require_auth(authorization)
    value = compact_text(text or queryText, limit=2000)
    if not value:
        raise HTTPException(status_code=400, detail=problem_detail("missing_text", "No text provided. Use form field 'text' or 'queryText'."))
    response = with_request_id(request, {"embedding": text_embedding_for(value)})
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
    authorization: Optional[str] = Header(default=None),
):
    started = time.time()
    endpoint = "/similarity"
    require_auth(authorization)
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
    doOcr: bool = Form(default=True),
    doBarcode: bool = Form(default=True),
    ocrLang: str = Form(default="eng"),
    ocrPsm: int = Form(default=6),
    authorization: Optional[str] = Header(default=None),
):
    started = time.time()
    endpoint = "/analyze-image"
    require_auth(authorization)
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
    except Exception:
        pass

    ocr_payload = None
    ocr_error = None
    if doOcr:
        ocr_payload, ocr_error = run_ocr(pil_image, ocrLang, ocrPsm)

    barcode_payload = None
    barcode_error = None
    if doBarcode:
        barcode_payload, barcode_error = run_barcode_scan(pil_image)

    scanned_barcodes = normalize_barcodes((barcode_payload or {}).get("barcodes"))
    combined_barcodes = merge_barcodes(manual_barcodes, scanned_barcodes)
    effective_ocr_text = manual_ocr_text or ((ocr_payload or {}).get("fullText"))
    effective_barcode_values = barcode_values(combined_barcodes)

    get_repository().add_audit_log(
        "IMAGE_ANALYZE",
        {
            "input": {"sha256": sha256_hex(raw), "bytes": len(raw), "contentType": getattr(upload, "content_type", None)},
            "ocrEnabled": doOcr,
            "barcodeEnabled": doBarcode,
            "ocrError": ocr_error,
            "barcodeError": barcode_error,
        },
        request_id=request_id,
    )

    response = with_request_id(
        request,
        {
            "governance": governance_meta(ocrEnabled=doOcr, barcodeEnabled=doBarcode),
            "image": image_info(pil_image, upload),
            "input": {"sha256": sha256_hex(raw), "bytes": len(raw), "contentType": getattr(upload, "content_type", None)},
            "embedding": image_embedding_for(pil_image),
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
    require_auth(request.headers.get("Authorization"))
    parsed_payload = await parse_items_request(request)
    upload = parsed_payload.pop("upload", None)
    normalized_patch = parse_item_payload(parsed_payload)

    existing = get_repository().get_item(normalized_patch["id"])
    stored_item = prepare_item_for_storage(existing, normalized_patch)

    if upload is not None:
        _, pil_image = await read_image_upload(upload, "file")
        stored_item["embedding"] = image_embedding_for(pil_image)
        stored_item["clipEmbedding"] = stored_item["embedding"]
        stored_item["image"] = image_info(pil_image, upload)
        stored_item["eligibleForMatching"] = bool(stored_item.get("released")) and True
        stored_item["status"] = "released" if stored_item["eligibleForMatching"] else "stored"

    saved_item = get_repository().upsert_item(stored_item)
    get_repository().add_audit_log(
        "ITEM_UPSERT",
        {"status": saved_item.get("status"), "released": saved_item.get("released"), "hasEmbedding": bool(saved_item.get("embedding"))},
        item_id=saved_item["id"],
        request_id=get_request_id(request),
    )

    response = with_request_id(request, {"governance": governance_meta(), "item": serialize_item(saved_item, include_embedding=True)})
    record_latency_metric(endpoint, started)
    record_request_metric(endpoint, 200)
    return response


@app.get("/items")
def list_items(
    request: Request,
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    items = [serialize_item(item, include_embedding=True) for item in get_repository().list_items(status=status, limit=limit)]
    return with_request_id(request, {"items": items})


@app.get("/items/{item_id}")
def get_item(request: Request, item_id: str, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    item = get_repository().get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=problem_detail("item_not_found", f"Item '{item_id}' was not found."))
    return with_request_id(request, {"item": serialize_item(item, include_embedding=True)})


@app.post("/items/{item_id}/release")
def release_item(request: Request, item_id: str, authorization: Optional[str] = Header(default=None)):
    started = time.time()
    endpoint = "/items/release"
    require_auth(authorization)
    existing = get_repository().get_item(item_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=problem_detail("item_not_found", f"Item '{item_id}' was not found."))
    if not existing.get("embedding"):
        raise HTTPException(
            status_code=409,
            detail=problem_detail("item_not_matchable", f"Item '{item_id}' cannot be released without an embedding or image."),
        )
    existing["released"] = True
    existing["eligibleForMatching"] = True
    existing["status"] = "released"
    existing["releasedAt"] = existing.get("releasedAt") or now_iso()
    existing["updatedAt"] = now_iso()
    saved_item = get_repository().upsert_item(existing)
    get_repository().add_audit_log(
        "ITEM_RELEASE",
        {"status": "released", "releasedAt": saved_item.get("releasedAt")},
        item_id=item_id,
        request_id=get_request_id(request),
    )
    response = with_request_id(request, {"item": serialize_item(saved_item, include_embedding=True)})
    record_latency_metric(endpoint, started)
    record_request_metric(endpoint, 200)
    return response


@app.get("/audit-logs")
def list_audit_logs(
    request: Request,
    itemId: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    logs = get_repository().list_audit_logs(item_id=itemId, limit=limit, offset=offset)
    return with_request_id(request, {"logs": logs})


@app.post("/match")
async def match(request: Request):
    started = time.time()
    endpoint = "/match"
    require_auth(request.headers.get("Authorization"))
    form = await request.form()

    upload_value = first_value(form, "file", "image")
    upload = upload_value if is_upload(upload_value) else None
    text = compact_text(first_value(form, "text", "queryText", "query"), limit=2000)
    if upload is None and not text:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("invalid_match_request", "Provide an image file, text query, or both."),
        )

    do_ocr = normalize_bool(form.get("doOcr"), default=True)
    do_barcode = normalize_bool(form.get("doBarcode"), default=True)
    ocr_lang = compact_text(str(form.get("ocrLang") or "eng"), limit=32) or "eng"
    ocr_psm = int(form.get("ocrPsm") or 6)
    top_k = parse_top_k(first_value(form, "topK", "k", "limit"))
    status_filter = compact_text(str(form.get("status") or "released"), limit=32) or "released"
    manual_ocr_text = compact_text(first_value(form, "ocrText", "ocr"), limit=500)
    manual_barcodes = normalize_barcodes(first_value(form, "barcodeValues", "barcodes"))
    manual_labels = normalize_string_list(first_value(form, "labels", "tags"))

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
        if do_ocr:
            query_ocr_payload, query_ocr_error = run_ocr(pil_image, ocr_lang, ocr_psm)
        if do_barcode:
            query_barcode_payload, query_barcode_error = run_barcode_scan(pil_image)

    if text:
        query_embeddings.append(text_embedding_for(text))

    query_vector = normalize_vectors(np.mean(np.asarray(query_embeddings, dtype=float), axis=0))[0]
    scanned_barcodes = normalize_barcodes((query_barcode_payload or {}).get("barcodes"))
    combined_barcodes = merge_barcodes(manual_barcodes, scanned_barcodes)
    query_barcode_values = barcode_values(combined_barcodes)
    query_ocr_text = manual_ocr_text or ((query_ocr_payload or {}).get("fullText"))

    if status_filter == "released":
        candidates = get_repository().list_released_items(limit=5000)
    else:
        candidates = [item for item in get_repository().list_items(status=status_filter, limit=5000) if item.get("embedding")]

    ranked: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.get("embedding"):
            continue
        candidate_vector = normalize_vectors(candidate["embedding"])[0]
        score = float(np.dot(query_vector, candidate_vector))
        ranked.append({"candidate": candidate, "score": score})

    ranked.sort(key=lambda row: row["score"], reverse=True)
    top_rows = ranked[:top_k]
    scores = [row["score"] for row in top_rows]
    confidences = softmax_confidences(scores, temperature=CONF_TEMPERATURE)
    top_score = scores[0] if scores else 0.0
    second_score = scores[1] if len(scores) > 1 else None
    no_match, no_match_meta = should_return_no_match(top_score, second_score, CONF_MIN_SCORE, CONF_MIN_MARGIN)

    results: List[Dict[str, Any]] = []
    if not no_match:
        for row, confidence in zip(top_rows, confidences):
            candidate = row["candidate"]
            score = row["score"]
            results.append(
                {
                    "item": serialize_item(candidate, include_embedding=False),
                    "score": round(score, 6),
                    "confidence": round(float(confidence), 6),
                    "explanation": match_explanation(candidate, query_ocr_text, combined_barcodes, manual_labels, score),
                }
            )

    get_repository().add_audit_log(
        "AI_MATCH_GENERATION",
        {
            "topK": top_k,
            "statusFilter": status_filter,
            "queryText": text,
            "resultIds": [result["item"]["id"] for result in results],
            "decisioning": {"noMatch": no_match, "meta": no_match_meta},
            "input": None if raw_bytes is None else {"sha256": sha256_hex(raw_bytes), "bytes": len(raw_bytes)},
        },
        request_id=get_request_id(request),
    )

    query_type_parts: List[str] = []
    if upload is not None:
        query_type_parts.append("image")
    if text:
        query_type_parts.append("text")

    response = with_request_id(
        request,
        {
            "governance": governance_meta(
                releasedOnly=status_filter == "released",
                topKRequested=top_k,
                candidatesEvaluated=len(ranked),
                decisioning={"noMatch": no_match, "meta": no_match_meta},
            ),
            "query": {
                "type": "+".join(query_type_parts),
                "text": text,
                "image": query_image_info,
                "ocr": query_ocr_payload,
                "ocrError": query_ocr_error,
                "barcode": {"barcodes": combined_barcodes, "meta": (query_barcode_payload or {}).get("meta")} if combined_barcodes or query_barcode_payload else None,
                "barcodeError": query_barcode_error,
                "signals": compact_signals(query_ocr_text, query_barcode_values, manual_labels),
            },
            "topK": results,
        },
    )
    record_latency_metric(endpoint, started)
    record_request_metric(endpoint, 200)
    return response
