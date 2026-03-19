import io
import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="CLIP Service")

ALLOWED_ORIGINS = [
    "https://ailostfound.al-amentech.io",
    "https://openailostfound-fccdb129f869.herokuapp.com",
    "http://localhost:3000",
]
ALLOW_ORIGIN_REGEX = r"^https:\/\/ai-lost-and-found-ver-2-.*\.vercel\.app$"
REQUEST_ID_HEADER = "X-Request-Id"
MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
CLIP_API_KEY = os.environ.get("CLIP_API_KEY", "").strip()
MONGODB_URI = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI") or ""
MONGODB_DB = os.environ.get("MONGODB_DB", "clip_service")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION", "items")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        code = default_error_code(exc.status_code)
        message = "Request failed."
        details = exc.detail

    return error_response(request, exc.status_code, code, message, details)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return error_response(
        request,
        422,
        "validation_error",
        "Request validation failed.",
        exc.errors(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return error_response(
        request,
        500,
        "internal_error",
        "An unexpected error occurred.",
    )


def require_auth(authorization: Optional[str]) -> None:
    if not CLIP_API_KEY:
        return
    if authorization != f"Bearer {CLIP_API_KEY}":
        raise HTTPException(status_code=401, detail=problem_detail("unauthorized", "Unauthorized"))


def is_upload(value: Any) -> bool:
    return hasattr(value, "filename") and hasattr(value, "read")


def compact_text(value: Optional[str], limit: int = 120) -> Optional[str]:
    if value is None:
        return None
    collapsed = " ".join(str(value).split())
    if not collapsed:
        return None
    return collapsed[: limit - 3] + "..." if len(collapsed) > limit else collapsed


def normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    raw_items: List[Any]
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, tuple):
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
        candidate = str(item).strip()
        key = candidate.lower()
        if candidate and key not in seen:
            normalized.append(candidate)
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
    raise HTTPException(
        status_code=400,
        detail=problem_detail("invalid_payload", f"Field '{field_name}' must be an object."),
    )


def normalize_embedding(value: Any) -> Optional[List[float]]:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=problem_detail("invalid_embedding", "Embedding must be valid JSON."),
            ) from exc

    vector = normalize_vectors(to_numpy(value))[0]
    return [float(component) for component in vector.tolist()]


def normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise HTTPException(
        status_code=400,
        detail=problem_detail("invalid_boolean", "Boolean field contains an invalid value."),
    )


def parse_top_k(value: Any) -> int:
    if value is None or value == "":
        return DEFAULT_TOP_K
    try:
        top_k = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("invalid_top_k", "Field 'topK' must be an integer."),
        ) from exc
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("invalid_top_k", f"Field 'topK' must be between 1 and {MAX_TOP_K}."),
        )
    return top_k


def first_value(container: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = container.get(key)
        if value not in (None, ""):
            return value
    return None


def serialize_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    public_item = dict(item)
    public_item.pop("embedding", None)
    return public_item


def compact_signals(ocr_text: Optional[str], barcode_values: List[str], labels: List[str]) -> Dict[str, Any]:
    signals: Dict[str, Any] = {}
    excerpt = compact_text(ocr_text)
    if excerpt:
        signals["ocr"] = {"excerpt": excerpt}
    if barcode_values:
        signals["barcode"] = {"count": len(barcode_values), "values": barcode_values[:3]}
    if labels:
        signals["labels"] = labels[:5]
    return signals


def tokenize(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return sorted({token.lower() for token in re.findall(r"[A-Za-z0-9]{3,}", value)})


def match_explanation(item: Mapping[str, Any], query_bundle: Mapping[str, Any], score: float) -> Dict[str, Any]:
    query_barcodes = {value.lower(): value for value in query_bundle.get("barcodeValues", [])}
    item_barcodes = {value.lower(): value for value in normalize_string_list(item.get("barcodeValues"))}
    barcode_overlap = [item_barcodes[key] for key in sorted(query_barcodes.keys() & item_barcodes.keys())][:3]

    query_ocr_tokens = set(query_bundle.get("ocrTokens", []))
    item_ocr_tokens = set(tokenize(item.get("ocrText")))
    ocr_overlap = sorted(query_ocr_tokens & item_ocr_tokens)[:5]

    query_labels = {value.lower(): value for value in query_bundle.get("labels", [])}
    item_labels = {value.lower(): value for value in normalize_string_list(item.get("labels"))}
    label_overlap = [item_labels[key] for key in sorted(query_labels.keys() & item_labels.keys())][:5]

    signals: Dict[str, Any] = {}
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


@contextmanager
def inference_mode():
    try:
        import torch
    except ImportError:
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
                        detail=problem_detail(
                            "model_unavailable",
                            "CLIP model dependencies are not installed.",
                        ),
                    ) from exc
                model = CLIPModel.from_pretrained(MODEL_NAME)
                processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def model_health() -> Dict[str, Any]:
    return {
        "name": MODEL_NAME,
        "loaded": model is not None and processor is not None,
    }


def image_info(image: Image.Image, upload: Optional[UploadFile] = None) -> Dict[str, Any]:
    return {
        "filename": getattr(upload, "filename", None),
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
    }


async def read_image_upload(upload: Optional[UploadFile], field_name: str) -> Image.Image:
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
        raise HTTPException(
            status_code=400,
            detail=problem_detail("invalid_image", "Uploaded image is empty."),
        )
    try:
        return Image.open(io.BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("invalid_image", "Uploaded file is not a valid image."),
        ) from exc


def image_embedding_for(image: Image.Image) -> List[float]:
    model_instance, processor_instance = load_model()
    try:
        inputs = processor_instance(images=image, return_tensors="pt")
        with inference_mode():
            features = model_instance.get_image_features(**inputs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=problem_detail("model_inference_failed", "Failed to encode image."),
        ) from exc
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
        raise HTTPException(
            status_code=503,
            detail=problem_detail("model_inference_failed", "Failed to encode text."),
        ) from exc
    return [float(value) for value in normalize_vectors(features)[0].tolist()]


class ItemRepository:
    def health(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def list_released_items(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class InMemoryItemRepository(ItemRepository):
    def __init__(self):
        self._items: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def health(self) -> Dict[str, Any]:
        return {
            "backend": "memory",
            "configured": False,
            "ok": True,
            "items": len(self._items),
        }

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._items.get(item_id)
            return dict(item) if item else None

    def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            stored = dict(item)
            self._items[item["id"]] = stored
            return dict(stored)

    def list_released_items(self) -> List[Dict[str, Any]]:
        with self._lock:
            items = []
            for value in self._items.values():
                if value.get("released") and value.get("eligibleForMatching"):
                    items.append(dict(value))
            return items


class UnavailableItemRepository(ItemRepository):
    def __init__(self, reason: str):
        self.reason = reason

    def health(self) -> Dict[str, Any]:
        return {
            "backend": "mongo",
            "configured": True,
            "ok": False,
            "error": self.reason,
        }

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        raise HTTPException(
            status_code=503,
            detail=problem_detail("store_unavailable", self.reason),
        )

    def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        raise HTTPException(
            status_code=503,
            detail=problem_detail("store_unavailable", self.reason),
        )

    def list_released_items(self) -> List[Dict[str, Any]]:
        raise HTTPException(
            status_code=503,
            detail=problem_detail("store_unavailable", self.reason),
        )


class MongoItemRepository(ItemRepository):
    def __init__(self, uri: str, database: str, collection: str):
        from pymongo import MongoClient

        self._client = MongoClient(uri, serverSelectionTimeoutMS=1000)
        self._collection = self._client[database][collection]

    def health(self) -> Dict[str, Any]:
        try:
            self._client.admin.command("ping")
            return {
                "backend": "mongo",
                "configured": True,
                "ok": True,
            }
        except Exception as exc:
            return {
                "backend": "mongo",
                "configured": True,
                "ok": False,
                "error": str(exc),
            }

    def _clean(self, item: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if item is None:
            return None
        cleaned = dict(item)
        cleaned.pop("_id", None)
        return cleaned

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        return self._clean(self._collection.find_one({"id": item_id}))

    def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        self._collection.update_one({"id": item["id"]}, {"$set": item}, upsert=True)
        stored = self.get_item(item["id"])
        if stored is None:
            raise HTTPException(
                status_code=503,
                detail=problem_detail("store_unavailable", "Failed to read item after upsert."),
            )
        return stored

    def list_released_items(self) -> List[Dict[str, Any]]:
        cursor = self._collection.find({"released": True, "eligibleForMatching": True})
        return [self._clean(item) for item in cursor if item is not None]


repository_lock = threading.Lock()
repository: Optional[ItemRepository] = None


def build_repository() -> ItemRepository:
    if not MONGODB_URI:
        return InMemoryItemRepository()
    try:
        return MongoItemRepository(MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION)
    except Exception as exc:
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


def parse_item_payload(raw_payload: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(raw_payload)
    nested = payload.get("item")
    if isinstance(nested, Mapping):
        payload = {**nested, **{key: value for key, value in payload.items() if key != "item"}}

    item_id = str(first_value(payload, "id", "itemId") or "").strip()
    if not item_id:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("missing_item_id", "Item id is required."),
        )

    item: Dict[str, Any] = {"id": item_id}

    if "title" in payload or "name" in payload:
        item["title"] = compact_text(first_value(payload, "title", "name"), limit=240)
    if "description" in payload:
        item["description"] = compact_text(first_value(payload, "description"), limit=500)
    if "category" in payload or "type" in payload:
        item["category"] = compact_text(first_value(payload, "category", "type"), limit=120)
    if "ocrText" in payload or "ocr" in payload:
        item["ocrText"] = compact_text(first_value(payload, "ocrText", "ocr"), limit=500)
    if "barcodeValues" in payload or "barcodes" in payload:
        item["barcodeValues"] = normalize_string_list(first_value(payload, "barcodeValues", "barcodes"))
    if "labels" in payload or "tags" in payload:
        item["labels"] = normalize_string_list(first_value(payload, "labels", "tags"))
    if "metadata" in payload:
        item["metadata"] = normalize_mapping(payload.get("metadata"), "metadata")
    if "embedding" in payload:
        item["embedding"] = normalize_embedding(payload.get("embedding"))
    if "released" in payload:
        item["released"] = normalize_bool(payload.get("released"), default=False)

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
            "category",
            "type",
            "ocrText",
            "ocr",
            "barcodeValues",
            "barcodes",
            "labels",
            "tags",
            "metadata",
            "embedding",
            "released",
        }
    }
    if extras:
        item["metadata"] = {**item.get("metadata", {}), "extra": extras}
    return item


async def parse_items_request(request: Request) -> Dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    parsed_payload: Dict[str, Any]
    upload: Optional[UploadFile] = None

    if "application/json" in content_type:
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=problem_detail("invalid_json", "Request body must contain valid JSON."),
            ) from exc
        if not isinstance(body, Mapping):
            raise HTTPException(
                status_code=400,
                detail=problem_detail("invalid_payload", "Request body must be an object."),
            )
        parsed_payload = dict(body)
    else:
        form = await request.form()
        parsed_payload = {key: value for key, value in form.items() if not is_upload(value)}
        upload_value = first_value(form, "file", "image")
        upload = upload_value if is_upload(upload_value) else None
        item_blob = form.get("item")
        if item_blob:
            if isinstance(item_blob, str):
                try:
                    nested = json.loads(item_blob)
                except json.JSONDecodeError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=problem_detail("invalid_json", "Field 'item' must contain valid JSON."),
                    ) from exc
                if not isinstance(nested, Mapping):
                    raise HTTPException(
                        status_code=400,
                        detail=problem_detail("invalid_payload", "Field 'item' must contain an object."),
                    )
                parsed_payload = {**nested, **parsed_payload}
        if upload is not None:
            parsed_payload["upload"] = upload

    return parsed_payload


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
    <p>This service exposes CLIP encoding endpoints plus backend-compatible item sync and matching routes.</p>
    <ul>
      <li><code>GET /health</code> - dependency health</li>
      <li><code>POST /encode-image</code> - image embedding</li>
      <li><code>POST /encode-text</code> - text embedding</li>
      <li><code>POST /analyze-image</code> - compact image analysis</li>
      <li><code>POST /items</code> - upsert stored item</li>
      <li><code>POST /items/{id}/release</code> - release item for matching</li>
      <li><code>POST /match</code> - match image/text query against released items</li>
    </ul>
    <p>Every response includes the <code>X-Request-Id</code> header, and JSON responses include <code>requestId</code>.</p>
  </body>
</html>
"""


@app.get("/")
def home(request: Request):
    return with_request_id(
        request,
        {
            "status": "running",
            "modelLoaded": model is not None and processor is not None,
        },
    )


@app.get("/health")
def health(request: Request):
    repo_health = get_repository().health()
    overall_status = "ok" if repo_health.get("ok", False) else "degraded"
    return with_request_id(
        request,
        {
            "status": overall_status,
            "dependencies": {
                "model": model_health(),
                "store": repo_health,
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
    require_auth(authorization)
    upload = file or image
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("missing_image", "No image uploaded. Use form field 'file' or 'image'."),
        )
    pil_image = await read_image_upload(upload, "file")
    return with_request_id(request, {"embedding": image_embedding_for(pil_image)})


@app.post("/encode-text")
async def encode_text(
    request: Request,
    text: Optional[str] = Form(default=None),
    queryText: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    value = compact_text(text or queryText, limit=2000)
    if not value:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("missing_text", "No text provided. Use form field 'text' or 'queryText'."),
        )
    return with_request_id(request, {"embedding": text_embedding_for(value)})


@app.post("/analyze-image")
async def analyze_image(request: Request):
    require_auth(request.headers.get("Authorization"))
    form = await request.form()
    upload_value = first_value(form, "file", "image")
    upload = upload_value if is_upload(upload_value) else None
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail=problem_detail("missing_image", "No image uploaded. Use form field 'file' or 'image'."),
        )
    pil_image = await read_image_upload(upload, "file")
    ocr_text = compact_text(first_value(form, "ocrText", "ocr"), limit=500)
    barcode_values = normalize_string_list(first_value(form, "barcodeValues", "barcodes"))
    labels = normalize_string_list(first_value(form, "labels", "tags"))
    return with_request_id(
        request,
        {
            "image": image_info(pil_image, upload),
            "signals": compact_signals(ocr_text, barcode_values, labels),
            "embedding": image_embedding_for(pil_image),
        },
    )


@app.post("/items")
async def upsert_item(request: Request):
    require_auth(request.headers.get("Authorization"))
    parsed_payload = await parse_items_request(request)
    upload = parsed_payload.pop("upload", None)
    normalized_item = parse_item_payload(parsed_payload)

    existing = get_repository().get_item(normalized_item["id"]) or {}
    stored_item = {**existing, **normalized_item}
    stored_item["id"] = normalized_item["id"]
    stored_item.setdefault("title", None)
    stored_item.setdefault("description", None)
    stored_item.setdefault("category", None)
    stored_item.setdefault("ocrText", None)
    stored_item.setdefault("barcodeValues", [])
    stored_item.setdefault("labels", [])
    stored_item.setdefault("metadata", {})
    stored_item.setdefault("released", False)
    stored_item["createdAt"] = existing.get("createdAt", now_iso())
    stored_item["updatedAt"] = now_iso()

    if upload is not None:
        pil_image = await read_image_upload(upload, "file")
        stored_item["embedding"] = image_embedding_for(pil_image)
        stored_item["image"] = image_info(pil_image, upload)

    stored_item["eligibleForMatching"] = bool(stored_item.get("released")) and bool(stored_item.get("embedding"))
    stored_item["status"] = "released" if stored_item["eligibleForMatching"] else "stored"

    saved_item = get_repository().upsert_item(stored_item)
    return with_request_id(request, {"item": serialize_item(saved_item)})


@app.post("/items/{item_id}/release")
async def release_item(item_id: str, request: Request):
    require_auth(request.headers.get("Authorization"))
    existing = get_repository().get_item(item_id)
    if existing is None:
        raise HTTPException(
            status_code=404,
            detail=problem_detail("item_not_found", f"Item '{item_id}' was not found."),
        )
    if not existing.get("embedding"):
        raise HTTPException(
            status_code=409,
            detail=problem_detail(
                "item_not_matchable",
                f"Item '{item_id}' cannot be released without an embedding or image.",
            ),
        )
    existing["released"] = True
    existing["eligibleForMatching"] = True
    existing["status"] = "released"
    existing["releasedAt"] = existing.get("releasedAt", now_iso())
    existing["updatedAt"] = now_iso()
    saved_item = get_repository().upsert_item(existing)
    return with_request_id(request, {"item": serialize_item(saved_item)})


@app.post("/match")
async def match(request: Request):
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

    top_k = parse_top_k(first_value(form, "topK", "limit"))
    barcode_values = normalize_string_list(first_value(form, "barcodeValues", "barcodes"))
    labels = normalize_string_list(first_value(form, "labels", "tags"))
    ocr_text = compact_text(first_value(form, "ocrText", "ocr"), limit=500)

    query_embeddings = []
    query_type_parts = []
    query_image_info = None
    if upload is not None:
        pil_image = await read_image_upload(upload, "file")
        query_embeddings.append(image_embedding_for(pil_image))
        query_type_parts.append("image")
        query_image_info = image_info(pil_image, upload)
    if text:
        query_embeddings.append(text_embedding_for(text))
        query_type_parts.append("text")

    query_vector = normalize_vectors(np.mean(np.asarray(query_embeddings, dtype=float), axis=0))[0]
    query_bundle = {
        "ocrText": ocr_text,
        "ocrTokens": tokenize(ocr_text),
        "barcodeValues": barcode_values,
        "labels": labels,
    }

    released_items = get_repository().list_released_items()
    ranked: List[Dict[str, Any]] = []
    for candidate in released_items:
        embedding = candidate.get("embedding")
        if not embedding:
            continue
        candidate_vector = normalize_vectors(embedding)[0]
        score = float(np.dot(query_vector, candidate_vector))
        ranked.append(
            {
                "item": serialize_item(candidate),
                "score": round(score, 6),
                "confidence": round(max(0.0, min(1.0, (score + 1.0) / 2.0)), 6),
                "explanation": match_explanation(candidate, query_bundle, score),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected = ranked[:top_k]
    query: Dict[str, Any] = {
        "type": "+".join(query_type_parts),
        "signals": compact_signals(ocr_text, barcode_values, labels),
    }
    if text:
        query["text"] = text
    if query_image_info:
        query["image"] = query_image_info

    governance = {
        "engine": "clip",
        "model": MODEL_NAME,
        "releasedOnly": True,
        "topKRequested": top_k,
        "candidatesEvaluated": len(ranked),
    }
    return with_request_id(
        request,
        {
            "governance": governance,
            "query": query,
            "topK": selected,
        },
    )


@app.post("/similarity")
async def similarity(
    request: Request,
    file1: Optional[UploadFile] = File(default=None),
    file2: Optional[UploadFile] = File(default=None),
    image1: Optional[UploadFile] = File(default=None),
    image2: Optional[UploadFile] = File(default=None),
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    upload1 = file1 or image1
    upload2 = file2 or image2
    if upload1 is None or upload2 is None:
        raise HTTPException(
            status_code=400,
            detail=problem_detail(
                "missing_image",
                "Two images are required. Use fields file1/file2 or image1/image2.",
            ),
        )
    embedding_1 = normalize_vectors(image_embedding_for(await read_image_upload(upload1, "file1")))[0]
    embedding_2 = normalize_vectors(image_embedding_for(await read_image_upload(upload2, "file2")))[0]
    similarity_score = float(np.dot(embedding_1, embedding_2))
    return with_request_id(request, {"similarity": similarity_score})
