"""P14: public Dubai AI Festival live demo facade.

This module is the ONLY thing a festival visitor's browser ever talks to
directly. It never accepts tenantId/siteId/datasetVersion/demoData/
actions/engine model IDs from the client -- those are entirely
server-controlled, taken from demo_config.py. It never returns a JWT, a
raw embedding, or a Python traceback. It exists purely to forward a
citizen-safe subset of the already-protected /v2/match and /v2/ab/match
endpoints, authenticated with a short-lived (<=120s) internal identity
minted server-side for every request and discarded immediately after.

Fails closed: every handler starts with _require_demo_ready(), which is
true only when DEMO_FACADE_ENABLED=true AND every other required DEMO_*
value is actually configured (demo_config.demo_config_ready()). If that's
false, every /demo/* endpoint behaves as though it doesn't exist (404) --
this is a deliberate, testable equivalent of "only mounted when enabled"
(see docs/P14_FESTIVAL_LIVE_GUI.md for why routes are always registered
but gated at call time, the same pattern already used for SIGLIP2_ENABLED
in embedding_engines/siglip2_engine.py).

Does not duplicate model matching business logic: every real inference
and ranking decision still happens inside /v2/match and /v2/ab/match via
a real HTTP call over loopback (async httpx) -- this module only mints
the identity, forwards the request, and re-shapes the response into a
smaller, public-safe schema.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import demo_config
import internal_auth

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_ASSET_EXTENSIONS: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_ALLOWED_MODES = {"clip", "siglip2", "compare"}

_rate_limit_lock = threading.Lock()
_rate_limit_windows: Dict[str, Tuple[int, int]] = {}


def _app():
    import app as app_module  # noqa: PLC0415 - deferred to avoid a circular import at module load time

    return app_module


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail=_app().problem_detail("not_found", "Not found."))


def _require_demo_ready() -> None:
    if not demo_config.demo_config_ready():
        raise _not_found()


def _client_ip(request: Request) -> str:
    # Deliberately ignores X-Forwarded-For: this service has no
    # trusted-proxy configuration (see app.py's own rate limiter, which
    # makes the same choice), so that header is caller-controlled and a
    # trivial rate-limit bypass if trusted blindly.
    return request.client.host if request.client else "unknown"


def _enforce_demo_rate_limit(request: Request) -> None:
    client_key = _client_ip(request)
    window = int(time.time() // 60)
    with _rate_limit_lock:
        stored_window, count = _rate_limit_windows.get(client_key, (window, 0))
        if stored_window != window:
            stored_window, count = window, 0
        count += 1
        _rate_limit_windows[client_key] = (stored_window, count)
    if count > demo_config.DEMO_RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=_app().problem_detail("demo_rate_limit_exceeded", "Demo rate limit exceeded. Please try again shortly."),
        )


def _mint_demo_identity() -> Tuple[str, str]:
    """Mints a short-lived (<=120s), server-only internal identity scoped
    to the configured demo tenant/site/dataset. Returns (bearer_token,
    request_id) for exactly one forwarded call; the token is never
    returned to the browser and is discarded after this function's
    caller finishes using it."""
    request_id = str(uuid.uuid4())
    claims = {
        "iss": internal_auth.JWT_ISSUER,
        "aud": internal_auth.JWT_AUDIENCE,
        "service": "festival-demo-facade",
        "tenantId": demo_config.DEMO_TENANT_ID,
        "siteId": demo_config.DEMO_SITE_ID,
        "siteIds": [demo_config.DEMO_SITE_ID],
        "actions": ["match:execute"],
        "demoData": True,
        "datasetVersion": demo_config.DEMO_DATASET_VERSION,
        "requestId": request_id,
        "exp": int(time.time()) + demo_config.DEMO_JWT_TTL_SECONDS,
    }
    token = internal_auth.sign_internal_token(claims)
    return token, request_id


async def _forward(
    path: str, *, engine: Optional[str], file_field: Optional[Tuple[str, bytes, str]], text: Optional[str]
) -> Tuple[int, Dict[str, Any]]:
    """POSTs to one of this same process's own already-protected /v2
    endpoints over loopback HTTP, using a freshly minted, single-use
    internal identity. Never raises for an upstream error status -- the
    caller decides how to represent status/candidates for that engine."""
    token, request_id = _mint_demo_identity()
    headers = {"Authorization": f"Bearer {token}", "X-Request-Id": request_id}
    data: Dict[str, str] = {"topK": str(demo_config.DEMO_TOP_K)}
    if engine:
        data["engine"] = engine
    if text:
        data["text"] = text
    files = {"file": file_field} if file_field else None
    try:
        async with httpx.AsyncClient(base_url=demo_config.DEMO_INTERNAL_BASE_URL, timeout=httpx.Timeout(30.0)) as client:
            response = await client.post(path, data=data, files=files, headers=headers)
    except httpx.HTTPError:
        logger.warning("Demo facade upstream call to %s failed", path, exc_info=True)
        return 503, {}
    try:
        body = response.json()
    except ValueError:
        body = {}
    return response.status_code, body


def _candidate_image_url(item: Dict[str, Any]) -> Optional[str]:
    filename = item.get("candidateFilename") or (item.get("metadata") or {}).get("candidateFilename")
    if not filename or not isinstance(filename, str):
        return None
    safe_name = Path(filename).name  # strips any accidental path components
    if safe_name != filename or not safe_name:
        return None
    if Path(safe_name).suffix.lower() not in _ALLOWED_ASSET_EXTENSIONS:
        return None
    return f"/demo/assets/candidates/{safe_name}"


def _safe_candidate(raw_candidate: Dict[str, Any], rank: int) -> Dict[str, Any]:
    item = raw_candidate.get("item") or {}
    explanation = raw_candidate.get("explanation") or {}
    return {
        "candidateId": raw_candidate.get("candidateId"),
        "rank": rank,
        "title": item.get("title") or item.get("name"),
        "description": item.get("description"),
        "score": raw_candidate.get("score"),
        "imageUrl": _candidate_image_url(item),
        "explanation": explanation.get("summary") or explanation.get("reason"),
    }


def _model_summary(info: Dict[str, Any], engine_id: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"modelId": info.get("modelId"), "embeddingDimension": info.get("embeddingDimension")}
    if engine_id == "siglip2_v1":
        summary["calibrationStatus"] = info.get("calibrationStatus", "uncalibrated")
    return summary


def _build_single_response(request_id: str, mode: str, status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    engine_id = "clip_v1" if mode == "clip" else "siglip2_v1"
    if status_code != 200 or not body:
        return {"requestId": request_id, "mode": mode, "status": "unavailable", "latencyMs": None, "model": None, "candidates": []}
    governance = body.get("governance") or {}
    candidates = [_safe_candidate(candidate, index + 1) for index, candidate in enumerate(body.get("topK") or [])]
    return {
        "requestId": request_id,
        "mode": mode,
        "status": "success",
        "latencyMs": governance.get("latencyMs"),
        "model": _model_summary(governance, engine_id),
        "candidates": candidates,
    }


def _empty_side(latency_ms: Optional[int] = None) -> Dict[str, Any]:
    return {"status": "unavailable", "latencyMs": latency_ms, "model": None, "candidates": []}


def _build_compare_response(request_id: str, status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    if status_code != 200 or not body:
        return {
            "requestId": request_id,
            "mode": "compare",
            "clip": _empty_side(),
            "siglip2": {**_empty_side(), "calibrationStatus": "uncalibrated"},
            "comparison": {"sameTop1": None, "topKOverlap": None, "rankChanges": None, "latencyDifferenceMs": None},
        }

    def _side(engine_id: str, raw_side: Dict[str, Any]) -> Dict[str, Any]:
        raw_side = raw_side or {}
        if raw_side.get("status") != "success":
            side = _empty_side(raw_side.get("latencyMs"))
        else:
            provenance = raw_side.get("provenance") or {}
            side = {
                "status": "success",
                "latencyMs": raw_side.get("latencyMs"),
                "model": _model_summary(provenance, engine_id),
                "candidates": [_safe_candidate(candidate, index + 1) for index, candidate in enumerate(raw_side.get("candidates") or [])],
            }
        if engine_id == "siglip2_v1":
            side["calibrationStatus"] = raw_side.get("calibrationStatus", "uncalibrated")
        return side

    raw_comparison = body.get("comparison") or {}
    return {
        "requestId": request_id,
        "mode": "compare",
        "clip": _side("clip_v1", body.get("clip") or {}),
        "siglip2": _side("siglip2_v1", body.get("siglip2") or {}),
        "comparison": {
            "sameTop1": raw_comparison.get("sameTop1"),
            "topKOverlap": raw_comparison.get("topKOverlap"),
            "rankChanges": raw_comparison.get("rankChanges"),
            "latencyDifferenceMs": raw_comparison.get("latencyDifferenceMs"),
        },
    }


def _audit_demo_match(*, request_id: str, mode: str, has_image: bool, has_text: bool, result: Dict[str, Any]) -> None:
    if mode == "compare":
        clip_status = (result.get("clip") or {}).get("status")
        siglip2_status = (result.get("siglip2") or {}).get("status")
        latency = None
        result_ids: List[Any] = []
        for side_key in ("clip", "siglip2"):
            for candidate in (result.get(side_key) or {}).get("candidates") or []:
                result_ids.append(candidate.get("candidateId"))
    else:
        clip_status = result.get("status") if mode == "clip" else None
        siglip2_status = result.get("status") if mode == "siglip2" else None
        latency = result.get("latencyMs")
        result_ids = [candidate.get("candidateId") for candidate in result.get("candidates") or []]
    _app().get_repository().add_audit_log(
        "festival_demo_match",
        {
            "mode": mode,
            "tenantId": demo_config.DEMO_TENANT_ID,
            "siteId": demo_config.DEMO_SITE_ID,
            "datasetVersion": demo_config.DEMO_DATASET_VERSION,
            "hasImage": has_image,
            "hasText": has_text,
            "clipStatus": clip_status,
            "siglip2Status": siglip2_status,
            "latency": latency,
            "resultCandidateIds": result_ids,
        },
        request_id=request_id,
        tenant_id=demo_config.DEMO_TENANT_ID,
    )


@router.get("/demo/status")
def demo_status():
    _require_demo_ready()
    from embedding_engines.registry import get_engine

    clip_engine = get_engine("clip_v1")
    siglip2_engine = get_engine("siglip2_v1")
    clip_readiness = clip_engine.readiness().to_dict()
    clip_provenance = clip_engine.provenance().to_dict()
    siglip2_readiness = siglip2_engine.readiness().to_dict()
    siglip2_provenance = siglip2_engine.provenance().to_dict()
    return {
        "enabled": True,
        "clip": {
            "ready": clip_readiness["ready"],
            "modelId": clip_provenance["modelId"],
            "embeddingDimension": clip_provenance["embeddingDimension"],
        },
        "siglip2": {
            "ready": siglip2_readiness["ready"],
            "modelId": siglip2_provenance["modelId"],
            "embeddingDimension": siglip2_provenance["embeddingDimension"],
            "calibrationStatus": siglip2_provenance["calibrationStatus"],
        },
        "datasetVersion": demo_config.DEMO_DATASET_VERSION,
        "mode": "demo",
    }


@router.post("/demo/match")
async def demo_match(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    mode: str = Form(default="clip"),
):
    app_module = _app()
    _require_demo_ready()
    _enforce_demo_rate_limit(request)

    normalized_mode = (mode or "clip").strip().lower()
    if normalized_mode not in _ALLOWED_MODES:
        raise HTTPException(
            status_code=400,
            detail=app_module.problem_detail("invalid_mode", f"mode must be one of {sorted(_ALLOWED_MODES)}."),
        )

    text_value = app_module.compact_text(text, limit=2000) if text else None
    file_field: Optional[Tuple[str, bytes, str]] = None
    has_image = False
    if file is not None and (file.filename or file.content_type):
        raw, _pil_image = await app_module.read_image_upload(file, "file")
        has_image = True
        file_field = (file.filename or "upload", raw, file.content_type or "application/octet-stream")

    if not has_image and not text_value:
        raise HTTPException(
            status_code=400, detail=app_module.problem_detail("missing_query", "Provide an image file, text, or both.")
        )

    request_id = app_module.get_request_id(request) or str(uuid.uuid4())

    if normalized_mode == "compare":
        status_code, body = await _forward("/v2/ab/match", engine=None, file_field=file_field, text=text_value)
        result = _build_compare_response(request_id, status_code, body)
    else:
        engine_id = "clip_v1" if normalized_mode == "clip" else "siglip2_v1"
        status_code, body = await _forward("/v2/match", engine=engine_id, file_field=file_field, text=text_value)
        result = _build_single_response(request_id, normalized_mode, status_code, body)

    _audit_demo_match(request_id=request_id, mode=normalized_mode, has_image=has_image, has_text=bool(text_value), result=result)

    response = JSONResponse(content=result)
    response.headers["Cache-Control"] = "no-store"
    response.headers[app_module.REQUEST_ID_HEADER] = request_id
    return response


@router.get("/demo/assets/candidates/{filename}")
def demo_asset(filename: str):
    _require_demo_ready()
    if not filename or "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise _not_found()
    candidate_name = Path(filename).name
    if candidate_name != filename:
        raise _not_found()
    extension = Path(candidate_name).suffix.lower()
    content_type = _ALLOWED_ASSET_EXTENSIONS.get(extension)
    if content_type is None:
        raise _not_found()
    base_dir = (Path(demo_config.DEMO_ASSET_DIR) / "candidates").resolve()
    resolved_path = (base_dir / candidate_name).resolve()
    try:
        resolved_path.relative_to(base_dir)
    except ValueError:
        raise _not_found()
    if not resolved_path.is_file():
        raise _not_found()
    return FileResponse(path=str(resolved_path), media_type=content_type, headers={"Cache-Control": "public, max-age=3600"})
