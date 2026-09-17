"""P13: model-engine-aware V2 API surface.

Adds `/v2/models`, `/v2/embeddings/image`, `/v2/embeddings/text`,
`/v2/items/{item_id}/embeddings/{engine}`, `/v2/match`, and `/v2/ab/match`
alongside (never in place of) the legacy top-level endpoints in app.py.
Mounted with `app.include_router(v2_router)` at the bottom of app.py.

Every legacy endpoint's contract, defaults, and CLIP-only behaviour are
untouched by this module. Auth, audit, request-id, error-response, OCR,
and barcode conventions are reused from app.py through a deferred import
(`_app()`) rather than duplicated -- the same pattern already used by
embedding_engines/clip_engine.py -- so this module never needs `app.py`
to be importable at *this* module's import time (avoiding a circular
import, since app.py is what imports and mounts this router).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from embedding_engines import config as engine_config
from embedding_engines.base import EmbeddingEngine, EngineUnavailableError
from embedding_engines.registry import all_engines, describe_models, get_engine
from embedding_engines.vector_index import NumpyVectorIndex

router = APIRouter()

ENGINE_IDS: Tuple[str, ...] = ("clip_v1", "siglip2_v1")
# Ground-truth (expectedCandidateId) evaluation is for internal/admin use
# only (section 16) -- gated behind its own action so it is never reachable
# through whatever permission set a normal citizen-facing integration holds.
EVALUATE_ACTION = "match:evaluate"


def _app():
    import app as app_module  # noqa: PLC0415 - deferred to avoid a circular import at module load time

    return app_module


def _problem(code: str, message: str, details: Any = None) -> Dict[str, Any]:
    return _app().problem_detail(code, message, details)


def _resolve_engine(engine_id: Optional[str]) -> EmbeddingEngine:
    normalized = (engine_id or engine_config.DEFAULT_EMBEDDING_ENGINE).strip()
    if normalized not in ENGINE_IDS:
        raise HTTPException(
            status_code=400,
            detail=_problem("invalid_engine", f"engine must be one of {list(ENGINE_IDS)}.", {"engine": engine_id}),
        )
    engine = get_engine(normalized)
    if not engine.is_enabled():
        raise HTTPException(
            status_code=503,
            detail=_problem("engine_disabled", f"Engine '{normalized}' is disabled by configuration."),
        )
    return engine


def _unavailable_detail(engine: EmbeddingEngine, exc: EngineUnavailableError) -> Dict[str, Any]:
    # Safe, structured error -- never a raw Python traceback.
    return _problem(
        "engine_unavailable",
        f"Engine '{engine.engine_id}' could not process this request.",
        {"errorCategory": exc.error_category, "engine": engine.engine_id},
    )


async def _encode_image_or_503(engine: EmbeddingEngine, image: Image.Image) -> List[float]:
    app_module = _app()
    started = time.time()
    try:
        vector = await engine.encode_image_async(image)
    except EngineUnavailableError as exc:
        app_module.record_engine_inference(engine.engine_id, "image", "error", time.time() - started)
        raise HTTPException(status_code=503, detail=_unavailable_detail(engine, exc)) from exc
    app_module.record_engine_inference(engine.engine_id, "image", "success", time.time() - started)
    return vector


async def _encode_text_or_503(engine: EmbeddingEngine, text: str) -> List[float]:
    app_module = _app()
    started = time.time()
    try:
        vector = await engine.encode_text_async(text)
    except EngineUnavailableError as exc:
        app_module.record_engine_inference(engine.engine_id, "text", "error", time.time() - started)
        raise HTTPException(status_code=503, detail=_unavailable_detail(engine, exc)) from exc
    app_module.record_engine_inference(engine.engine_id, "text", "success", time.time() - started)
    return vector


def _write_embedding_block(item: Dict[str, Any], engine: EmbeddingEngine, vector: List[float], modality: str) -> None:
    """Writes embeddings.<engine_id> (section 8). For clip_v1 only, also
    mirrors into the legacy top-level embedding/clipEmbedding fields --
    same vector space, so this keeps them fresh exactly as the legacy
    /items/{id}/re-embed endpoint already would. SigLIP2 (or any other
    non-clip_v1 engine) NEVER touches those legacy fields."""
    app_module = _app()
    provenance = engine.provenance()
    now = app_module.now_iso()
    embeddings = dict(item.get("embeddings") or {})
    existing_block = embeddings.get(engine.engine_id) or {}
    embeddings[engine.engine_id] = {
        "vector": vector,
        "modelId": provenance.model_id,
        "modelRevision": provenance.model_revision,
        "embeddingDimension": len(vector),
        "preprocessingVersion": provenance.preprocessing_version,
        "embeddingModality": modality,
        "createdAt": existing_block.get("createdAt") or now,
        "updatedAt": now,
    }
    item["embeddings"] = embeddings
    if engine.engine_id == "clip_v1":
        item["embedding"] = vector
        item["clipEmbedding"] = vector
        item["embeddingDimension"] = len(vector)
        item["embeddingModality"] = modality


def _serialize_item_v2(item: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy serialize_item() plus a non-secret `embeddings` summary --
    metadata only, the raw vector is stripped (section 10: never expose
    embeddings where the caller doesn't need them)."""
    app_module = _app()
    payload = app_module.serialize_item(item)
    embeddings_meta: Dict[str, Any] = {}
    for engine_id, block in (item.get("embeddings") or {}).items():
        if isinstance(block, dict):
            embeddings_meta[engine_id] = {key: value for key, value in block.items() if key != "vector"}
    payload["embeddings"] = embeddings_meta
    return payload


def _candidate_vector(candidate: Dict[str, Any], engine: EmbeddingEngine) -> Optional[Tuple[List[float], Dict[str, Any]]]:
    """Returns (vector, meta) for `candidate` under `engine`'s own vector
    space, or None if unusable. Never returns a vector belonging to a
    different engine's space (section 9: cross-model comparison is
    always rejected by simply never producing a vector to compare)."""
    block = (candidate.get("embeddings") or {}).get(engine.engine_id)
    vector: Optional[List[float]] = None
    modality: Optional[str] = None
    if isinstance(block, dict) and block.get("vector"):
        vector = block.get("vector")
        modality = block.get("embeddingModality")
    elif engine.engine_id == "clip_v1" and candidate.get("embedding"):
        # Legacy items carry their CLIP vector directly on the item (never
        # under embeddings.clip_v1) until explicitly re-embedded through the
        # V2 API. Same vector space, so this fallback is safe -- and is
        # deliberately clip_v1-only.
        vector = candidate.get("embedding")
        modality = candidate.get("embeddingModality") or ("image" if candidate.get("image") else None)
    if not vector or len(vector) != engine.expected_dimension:
        return None
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm <= 0 or not np.isfinite(array).all():
        return None
    return vector, {"embeddingModality": modality}


def _v2_match_explanation(
    candidate: Dict[str, Any],
    engine: EmbeddingEngine,
    query_ocr_text: Optional[str],
    query_barcodes: List[Dict[str, Any]],
    query_labels: List[str],
    score: float,
    image_cosine: Optional[float],
    text_cosine: Optional[float],
) -> Dict[str, Any]:
    app_module = _app()
    block = (candidate.get("embeddings") or {}).get(engine.engine_id) or {}
    candidate_embedding_modality = (
        block.get("embeddingModality") or candidate.get("embeddingModality") or ("image" if candidate.get("image") else None)
    )
    item_barcodes = app_module.normalize_barcodes(candidate.get("barcodes") or candidate.get("barcodeValues"))
    barcode_overlap = app_module.barcode_overlap_values(query_barcodes, item_barcodes)
    query_ocr_tokens = set(app_module.tokenize(query_ocr_text))
    item_ocr_tokens = set(app_module.tokenize(candidate.get("ocrText")))
    ocr_overlap = sorted(query_ocr_tokens & item_ocr_tokens)[:5]
    query_label_map = {value.lower(): value for value in query_labels}
    item_label_map = {value.lower(): value for value in app_module.normalize_string_list(candidate.get("labels"))}
    label_overlap = [item_label_map[key] for key in sorted(query_label_map.keys() & item_label_map.keys())][:5]

    reasons: List[str] = []
    signals: Dict[str, Any] = {"cosine": round(float(score), 6)}
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
        reasons.append(
            "visual similarity" if candidate_embedding_modality != "text" else "text similarity (no image evidence available)"
        )
    reason = ", ".join(reasons)
    similarity: Dict[str, Any] = {"cosine": round(float(score), 6), "combinedCosine": round(float(score), 6)}
    if image_cosine is not None:
        similarity["imageCosine"] = round(float(image_cosine), 6)
    if text_cosine is not None:
        similarity["textCosine"] = round(float(text_cosine), 6)
    return {
        "reason": reason,
        "summary": reason,
        "signals": signals,
        "similarity": similarity,
        "candidateEmbeddingModality": candidate_embedding_modality,
        "ocr": {"matchedTerms": ocr_overlap, "matchCount": len(ocr_overlap)},
        "barcode": {"matchedValues": barcode_overlap, "matchCount": len(barcode_overlap)},
        "labels": {"matchedValues": label_overlap, "matchCount": len(label_overlap)},
        "model": {"engine": engine.engine_id, **engine.provenance().to_dict()},
    }


async def _parse_query(request: Request) -> Dict[str, Any]:
    app_module = _app()
    form = await request.form()
    upload_value = app_module.first_value(form, "file", "image")
    upload = upload_value if app_module.is_upload(upload_value) else None
    return {
        "upload": upload,
        "text": app_module.compact_text(app_module.first_value(form, "text", "queryText", "query"), limit=2000),
        "topK": app_module.parse_top_k(app_module.first_value(form, "topK", "k", "limit")),
        "ocrTextManual": app_module.compact_text(app_module.first_value(form, "ocrText", "ocr"), limit=500),
        "barcodeValuesManual": app_module.normalize_barcodes(app_module.first_value(form, "barcodeValues", "barcodes")),
        "labels": app_module.normalize_string_list(app_module.first_value(form, "labels", "tags")),
        "doOcr": app_module.normalize_bool(app_module.first_value(form, "doOcr"), default=app_module.OCR_ENABLED),
        "doBarcode": app_module.normalize_bool(app_module.first_value(form, "doBarcode"), default=True),
        "expectedCandidateId": app_module.compact_text(app_module.first_value(form, "expectedCandidateId"), limit=200),
        "engine": form.get("engine"),
    }


async def _execute_engine_match(
    identity,
    engine: EmbeddingEngine,
    pil_image: Optional[Image.Image],
    text: Optional[str],
    top_k: int,
    query_ocr_text: Optional[str],
    combined_barcodes: List[Dict[str, Any]],
    labels: List[str],
) -> Dict[str, Any]:
    app_module = _app()
    started = time.time()
    query_vectors: List[List[float]] = []
    image_vector: Optional[List[float]] = None
    text_vector: Optional[List[float]] = None
    if pil_image is not None:
        image_vector = await _encode_image_or_503(engine, pil_image)
        query_vectors.append(image_vector)
    if text:
        text_vector = await _encode_text_or_503(engine, text)
        query_vectors.append(text_vector)
    if not query_vectors:
        raise HTTPException(
            status_code=400, detail=_problem("invalid_match_request", "Provide an image file, text query, or both.")
        )

    query_vector = app_module.normalize_vectors(np.mean(np.asarray(query_vectors, dtype=float), axis=0))[0]
    image_query_vector = app_module.normalize_vectors(image_vector)[0] if image_vector is not None else None
    text_query_vector = app_module.normalize_vectors(text_vector)[0] if text_vector is not None else None

    candidates = app_module.get_repository().list_matchable_items(identity, limit=app_module.CORPUS_LIMIT)
    rows: List[Tuple[str, str, str, List[float]]] = []
    candidate_by_id: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        extracted = _candidate_vector(candidate, engine)
        if extracted is None:
            continue
        vector, _meta = extracted
        candidate_id = candidate["id"]
        rows.append((candidate_id, candidate["tenantId"], candidate["siteId"], vector))
        candidate_by_id[candidate_id] = candidate

    index = NumpyVectorIndex()
    index.build(engine.engine_id, rows)
    ranked = index.search(
        engine.engine_id, identity.tenant_id, identity.permitted_site_ids, query_vector.tolist(), top_k=top_k
    )

    results: List[Dict[str, Any]] = []
    for candidate_id, score in ranked:
        candidate = candidate_by_id[candidate_id]
        extracted = _candidate_vector(candidate, engine)
        candidate_vector = np.asarray(extracted[0], dtype=float)
        candidate_vector = candidate_vector / (np.linalg.norm(candidate_vector) or 1.0)
        image_cosine = float(np.dot(image_query_vector, candidate_vector)) if image_query_vector is not None else None
        text_cosine = float(np.dot(text_query_vector, candidate_vector)) if text_query_vector is not None else None
        results.append(
            {
                "candidateId": candidate_id,
                "item": _serialize_item_v2(candidate),
                "score": round(float(score), 6),
                "explanation": _v2_match_explanation(
                    candidate, engine, query_ocr_text, combined_barcodes, labels, float(score), image_cosine, text_cosine
                ),
            }
        )

    scores = [row["score"] for row in results]
    top_score = scores[0] if scores else None
    second_score = scores[1] if len(scores) > 1 else None
    latency_s = time.time() - started
    app_module.record_engine_inference(engine.engine_id, "match", "success", latency_s)
    return {
        "engine": engine.engine_id,
        "provenance": engine.provenance().to_dict(),
        "latencyMs": int(latency_s * 1000),
        "candidatesEvaluated": len(rows),
        "candidates": results,
        "top1Score": top_score,
        "top1Top2Margin": (top_score - second_score) if (top_score is not None and second_score is not None) else None,
    }


def _decision_for_engine(engine_id: str, top_score: Optional[float], second_score: Optional[float]) -> Dict[str, Any]:
    app_module = _app()
    if top_score is None:
        return {"noMatch": True, "reason": "NO_CANDIDATES", "details": {}}
    if engine_id == "clip_v1":
        # clip_v1 keeps using CLIP's own existing, production-calibrated
        # thresholds untouched (CONF_MIN_SCORE/CONF_MIN_MARGIN) -- section 14.
        no_match, meta = app_module.should_return_no_match(
            top_score, second_score, app_module.CONF_MIN_SCORE, app_module.CONF_MIN_MARGIN
        )
        return {"noMatch": no_match, "reason": str(meta.get("reason")), "details": meta}
    # Any other engine (siglip2_v1 today) is uncalibrated by default -- CLIP's
    # thresholds carry no meaning in a different score space and must never
    # be silently reused (section 14). Only gate if an operator has
    # explicitly configured a real threshold from evaluation data.
    min_score = engine_config.SIGLIP2_MIN_SCORE
    if min_score is None:
        return {
            "noMatch": False,
            "reason": "UNCALIBRATED",
            "details": {"calibrationStatus": "uncalibrated", "topScore": top_score, "secondScore": second_score},
        }
    min_margin = float(engine_config.SIGLIP2_MIN_MARGIN or 0.0)
    no_match, meta = app_module.should_return_no_match(top_score, second_score, float(min_score), min_margin)
    meta["calibrationStatus"] = "uncalibrated"
    return {"noMatch": no_match, "reason": str(meta.get("reason")), "details": meta}


def _evaluation_metrics(candidates: List[Dict[str, Any]], expected_candidate_id: str) -> Dict[str, Any]:
    ids = [candidate["candidateId"] for candidate in candidates]
    rank = ids.index(expected_candidate_id) + 1 if expected_candidate_id in ids else None
    return {
        "expectedCandidateId": expected_candidate_id,
        "expectedRank": rank,
        "recallAt1": bool(rank is not None and rank <= 1),
        "recallAt5": bool(rank is not None and rank <= 5),
        "recallAt10": bool(rank is not None and rank <= 10),
        "reciprocalRank": (1.0 / rank) if rank else 0.0,
    }


def _require_evaluate_permission(identity, expected_candidate_id: Optional[str]) -> None:
    if expected_candidate_id and EVALUATE_ACTION not in identity.actions:
        raise HTTPException(
            status_code=403,
            detail=_problem(
                "action_not_permitted", f"Ground-truth evaluation requires the '{EVALUATE_ACTION}' permission."
            ),
        )


@router.get("/v2/models")
def v2_models(request: Request):
    app_module = _app()
    app_module.authorize(request, "health:read")
    return app_module.with_request_id(request, {"models": describe_models()})


@router.post("/v2/embeddings/image")
async def v2_encode_image(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    engine: Optional[str] = Form(default=None),
):
    app_module = _app()
    started = time.time()
    endpoint = "/v2/embeddings/image"
    app_module.authorize(request, "match:execute")
    engine_instance = _resolve_engine(engine)
    upload = file or image
    if upload is None:
        raise HTTPException(
            status_code=400, detail=_problem("missing_image", "No image uploaded. Use form field 'file' or 'image'.")
        )
    _, pil_image = await app_module.read_image_upload(upload, "file")
    vector = await _encode_image_or_503(engine_instance, pil_image)
    provenance = engine_instance.provenance().to_dict()
    response = app_module.with_request_id(
        request,
        {
            "encoded": True,
            "engine": engine_instance.engine_id,
            "modelId": provenance["modelId"],
            "modelRevision": provenance["modelRevision"],
            "embeddingDimension": len(vector),
            "preprocessingVersion": provenance["preprocessingVersion"],
            "calibrationStatus": provenance["calibrationStatus"],
        },
    )
    app_module.record_latency_metric(endpoint, started)
    app_module.record_request_metric(endpoint, 200)
    return response


@router.post("/v2/embeddings/text")
async def v2_encode_text(
    request: Request,
    text: Optional[str] = Form(default=None),
    queryText: Optional[str] = Form(default=None),
    engine: Optional[str] = Form(default=None),
):
    app_module = _app()
    started = time.time()
    endpoint = "/v2/embeddings/text"
    app_module.authorize(request, "match:execute")
    engine_instance = _resolve_engine(engine)
    value = app_module.compact_text(text or queryText, limit=2000)
    if not value:
        raise HTTPException(
            status_code=400, detail=_problem("missing_text", "No text provided. Use form field 'text' or 'queryText'.")
        )
    vector = await _encode_text_or_503(engine_instance, value)
    provenance = engine_instance.provenance().to_dict()
    response = app_module.with_request_id(
        request,
        {
            "encoded": True,
            "engine": engine_instance.engine_id,
            "modelId": provenance["modelId"],
            "modelRevision": provenance["modelRevision"],
            "embeddingDimension": len(vector),
            "preprocessingVersion": provenance["preprocessingVersion"],
            "calibrationStatus": provenance["calibrationStatus"],
        },
    )
    app_module.record_latency_metric(endpoint, started)
    app_module.record_request_metric(endpoint, 200)
    return response


@router.post("/v2/items/{item_id}/embeddings/{engine}")
async def v2_reembed_item(request: Request, item_id: str, engine: str, file: UploadFile = File(...)):
    app_module = _app()
    identity = app_module.authorize(request, "corpus:write")
    engine_instance = _resolve_engine(engine)
    existing = app_module.get_repository().get_item(item_id, identity)
    if existing is None:
        raise HTTPException(status_code=404, detail=_problem("item_not_found", f"Item '{item_id}' was not found."))
    app_module.assert_dataset_scope_consistent(existing, identity, item_id=item_id)
    _, pil_image = await app_module.read_image_upload(file, "file")
    vector = await _encode_image_or_503(engine_instance, pil_image)
    _write_embedding_block(existing, engine_instance, vector, "image")
    existing["updatedAt"] = app_module.now_iso()
    saved = app_module.get_repository().upsert_item(existing)
    provenance = engine_instance.provenance().to_dict()
    app_module.get_repository().add_audit_log(
        "re_embedding_v2",
        {
            "engine": engine_instance.engine_id,
            "modelId": provenance["modelId"],
            "modelRevision": provenance["modelRevision"],
            "preprocessingVersion": provenance["preprocessingVersion"],
            "embeddingDimension": len(vector),
        },
        item_id=item_id,
        request_id=app_module.get_request_id(request),
        tenant_id=identity.tenant_id,
    )
    return app_module.with_request_id(request, {"item": _serialize_item_v2(saved)})


@router.post("/v2/match")
async def v2_match(request: Request):
    app_module = _app()
    started = time.time()
    endpoint = "/v2/match"
    identity = app_module.authorize(request, "match:execute")
    parsed = await _parse_query(request)
    engine_instance = _resolve_engine(parsed["engine"])
    _require_evaluate_permission(identity, parsed["expectedCandidateId"])

    raw_bytes = None
    pil_image = None
    query_image_info = None
    query_ocr_payload = None
    query_barcode_payload = None
    query_ocr_error = None
    query_barcode_error = None
    if parsed["upload"] is not None:
        raw_bytes, pil_image = await app_module.read_image_upload(parsed["upload"], "file")
        query_image_info = app_module.image_info(pil_image, parsed["upload"])
        if parsed["doOcr"]:
            query_ocr_payload, query_ocr_error = await app_module.run_ocr(pil_image, app_module.OCR_LANGUAGES, app_module.OCR_PSM)
        if parsed["doBarcode"]:
            query_barcode_payload, query_barcode_error = await app_module.run_barcode_scan(pil_image)
    if pil_image is None and not parsed["text"]:
        raise HTTPException(
            status_code=400, detail=_problem("invalid_match_request", "Provide an image file, text query, or both.")
        )

    scanned_barcodes = app_module.normalize_barcodes((query_barcode_payload or {}).get("barcodes"))
    combined_barcodes = app_module.merge_barcodes(parsed["barcodeValuesManual"], scanned_barcodes)
    query_barcode_values = app_module.barcode_values(combined_barcodes)
    query_ocr_text = parsed["ocrTextManual"] or ((query_ocr_payload or {}).get("fullText"))

    engine_result = await _execute_engine_match(
        identity, engine_instance, pil_image, parsed["text"], parsed["topK"], query_ocr_text, combined_barcodes, parsed["labels"]
    )
    scores = [c["score"] for c in engine_result["candidates"]]
    decision = _decision_for_engine(
        engine_instance.engine_id, engine_result["top1Score"], scores[1] if len(scores) > 1 else None
    )
    evaluation = (
        _evaluation_metrics(engine_result["candidates"], parsed["expectedCandidateId"])
        if parsed["expectedCandidateId"]
        else None
    )

    app_module.get_repository().add_audit_log(
        "match_execution_v2",
        {
            "engine": engine_instance.engine_id,
            "modelId": engine_result["provenance"]["modelId"],
            "modelRevision": engine_result["provenance"]["modelRevision"],
            "preprocessingVersion": engine_result["provenance"]["preprocessingVersion"],
            "scoringVersion": engine_config.V2_SCORING_VERSION,
            "topK": parsed["topK"],
            "resultIds": [c["candidateId"] for c in engine_result["candidates"]],
            "scores": scores,
            "latencyMs": engine_result["latencyMs"],
            "decisioning": decision,
        },
        request_id=app_module.get_request_id(request),
        tenant_id=identity.tenant_id,
    )
    if query_ocr_error:
        app_module.get_repository().add_audit_log(
            "ocr_failure",
            {"reason": query_ocr_error, "inputSha256": app_module.sha256_hex(raw_bytes or b"")},
            request_id=app_module.get_request_id(request),
            tenant_id=identity.tenant_id,
        )

    response = app_module.with_request_id(
        request,
        {
            "governance": {
                "engine": engine_instance.engine_id,
                **engine_result["provenance"],
                "releasedOnly": True,
                "topKRequested": parsed["topK"],
                "candidatesEvaluated": engine_result["candidatesEvaluated"],
                "latencyMs": app_module.request_latency_ms(request),
                "decision": decision,
            },
            "query": {
                "type": "+".join(filter(None, ["image" if pil_image is not None else None, "text" if parsed["text"] else None])),
                "text": parsed["text"],
                "image": query_image_info,
                "ocrText": query_ocr_text,
                "barcodeValues": query_barcode_values,
                "labels": parsed["labels"],
            },
            "decision": decision,
            "topK": engine_result["candidates"],
            "evaluation": evaluation,
        },
    )
    app_module.record_latency_metric(endpoint, started)
    app_module.record_request_metric(endpoint, 200)
    return response


def _compare_engine_outputs(outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    clip_out = outputs.get("clip_v1", {})
    sig_out = outputs.get("siglip2_v1", {})
    if clip_out.get("status") != "success" or sig_out.get("status") != "success":
        return {
            "topKOverlap": None,
            "sameTop1": None,
            "rankChanges": None,
            "latencyDifferenceMs": None,
            "note": "Comparison metrics require both engines to have succeeded.",
        }
    clip_ids = [c["candidateId"] for c in clip_out["candidates"]]
    sig_ids = [c["candidateId"] for c in sig_out["candidates"]]
    clip_rank = {cid: i for i, cid in enumerate(clip_ids)}
    sig_rank = {cid: i for i, cid in enumerate(sig_ids)}
    rank_changes = [
        {"candidateId": cid, "clipRank": clip_rank[cid] + 1, "siglip2Rank": sig_rank[cid] + 1}
        for cid in (set(clip_ids) & set(sig_ids))
        if clip_rank[cid] != sig_rank[cid]
    ]
    return {
        "topKOverlap": len(set(clip_ids) & set(sig_ids)),
        "sameTop1": bool(clip_ids and sig_ids and clip_ids[0] == sig_ids[0]),
        "rankChanges": rank_changes,
        # Never used to declare a "winner": the two engines' cosine scales
        # are not directly comparable (section 15) -- this is timing only.
        "latencyDifferenceMs": sig_out["latencyMs"] - clip_out["latencyMs"],
    }


@router.post("/v2/ab/match")
async def v2_ab_match(request: Request):
    app_module = _app()
    started = time.time()
    endpoint = "/v2/ab/match"
    identity = app_module.authorize(request, "match:execute")
    if not engine_config.AB_TEST_ENABLED:
        raise HTTPException(
            status_code=503, detail=_problem("ab_testing_disabled", "A/B testing is disabled by configuration.")
        )
    parsed = await _parse_query(request)
    _require_evaluate_permission(identity, parsed["expectedCandidateId"])

    pil_image = None
    query_image_info = None
    if parsed["upload"] is not None:
        _, pil_image = await app_module.read_image_upload(parsed["upload"], "file")
        query_image_info = app_module.image_info(pil_image, parsed["upload"])
    if pil_image is None and not parsed["text"]:
        raise HTTPException(
            status_code=400, detail=_problem("invalid_match_request", "Provide an image file, text query, or both.")
        )

    query_ocr_text = parsed["ocrTextManual"]
    combined_barcodes = parsed["barcodeValuesManual"]

    engine_outputs: Dict[str, Dict[str, Any]] = {}
    for engine_id in ENGINE_IDS:
        engine_instance = get_engine(engine_id)
        if not engine_instance.is_enabled():
            engine_outputs[engine_id] = {
                "status": "unavailable",
                "errorCode": "engine_disabled",
                "message": f"{engine_id} is disabled by configuration.",
            }
            continue
        try:
            result = await _execute_engine_match(
                identity, engine_instance, pil_image, parsed["text"], parsed["topK"], query_ocr_text, combined_barcodes, parsed["labels"]
            )
            scores = [c["score"] for c in result["candidates"]]
            decision = _decision_for_engine(engine_id, result["top1Score"], scores[1] if len(scores) > 1 else None)
            evaluation = (
                _evaluation_metrics(result["candidates"], parsed["expectedCandidateId"])
                if parsed["expectedCandidateId"]
                else None
            )
            engine_outputs[engine_id] = {
                "status": "success",
                "provenance": result["provenance"],
                "latencyMs": result["latencyMs"],
                "candidates": result["candidates"],
                "candidatesEvaluated": result["candidatesEvaluated"],
                "top1Score": result["top1Score"],
                "top1Top2Margin": result["top1Top2Margin"],
                "decision": decision,
                "calibrationStatus": result["provenance"]["calibrationStatus"],
                "evaluation": evaluation,
            }
        except EngineUnavailableError as exc:
            engine_outputs[engine_id] = {
                "status": "unavailable",
                "errorCode": exc.error_category,
                "message": f"{engine_id} failed to produce results; see errorCode.",
            }
        except HTTPException as exc:
            code = exc.detail.get("code") if isinstance(exc.detail, dict) else "request_error"
            engine_outputs[engine_id] = {
                "status": "unavailable",
                "errorCode": str(code or "request_error"),
                "message": f"{engine_id} could not process this request.",
            }

    comparison = _compare_engine_outputs(engine_outputs)

    app_module.get_repository().add_audit_log(
        "ab_match_execution",
        {
            "clipStatus": engine_outputs.get("clip_v1", {}).get("status"),
            "siglip2Status": engine_outputs.get("siglip2_v1", {}).get("status"),
            "comparison": comparison,
        },
        request_id=app_module.get_request_id(request),
        tenant_id=identity.tenant_id,
    )

    response = app_module.with_request_id(
        request,
        {
            "query": {
                "type": "+".join(filter(None, ["image" if pil_image is not None else None, "text" if parsed["text"] else None])),
                "text": parsed["text"],
                "image": query_image_info,
            },
            "clip": engine_outputs.get("clip_v1"),
            "siglip2": engine_outputs.get("siglip2_v1"),
            "comparison": comparison,
        },
    )
    app_module.record_latency_metric(endpoint, started)
    app_module.record_request_metric(endpoint, 200)
    return response
