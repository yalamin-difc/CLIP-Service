"""Model registry: the single place that knows about every embedding
engine (clip_v1, siglip2_v1). Backs GET /v2/models and the V2 match/AB
endpoints. Never exposes secrets or raw embeddings.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import EmbeddingEngine
from .clip_engine import ClipEngine
from .siglip2_engine import Siglip2Engine

_clip_engine = ClipEngine()
_siglip2_engine = Siglip2Engine()

_ENGINES: Dict[str, EmbeddingEngine] = {
    _clip_engine.engine_id: _clip_engine,
    _siglip2_engine.engine_id: _siglip2_engine,
}


class UnknownEngineError(ValueError):
    pass


def get_engine(engine_id: str) -> EmbeddingEngine:
    engine = _ENGINES.get(engine_id)
    if engine is None:
        raise UnknownEngineError(f"Unknown embedding engine '{engine_id}'")
    return engine


def list_engine_ids() -> List[str]:
    return list(_ENGINES.keys())


def all_engines() -> List[EmbeddingEngine]:
    return list(_ENGINES.values())


def describe_models() -> List[Dict[str, Any]]:
    """Non-secret metadata + readiness for every registered engine."""
    descriptions: List[Dict[str, Any]] = []
    for engine in all_engines():
        readiness = engine.readiness().to_dict()
        provenance = engine.provenance().to_dict()
        descriptions.append(
            {
                "engine": engine.engine_id,
                "modelId": provenance["modelId"],
                "modelRevision": provenance["modelRevision"],
                "embeddingDimension": provenance["embeddingDimension"],
                "expectedEmbeddingDimension": readiness["expectedEmbeddingDimension"],
                "device": readiness["device"],
                "enabled": readiness["enabled"],
                "loaded": readiness["loaded"],
                "ready": readiness["ready"],
                "preprocessingVersion": provenance["preprocessingVersion"],
                "warmupCompleted": readiness["warmupCompleted"],
                "calibrationStatus": provenance["calibrationStatus"],
                "lastLoadErrorCategory": readiness["lastLoadErrorCategory"],
            }
        )
    return descriptions
