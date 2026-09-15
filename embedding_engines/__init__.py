"""Model-engine abstraction for the CLIP service (P13).

Exposes a common ``EmbeddingEngine`` interface implemented by ``clip_v1``
(the existing production CLIP model) and ``siglip2_v1`` (the new
experimental SigLIP2 model), plus a small registry used by the ``/v2``
API surface. Nothing in this package changes the behaviour of the legacy
top-level endpoints in ``app.py``.
"""

from .base import EmbeddingEngine, EngineProvenance, EngineReadiness, EngineUnavailableError
from .registry import describe_models, get_engine, list_engine_ids

__all__ = [
    "EmbeddingEngine",
    "EngineProvenance",
    "EngineReadiness",
    "EngineUnavailableError",
    "describe_models",
    "get_engine",
    "list_engine_ids",
]
