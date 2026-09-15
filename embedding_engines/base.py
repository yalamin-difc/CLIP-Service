from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image


class EngineUnavailableError(RuntimeError):
    """Raised when an engine cannot load or perform inference right now.

    This is distinct from a Python crash/stack trace: callers (the V2 API,
    the A/B endpoint) catch this specifically and turn it into a structured
    ``status=unavailable`` response with a safe error code, never a raw
    traceback. Failure of one engine (e.g. SigLIP2) must never propagate
    into another engine's (CLIP's) request path.
    """

    def __init__(self, message: str, *, error_category: str = "engine_unavailable"):
        super().__init__(message)
        self.error_category = error_category


@dataclass(frozen=True)
class EngineProvenance:
    engine: str
    model_id: str
    model_revision: Optional[str]
    embedding_dimension: Optional[int]
    preprocessing_version: Optional[str]
    device: str
    calibration_status: str
    scoring_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "modelId": self.model_id,
            "modelRevision": self.model_revision,
            "embeddingDimension": self.embedding_dimension,
            "preprocessingVersion": self.preprocessing_version,
            "device": self.device,
            "calibrationStatus": self.calibration_status,
            "scoringVersion": self.scoring_version,
        }


@dataclass(frozen=True)
class EngineReadiness:
    engine: str
    enabled: bool
    loaded: bool
    ready: bool
    warmup_completed: bool
    device: str
    expected_embedding_dimension: int
    embedding_dimension: Optional[int]
    last_load_error_category: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "enabled": self.enabled,
            "loaded": self.loaded,
            "ready": self.ready,
            "warmupCompleted": self.warmup_completed,
            "device": self.device,
            "expectedEmbeddingDimension": self.expected_embedding_dimension,
            "embeddingDimension": self.embedding_dimension,
            "embeddingDimensionMatchesExpected": (
                self.embedding_dimension == self.expected_embedding_dimension
                if self.embedding_dimension is not None
                else False
            ),
            "lastLoadErrorCategory": self.last_load_error_category,
        }


class EmbeddingEngine(abc.ABC):
    """Common surface every model engine (clip_v1, siglip2_v1, ...) exposes.

    Implementations must return normalized (unit-length) vectors suitable
    for cosine similarity, and must never silently fall back to another
    engine's model or vector space.
    """

    engine_id: str

    @abc.abstractmethod
    def is_enabled(self) -> bool:
        """Whether this engine is turned on by configuration."""

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights if not already loaded. Idempotent."""

    @abc.abstractmethod
    def readiness(self) -> EngineReadiness:
        """Non-secret readiness/health snapshot for this engine."""

    @abc.abstractmethod
    def provenance(self) -> EngineProvenance:
        """Model provenance to attach to any scored response."""

    @abc.abstractmethod
    def encode_image(self, image: Image.Image) -> List[float]:
        """Synchronous, normalized image embedding. Runs in a worker thread by callers."""

    @abc.abstractmethod
    def encode_text(self, text: str) -> List[float]:
        """Synchronous, normalized text embedding. Runs in a worker thread by callers."""

    async def encode_image_async(self, image: Image.Image) -> List[float]:
        raise NotImplementedError

    async def encode_text_async(self, text: str) -> List[float]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model_id(self) -> str: ...

    @property
    @abc.abstractmethod
    def model_revision(self) -> Optional[str]: ...

    @property
    @abc.abstractmethod
    def expected_dimension(self) -> int: ...

    @property
    @abc.abstractmethod
    def device(self) -> str: ...

    @property
    def calibration_status(self) -> str:
        return "uncalibrated"
