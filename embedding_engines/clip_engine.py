"""clip_v1 engine adapter.

This is a thin wrapper around app.py's existing, unmodified CLIP loading
and inference code (load_model/image_embedding_for/text_embedding_for/
run_image_embedding/run_text_embedding/runtime_provenance/model_health).
It exists so the registry and the /v2 API can address CLIP through the
same EmbeddingEngine interface as siglip2_v1, without moving or altering
a single line of the production CLIP path that the legacy endpoints and
their existing test suite already depend on.

The import of `app` is deferred to call time (not module import time) to
avoid a circular import: app.py imports this package to build the
registry, so this module cannot import app.py at the top level.
"""
from __future__ import annotations

from typing import List, Optional

from PIL import Image

from .base import EmbeddingEngine, EngineProvenance, EngineReadiness

ENGINE_ID = "clip_v1"


def _app():
    import app as app_module  # noqa: PLC0415 - deferred to avoid a circular import

    return app_module


class ClipEngine(EmbeddingEngine):
    engine_id = ENGINE_ID

    def is_enabled(self) -> bool:
        # clip_v1 is the existing production engine; it is always enabled
        # and is never gated behind a feature flag.
        return True

    def load(self) -> None:
        _app().load_model()

    @property
    def model_id(self) -> str:
        return _app().MODEL_NAME

    @property
    def model_revision(self) -> Optional[str]:
        return _app().MODEL_REVISION

    @property
    def expected_dimension(self) -> int:
        return _app().EXPECTED_EMBEDDING_DIMENSION

    @property
    def device(self) -> str:
        return _app().INFERENCE_DEVICE

    @property
    def calibration_status(self) -> str:
        # CLIP's existing minScore/minMargin/temperature thresholds are the
        # production-calibrated ones already in use by legacy /match.
        return "calibrated"

    def encode_image(self, image: Image.Image) -> List[float]:
        return _app().image_embedding_for(image)

    def encode_text(self, text: str) -> List[float]:
        return _app().text_embedding_for(text)

    async def encode_image_async(self, image: Image.Image) -> List[float]:
        return await _app().run_image_embedding(image)

    async def encode_text_async(self, text: str) -> List[float]:
        return await _app().run_text_embedding(text)

    def readiness(self) -> EngineReadiness:
        app_module = _app()
        loaded = app_module.model is not None and app_module.processor is not None
        return EngineReadiness(
            engine=self.engine_id,
            enabled=True,
            loaded=loaded,
            ready=loaded and app_module.model_warmup_completed,
            warmup_completed=app_module.model_warmup_completed,
            device=app_module.INFERENCE_DEVICE,
            expected_embedding_dimension=app_module.EXPECTED_EMBEDDING_DIMENSION,
            embedding_dimension=app_module.embedding_dimension,
            last_load_error_category=app_module.model_load_error,
        )

    def provenance(self) -> EngineProvenance:
        app_module = _app()
        return EngineProvenance(
            engine=self.engine_id,
            model_id=app_module.MODEL_NAME,
            model_revision=app_module.MODEL_REVISION,
            embedding_dimension=app_module.embedding_dimension,
            preprocessing_version=app_module.preprocessing_version,
            device=app_module.INFERENCE_DEVICE,
            calibration_status=self.calibration_status,
            scoring_version=app_module.SCORING_VERSION,
        )
