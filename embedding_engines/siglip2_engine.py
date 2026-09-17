"""siglip2_v1 engine: google/siglip2-so400m-patch14-384 via transformers'
AutoProcessor/AutoModel, kept fully independent of the CLIP code path.

Design constraints (see docs/P13_SIGLIP2_AB_IMPLEMENTATION.md):
  - Lazy loading only: importing this module, or building the registry,
    never downloads or loads model weights. Loading happens on first use
    (or explicit warmup) and only when SIGLIP2_ENABLED=true.
  - Any load/inference failure raises EngineUnavailableError with a safe,
    non-traceback error category; it must never take down the CLIP engine
    or the process.
  - SigLIP2 has its own preprocessing (its own image processor config, and
    fixed-length text padding/truncation) -- CLIP's 77-token assumption is
    never reused here.
  - Uses torch.inference_mode() (not the older no_grad()) for inference.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

import anyio
import numpy as np
from PIL import Image

from . import config as cfg
from .base import EmbeddingEngine, EngineProvenance, EngineReadiness, EngineUnavailableError

logger = logging.getLogger(__name__)

ENGINE_ID = "siglip2_v1"

# SigLIP2's own image_processor/tokenizer config is what actually drives
# preprocessing; this key set mirrors app.py's compute_preprocessing_version
# but is computed independently against SigLIP2's real, loaded processor.
_RELEVANT_IMAGE_CONFIG_KEYS = (
    "size",
    "resample",
    "do_resize",
    "do_rescale",
    "rescale_factor",
    "do_normalize",
    "image_mean",
    "image_std",
)


class Siglip2Engine(EmbeddingEngine):
    engine_id = ENGINE_ID

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()
        self._embedding_dimension: Optional[int] = None
        self._preprocessing_version: Optional[str] = None
        self._load_error_category: Optional[str] = None
        self._warmup_completed = False
        # Set the first time a real encode_image/encode_text call fully
        # succeeds (see _normalize). Distinct from _warmup_completed, which
        # is reserved for an explicit startup/manual warmup() call -- with
        # SIGLIP2_LOAD_ON_START=false (the default), warmup() never runs,
        # but the engine is still genuinely ready once a normal on-demand
        # inference has actually worked (see readiness() below).
        self._inference_succeeded = False
        self._inference_limiter = anyio.CapacityLimiter(1)

    # -- configuration -----------------------------------------------
    def is_enabled(self) -> bool:
        return cfg.SIGLIP2_ENABLED

    @property
    def model_id(self) -> str:
        return cfg.SIGLIP2_MODEL_ID

    @property
    def model_revision(self) -> Optional[str]:
        return cfg.SIGLIP2_MODEL_REVISION

    @property
    def expected_dimension(self) -> int:
        return cfg.SIGLIP2_EXPECTED_DIMENSION

    @property
    def device(self) -> str:
        return cfg.SIGLIP2_DEVICE

    @property
    def calibration_status(self) -> str:
        return cfg.SIGLIP2_CALIBRATION_STATUS

    def _configured_device(self) -> str:
        device = cfg.SIGLIP2_DEVICE
        if device not in {"cpu", "cuda"}:
            raise EngineUnavailableError(
                "SIGLIP2_DEVICE must be explicitly 'cpu' or 'cuda'", error_category="invalid_device_config"
            )
        if device == "cuda":
            try:
                import torch
            except ImportError as exc:
                raise EngineUnavailableError(
                    "CUDA was configured for SigLIP2 but PyTorch is unavailable",
                    error_category="dependency_missing",
                ) from exc
            if not torch.cuda.is_available():
                raise EngineUnavailableError(
                    "CUDA was configured for SigLIP2 but no CUDA device is available",
                    error_category="cuda_unavailable",
                )
        return device

    # -- loading -------------------------------------------------------
    def load(self) -> None:
        if not self.is_enabled():
            raise EngineUnavailableError("SigLIP2 is disabled by configuration", error_category="engine_disabled")
        if self._model is not None and self._processor is not None:
            return
        with self._lock:
            if self._model is not None and self._processor is not None:
                return
            try:
                from transformers import AutoModel, AutoProcessor
            except ImportError as exc:
                self._load_error_category = "dependency_missing"
                raise EngineUnavailableError(
                    "SigLIP2 model dependencies are not installed", error_category="dependency_missing"
                ) from exc
            load_started = time.time()
            try:
                device = self._configured_device()
                processor = AutoProcessor.from_pretrained(self.model_id, revision=self.model_revision)
                model = AutoModel.from_pretrained(self.model_id, revision=self.model_revision)
                if hasattr(model, "to"):
                    model = model.to(device)
                if hasattr(model, "eval"):
                    model.eval()
                self._processor = processor
                self._model = model
                self._preprocessing_version = self._compute_preprocessing_version(processor)
                self._load_error_category = None
                load_seconds = time.time() - load_started
                logger.info(
                    json.dumps(
                        {
                            "event": "model_load",
                            "engine": self.engine_id,
                            "modelId": self.model_id,
                            "modelRevision": self.model_revision,
                            "preprocessingVersion": self._preprocessing_version,
                            "device": device,
                            "loadSeconds": round(load_seconds, 3),
                        }
                    )
                )
                self._record_load_metric(load_seconds)
            except EngineUnavailableError:
                raise
            except Exception as exc:  # pragma: no cover - exercised via mocked failure tests
                self._load_error_category = type(exc).__name__
                logger.error(
                    json.dumps(
                        {
                            "event": "model_failure",
                            "engine": self.engine_id,
                            "modelId": self.model_id,
                            "modelRevision": self.model_revision,
                            "errorType": type(exc).__name__,
                        }
                    )
                )
                raise EngineUnavailableError(
                    f"Failed to load SigLIP2: {type(exc).__name__}", error_category="model_load_failed"
                ) from exc

    @staticmethod
    def _record_load_metric(duration_s: float) -> None:
        """Best-effort Prometheus observation. Deferred import to avoid a
        circular dependency on app.py; must never raise, since a metrics
        recording failure must not affect a real model load."""
        try:
            import app as app_module

            app_module.record_engine_load_seconds(ENGINE_ID, duration_s)
        except Exception:  # pragma: no cover - defensive only
            pass

    @staticmethod
    def _compute_preprocessing_version(processor_instance: Any) -> Optional[str]:
        image_processor = getattr(processor_instance, "image_processor", None) or processor_instance
        try:
            config = image_processor.to_dict() if hasattr(image_processor, "to_dict") else vars(image_processor)
        except Exception:  # pragma: no cover - defensive; a config dump should never fail
            return None
        fingerprint_source = {key: config[key] for key in _RELEVANT_IMAGE_CONFIG_KEYS if key in config}
        if not fingerprint_source:
            return None
        serialized = json.dumps(fingerprint_source, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _text_max_tokens(self) -> int:
        tokenizer = getattr(self._processor, "tokenizer", None)
        configured = getattr(tokenizer, "model_max_length", None) if tokenizer is not None else None
        if isinstance(configured, int) and 8 <= configured <= 256:
            return configured
        # SigLIP2's own training used a fixed, padded sequence length distinct
        # from CLIP's 77-token context -- never fall back to CLIP's constant.
        return cfg.SIGLIP2_TEXT_MAX_TOKENS

    @staticmethod
    def _feature_tensor(value: Any) -> Any:
        """Normalize get_text_features()/get_image_features() return shapes
        across transformers versions: some implementations return a raw
        tensor directly; others return a ModelOutput (e.g.
        BaseModelOutputWithPooling, observed on the Azure canary) whose
        projected embedding lives in `.pooler_output`. This exists ONLY to
        resolve that shape difference -- it never picks last_hidden_state
        when a projected pooler_output exists, and it never guesses at an
        arbitrary tuple/mapping member. Raises EngineUnavailableError
        (error_category="invalid_output") if no usable tensor can be found,
        rather than passing an unsupported object into NumPy."""
        pooler_output = getattr(value, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        if isinstance(value, Mapping):
            mapped = value.get("pooler_output")
            if mapped is not None:
                return mapped
        if hasattr(value, "shape") or hasattr(value, "detach") or hasattr(value, "numpy"):
            # Already a tensor/ndarray -- return unchanged.
            return value
        raise EngineUnavailableError(
            "SigLIP2 returned no usable pooled feature tensor (no pooler_output and not a tensor)",
            error_category="invalid_output",
        )

    def _normalize(self, features: Any) -> List[float]:
        array = features
        if hasattr(array, "detach"):
            array = array.detach()
        if hasattr(array, "cpu"):
            array = array.cpu()
        if hasattr(array, "numpy"):
            array = array.numpy()
        array = np.asarray(array, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] == 0:
            raise EngineUnavailableError("SigLIP2 produced a malformed embedding", error_category="invalid_output")
        vector = array[0]
        if not np.isfinite(vector).all():
            raise EngineUnavailableError("SigLIP2 produced a non-finite embedding", error_category="invalid_output")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise EngineUnavailableError("SigLIP2 produced a zero-norm embedding", error_category="invalid_output")
        normalized = (vector / norm).tolist()
        self._embedding_dimension = len(normalized)
        if self._embedding_dimension != self.expected_dimension:
            raise EngineUnavailableError(
                f"SigLIP2 embedding dimension {self._embedding_dimension} does not match the "
                f"expected dimension {self.expected_dimension}",
                error_category="dimension_mismatch",
            )
        self._inference_succeeded = True
        return [float(value) for value in normalized]

    def _move_inputs(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        device = self._configured_device()
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    # -- inference -------------------------------------------------------
    def encode_image(self, image: Image.Image) -> List[float]:
        self.load()
        try:
            import torch

            inputs = self._move_inputs(self._processor(images=image, return_tensors="pt"))
            with torch.inference_mode():
                features = self._feature_tensor(self._model.get_image_features(**inputs))
        except EngineUnavailableError:
            raise
        except Exception as exc:
            raise EngineUnavailableError(
                f"SigLIP2 image inference failed: {type(exc).__name__}", error_category="inference_failed"
            ) from exc
        return self._normalize(features)

    def encode_text(self, text: str) -> List[float]:
        self.load()
        try:
            import torch

            inputs = self._move_inputs(
                self._processor(
                    text=[text],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self._text_max_tokens(),
                )
            )
            with torch.inference_mode():
                features = self._feature_tensor(self._model.get_text_features(**inputs))
        except EngineUnavailableError:
            raise
        except Exception as exc:
            raise EngineUnavailableError(
                f"SigLIP2 text inference failed: {type(exc).__name__}", error_category="inference_failed"
            ) from exc
        return self._normalize(features)

    async def encode_image_async(self, image: Image.Image) -> List[float]:
        try:
            with anyio.fail_after(max(0.1, cfg.SIGLIP2_INFERENCE_TIMEOUT_MS / 1000.0)):
                return await anyio.to_thread.run_sync(
                    self.encode_image, image, limiter=self._inference_limiter, abandon_on_cancel=True
                )
        except TimeoutError as exc:
            raise EngineUnavailableError("SigLIP2 image inference timed out", error_category="timeout") from exc

    async def encode_text_async(self, text: str) -> List[float]:
        try:
            with anyio.fail_after(max(0.1, cfg.SIGLIP2_INFERENCE_TIMEOUT_MS / 1000.0)):
                return await anyio.to_thread.run_sync(
                    self.encode_text, text, limiter=self._inference_limiter, abandon_on_cancel=True
                )
        except TimeoutError as exc:
            raise EngineUnavailableError("SigLIP2 text inference timed out", error_category="timeout") from exc

    def warmup(self) -> None:
        """Best-effort warmup; never raises. A SigLIP2 warmup failure must
        never affect process startup or the CLIP engine's own readiness."""
        if not self.is_enabled() or not cfg.SIGLIP2_LOAD_ON_START:
            return
        try:
            self.encode_image(Image.new("RGB", (32, 32), "white"))
            self._warmup_completed = True
        except Exception:
            logger.warning("SigLIP2 warmup failed; engine remains lazily loadable on demand", exc_info=True)

    # -- observability -----------------------------------------------
    def readiness(self) -> EngineReadiness:
        loaded = self._model is not None and self._processor is not None
        # "ready" means: enabled, loaded, no current fatal load error, and
        # at least one successful inference -- either an explicit startup/
        # manual warmup() (_warmup_completed) or a real on-demand encode
        # that actually completed (_inference_succeeded). With
        # SIGLIP2_LOAD_ON_START=false (the default), warmup() never runs,
        # so without _inference_succeeded this engine would misleadingly
        # report ready=false forever even after serving real traffic
        # successfully.
        ready = (
            self.is_enabled()
            and loaded
            and self._load_error_category is None
            and (self._warmup_completed or self._inference_succeeded)
        )
        return EngineReadiness(
            engine=self.engine_id,
            enabled=self.is_enabled(),
            loaded=loaded,
            ready=ready,
            warmup_completed=self._warmup_completed,
            device=self.device,
            expected_embedding_dimension=self.expected_dimension,
            embedding_dimension=self._embedding_dimension,
            last_load_error_category=self._load_error_category,
        )

    def provenance(self) -> EngineProvenance:
        return EngineProvenance(
            engine=self.engine_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            embedding_dimension=self._embedding_dimension,
            preprocessing_version=self._preprocessing_version,
            device=self.device,
            calibration_status=self.calibration_status,
            scoring_version=cfg.V2_SCORING_VERSION,
        )
