"""P13: unit tests for the embedding_engines package -- the clip_v1/
siglip2_v1 abstraction, the model registry, and the NumPy vector index.
These do not require real model weights or network access: siglip2_v1's
loading/inference is exercised through its own real code paths but with
transformers' AutoModel/AutoProcessor mocked out, exactly like
tests/test_app.py already mocks CLIP's load_model for the legacy suite.
"""
import contextlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


class _FakeTorchModule(types.ModuleType):
    """Stands in for `torch` in environments where it isn't installed (as
    in this sandbox), so siglip2_engine's `with torch.inference_mode():`
    can be exercised without a real PyTorch install. Production always has
    real torch (it's a hard dependency of transformers' PyTorch backend);
    this only fills the gap created by these tests injecting a fake model/
    processor directly instead of going through the real load()."""

    def inference_mode(self):
        return contextlib.nullcontext()

from embedding_engines import config as engine_config
from embedding_engines.base import EngineUnavailableError
from embedding_engines.clip_engine import ClipEngine
from embedding_engines.registry import UnknownEngineError, describe_models, get_engine, list_engine_ids
from embedding_engines.siglip2_engine import Siglip2Engine
from embedding_engines.vector_index import NumpyVectorIndex


class RegistryTests(unittest.TestCase):
    def test_known_engines_are_registered(self):
        self.assertEqual(set(list_engine_ids()), {"clip_v1", "siglip2_v1"})

    def test_unknown_engine_raises(self):
        with self.assertRaises(UnknownEngineError):
            get_engine("does_not_exist_v9")

    def test_describe_models_never_includes_raw_vectors(self):
        for description in describe_models():
            self.assertNotIn("vector", description)
            self.assertIn("engine", description)
            self.assertIn("calibrationStatus", description)

    def test_clip_default_expected_dimension_is_512(self):
        # Straight from app.py's own constant, unpatched -- the production default.
        import app as clip_service

        self.assertEqual(clip_service.EXPECTED_EMBEDDING_DIMENSION, 512)

    def test_siglip2_default_expected_dimension_is_1152(self):
        self.assertEqual(engine_config.SIGLIP2_EXPECTED_DIMENSION, 1152)

    def test_clip_is_calibrated_and_siglip2_is_uncalibrated(self):
        self.assertEqual(get_engine("clip_v1").calibration_status, "calibrated")
        self.assertEqual(get_engine("siglip2_v1").calibration_status, "uncalibrated")


class ClipEngineDelegationTests(unittest.TestCase):
    """clip_v1 is a thin adapter over app.py's own, unmodified globals --
    these tests prove it reads/writes through to the real production state
    rather than keeping any separate copy."""

    def test_provenance_reflects_live_app_state(self):
        import app as clip_service

        engine = ClipEngine()
        with patch.object(clip_service, "MODEL_NAME", "fake/model"), patch.object(
            clip_service, "MODEL_REVISION", "fake-rev"
        ), patch.object(clip_service, "embedding_dimension", 512), patch.object(
            clip_service, "preprocessing_version", "fp-123"
        ):
            provenance = engine.provenance()
            self.assertEqual(provenance.model_id, "fake/model")
            self.assertEqual(provenance.model_revision, "fake-rev")
            self.assertEqual(provenance.embedding_dimension, 512)
            self.assertEqual(provenance.preprocessing_version, "fp-123")

    def test_is_always_enabled(self):
        self.assertTrue(ClipEngine().is_enabled())


class Siglip2EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = Siglip2Engine()

    def test_disabled_by_default(self):
        self.assertFalse(self.engine.is_enabled())

    def test_load_raises_when_disabled(self):
        with self.assertRaises(EngineUnavailableError) as ctx:
            self.engine.load()
        self.assertEqual(ctx.exception.error_category, "engine_disabled")

    def test_load_raises_when_transformers_missing(self):
        with patch.object(engine_config, "SIGLIP2_ENABLED", True):
            with patch.dict("sys.modules", {"transformers": None}):
                with self.assertRaises(EngineUnavailableError) as ctx:
                    self.engine.load()
                self.assertEqual(ctx.exception.error_category, "dependency_missing")

    def test_cuda_requested_but_unavailable_raises(self):
        with patch.object(engine_config, "SIGLIP2_ENABLED", True), patch.object(engine_config, "SIGLIP2_DEVICE", "cuda"):
            with self.assertRaises(EngineUnavailableError) as ctx:
                self.engine._configured_device()
            self.assertIn(ctx.exception.error_category, {"dependency_missing", "cuda_unavailable"})

    def test_invalid_device_config_raises(self):
        with patch.object(engine_config, "SIGLIP2_DEVICE", "tpu"):
            with self.assertRaises(EngineUnavailableError) as ctx:
                self.engine._configured_device()
            self.assertEqual(ctx.exception.error_category, "invalid_device_config")

    def _load_with_fakes(self, image_vector=None, text_vector=None, image_dim=1152):
        image_vector = image_vector if image_vector is not None else [1.0] * image_dim

        class FakeTokenizer:
            model_max_length = 64

        class FakeImageProcessor:
            def to_dict(self):
                return {"size": {"height": 384, "width": 384}, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}

        class FakeProcessor:
            tokenizer = FakeTokenizer()
            image_processor = FakeImageProcessor()

            def __call__(self, images=None, text=None, **kwargs):
                if images is not None:
                    return {"pixel_values": np.zeros((1, 3, 384, 384))}
                return {"input_ids": np.zeros((1, kwargs.get("max_length", 64)))}

        class FakeModel:
            def get_image_features(self, **inputs):
                return np.asarray([image_vector])

            def get_text_features(self, **inputs):
                return np.asarray([text_vector if text_vector is not None else [1.0] * image_dim])

        self.engine._processor = FakeProcessor()
        self.engine._model = FakeModel()

    def test_encode_image_normalizes_and_records_dimension(self):
        with patch.object(engine_config, "SIGLIP2_ENABLED", True), patch.object(
            engine_config, "SIGLIP2_EXPECTED_DIMENSION", 4
        ), patch.dict(sys.modules, {"torch": _FakeTorchModule("torch")}):
            self._load_with_fakes(image_vector=[3.0, 0.0, 0.0, 0.0], image_dim=4)
            from PIL import Image

            vector = self.engine.encode_image(Image.new("RGB", (8, 8)))
            self.assertEqual(len(vector), 4)
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)
            self.assertEqual(self.engine.readiness().embedding_dimension, 4)

    def test_dimension_mismatch_raises_unavailable(self):
        with patch.object(engine_config, "SIGLIP2_ENABLED", True), patch.object(
            engine_config, "SIGLIP2_EXPECTED_DIMENSION", 1152
        ), patch.dict(sys.modules, {"torch": _FakeTorchModule("torch")}):
            self._load_with_fakes(image_vector=[1.0, 0.0, 0.0, 0.0], image_dim=4)  # wrong dimension on purpose
            from PIL import Image

            with self.assertRaises(EngineUnavailableError) as ctx:
                self.engine.encode_image(Image.new("RGB", (8, 8)))
            self.assertEqual(ctx.exception.error_category, "dimension_mismatch")

    def test_text_padding_uses_fixed_length_not_clips_77(self):
        # SigLIP2's own tokenizer config must drive padding -- never CLIP's
        # 77-token context length constant.
        with patch.object(engine_config, "SIGLIP2_ENABLED", True), patch.object(
            engine_config, "SIGLIP2_EXPECTED_DIMENSION", 4
        ), patch.dict(sys.modules, {"torch": _FakeTorchModule("torch")}):
            self._load_with_fakes(text_vector=[0.0, 3.0, 0.0, 0.0], image_dim=4)
            self.assertEqual(self.engine._text_max_tokens(), 64)
            vector = self.engine.encode_text("hello")
            self.assertEqual(len(vector), 4)

    def test_readiness_reports_disabled_engine(self):
        readiness = self.engine.readiness()
        self.assertFalse(readiness.enabled)
        self.assertFalse(readiness.loaded)
        self.assertFalse(readiness.ready)


class NumpyVectorIndexTests(unittest.TestCase):
    def setUp(self):
        self.index = NumpyVectorIndex()

    def test_tenant_isolation_in_search(self):
        self.index.build(
            "clip_v1",
            [
                ("a", "tenant-1", "site-1", [1.0, 0.0]),
                ("b", "tenant-2", "site-1", [1.0, 0.0]),
            ],
        )
        results = self.index.search("clip_v1", "tenant-1", ["site-1"], [1.0, 0.0], top_k=5)
        ids = [candidate_id for candidate_id, _score in results]
        self.assertEqual(ids, ["a"])

    def test_site_isolation_in_search(self):
        self.index.build(
            "clip_v1",
            [
                ("a", "tenant-1", "site-1", [1.0, 0.0]),
                ("b", "tenant-1", "site-2", [1.0, 0.0]),
            ],
        )
        results = self.index.search("clip_v1", "tenant-1", ["site-1"], [1.0, 0.0], top_k=5)
        ids = [candidate_id for candidate_id, _score in results]
        self.assertEqual(ids, ["a"])

    def test_engines_are_isolated_from_each_other(self):
        self.index.build("clip_v1", [("a", "t1", "s1", [1.0, 0.0])])
        self.index.build("siglip2_v1", [("b", "t1", "s1", [0.0, 1.0, 0.0])])
        self.assertEqual(self.index.stats("clip_v1")["dimension"], 2)
        self.assertEqual(self.index.stats("siglip2_v1")["dimension"], 3)
        # Searching one engine never returns the other engine's rows.
        results = self.index.search("clip_v1", "t1", ["s1"], [1.0, 0.0], top_k=5)
        self.assertEqual([cid for cid, _ in results], ["a"])

    def test_build_rejects_mixed_dimensions(self):
        with self.assertRaises(ValueError):
            self.index.build("clip_v1", [("a", "t1", "s1", [1.0, 0.0]), ("b", "t1", "s1", [1.0, 0.0, 0.0])])

    def test_add_update_remove(self):
        self.index.build("clip_v1", [])
        self.index.add_or_update("clip_v1", "a", "t1", "s1", [1.0, 0.0])
        self.assertEqual(self.index.stats("clip_v1")["vectors"], 1)
        self.index.add_or_update("clip_v1", "a", "t1", "s1", [0.0, 1.0])
        self.assertEqual(self.index.stats("clip_v1")["vectors"], 1)
        results = self.index.search("clip_v1", "t1", ["s1"], [0.0, 1.0], top_k=1)
        self.assertAlmostEqual(results[0][1], 1.0, places=5)
        self.assertTrue(self.index.remove("clip_v1", "t1", "a"))
        self.assertEqual(self.index.stats("clip_v1")["vectors"], 0)
        self.assertFalse(self.index.remove("clip_v1", "t1", "a"))


if __name__ == "__main__":
    unittest.main()
