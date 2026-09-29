"""
D-3: SIGLIP2_MODEL_REVISION defaulted to "main" -- a real, resolvable git
ref, but a *mutable* one, not a commit pin -- and nothing enforced that an
operator had actually pinned it before setting SIGLIP2_ENABLED=true. A
misconfigured deployment could silently serve a SigLIP2 model that drifts
to different weights between container restarts, or even between two
requests to a long-running process if the upstream ref moves and the
local HF cache is later evicted and re-pulled.

validate_siglip2_configuration() (embedding_engines/config.py), wired into
app.py's initialize_runtime() unconditionally and before any engine
warmup, now turns "SigLIP2 enabled + unpinned revision" into a hard
startup failure -- the same category of check as
internal_auth.validate_auth_configuration() and app.py's
configured_device(), which already fail startup rather than log-and-
continue.

Siglip2Engine.load() (embedding_engines/siglip2_engine.py) is also the
startup canary described in its own warmup()/docstring (triggered from
app.py's initialize_runtime() -> warmup_optional_engines() when
SIGLIP2_LOAD_ON_START=true): it now additionally computes and logs a
weight checksum -- a fingerprint of the *actual loaded weight tensors*,
never the configured string alone -- alongside the resolved revision.
"""
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from embedding_engines import config as engine_config
from embedding_engines.siglip2_engine import Siglip2Engine

import app as clip_service
import internal_auth


TEST_SECRET = "festival-test-secret-with-at-least-32-bytes"
REAL_LOOKING_SHA = "f4a1c9e2b6d7a08351e6c2f9d4b7a1e0c3f6a9d2"


class ValidateSiglip2ConfigurationTests(unittest.TestCase):
    def test_disabled_never_raises_regardless_of_revision(self):
        for revision in ("", "main", "MAIN", "latest", "LATEST", REAL_LOOKING_SHA):
            with patch.object(engine_config, "SIGLIP2_ENABLED", False), patch.object(
                engine_config, "SIGLIP2_MODEL_REVISION", revision
            ):
                engine_config.validate_siglip2_configuration()  # must not raise

    def test_enabled_with_empty_revision_raises(self):
        with patch.object(engine_config, "SIGLIP2_ENABLED", True), patch.object(
            engine_config, "SIGLIP2_MODEL_REVISION", ""
        ):
            with self.assertRaises(RuntimeError) as ctx:
                engine_config.validate_siglip2_configuration()
            self.assertIn("SIGLIP2_MODEL_REVISION", str(ctx.exception))

    def test_enabled_with_main_raises(self):
        with patch.object(engine_config, "SIGLIP2_ENABLED", True), patch.object(
            engine_config, "SIGLIP2_MODEL_REVISION", "main"
        ):
            with self.assertRaises(RuntimeError):
                engine_config.validate_siglip2_configuration()

    def test_enabled_with_latest_raises(self):
        with patch.object(engine_config, "SIGLIP2_ENABLED", True), patch.object(
            engine_config, "SIGLIP2_MODEL_REVISION", "latest"
        ):
            with self.assertRaises(RuntimeError):
                engine_config.validate_siglip2_configuration()

    def test_case_insensitive_match(self):
        for revision in ("Main", "MAIN", "Latest", "LATEST"):
            with patch.object(engine_config, "SIGLIP2_ENABLED", True), patch.object(
                engine_config, "SIGLIP2_MODEL_REVISION", revision
            ):
                with self.assertRaises(RuntimeError):
                    engine_config.validate_siglip2_configuration()

    def test_enabled_with_real_pinned_sha_does_not_raise(self):
        with patch.object(engine_config, "SIGLIP2_ENABLED", True), patch.object(
            engine_config, "SIGLIP2_MODEL_REVISION", REAL_LOOKING_SHA
        ):
            engine_config.validate_siglip2_configuration()  # must not raise


class InitializeRuntimeRejectsUnpinnedSiglip2Tests(unittest.TestCase):
    """Proves the check is actually wired into app startup -- not just
    unit-testable in isolation -- and that it runs before any model
    warmup, so a misconfigured process never starts accepting traffic."""

    def setUp(self):
        self.auth_patch = patch.object(internal_auth, "INTERNAL_JWT_SECRET", TEST_SECRET)
        self.auth_patch.start()
        self.device_patch = patch.object(clip_service, "INFERENCE_DEVICE", "cpu")
        self.device_patch.start()
        self.warmup_model_patch = patch.object(clip_service, "warmup_model")
        self.mock_warmup_model = self.warmup_model_patch.start()
        self.warmup_optional_patch = patch.object(clip_service, "warmup_optional_engines")
        self.mock_warmup_optional = self.warmup_optional_patch.start()

    def tearDown(self):
        self.warmup_optional_patch.stop()
        self.warmup_model_patch.stop()
        self.device_patch.stop()
        self.auth_patch.stop()

    def test_startup_rejects_enabled_unpinned_siglip2_before_any_warmup(self):
        with patch.object(engine_config, "SIGLIP2_ENABLED", True), patch.object(
            engine_config, "SIGLIP2_MODEL_REVISION", "main"
        ):
            with self.assertRaises(RuntimeError) as ctx:
                clip_service.initialize_runtime()
            self.assertIn("SIGLIP2_MODEL_REVISION", str(ctx.exception))
        # The whole point of checking this before database/OCR/model
        # warmup is that a misconfigured process never even attempts to
        # warm up a model -- prove neither warmup ever ran.
        self.mock_warmup_model.assert_not_called()
        self.mock_warmup_optional.assert_not_called()

    def test_startup_with_siglip2_disabled_is_unaffected_by_unpinned_revision(self):
        # SIGLIP2_ENABLED defaults to False in every real deployment that
        # hasn't opted in -- confirms this new check never blocks the
        # ordinary (SigLIP2-disabled) startup path just because the
        # placeholder "main" default is still sitting in the environment.
        with patch.object(engine_config, "SIGLIP2_ENABLED", False), patch.object(
            engine_config, "SIGLIP2_MODEL_REVISION", "main"
        ), patch.object(clip_service, "get_repository") as mock_get_repository:
            mock_get_repository.return_value.health.return_value = {"ok": True}
            with patch.object(clip_service, "OCR_ENABLED", False):
                clip_service.initialize_runtime()  # must not raise
        self.mock_warmup_model.assert_called_once()
        self.mock_warmup_optional.assert_called_once()


class _FakeTorchModule(types.ModuleType):
    def inference_mode(self):
        import contextlib

        return contextlib.nullcontext()


class _FakeTensor:
    """Minimal stand-in for a torch.Tensor's weight-checksum-relevant
    surface: .shape, .dtype, .detach(), .cpu(), .numpy(), and slicing."""

    def __init__(self, array):
        self._array = np.asarray(array)

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def reshape(self, *shape):
        return _FakeTensor(self._array.reshape(*shape))

    def __getitem__(self, key):
        return _FakeTensor(self._array[key])


class WeightChecksumTests(unittest.TestCase):
    def setUp(self):
        self.engine = Siglip2Engine()

    def test_returns_none_when_model_has_no_state_dict(self):
        class NoStateDict:
            pass

        self.assertIsNone(self.engine._compute_weight_checksum(NoStateDict()))

    def test_returns_none_when_state_dict_raises(self):
        class ExplodingStateDict:
            def state_dict(self):
                raise RuntimeError("boom")

        self.assertIsNone(self.engine._compute_weight_checksum(ExplodingStateDict()))

    def test_deterministic_for_identical_weights(self):
        class FakeModel:
            def __init__(self, seed):
                rng = np.random.RandomState(seed)
                self._weights = {
                    "layer.weight": _FakeTensor(rng.rand(16)),
                    "layer.bias": _FakeTensor(rng.rand(4)),
                }

            def state_dict(self):
                return self._weights

        checksum_a = self.engine._compute_weight_checksum(FakeModel(seed=1))
        checksum_b = self.engine._compute_weight_checksum(FakeModel(seed=1))
        self.assertIsNotNone(checksum_a)
        self.assertEqual(checksum_a, checksum_b)

    def test_different_for_different_weights(self):
        class FakeModel:
            def __init__(self, seed):
                rng = np.random.RandomState(seed)
                self._weights = {"layer.weight": _FakeTensor(rng.rand(16))}

            def state_dict(self):
                return self._weights

        checksum_a = self.engine._compute_weight_checksum(FakeModel(seed=1))
        checksum_b = self.engine._compute_weight_checksum(FakeModel(seed=2))
        self.assertIsNotNone(checksum_a)
        self.assertIsNotNone(checksum_b)
        self.assertNotEqual(checksum_a, checksum_b)


class StartupCanaryReportsResolvedRevisionAndChecksumTests(unittest.TestCase):
    """'with a pinned revision the canary reports the SHA' -- exercises the
    real load() path (mocked transformers, exactly like
    test_load_raises_when_transformers_missing already does) rather than
    bypassing it by setting engine._model/_processor directly."""

    def setUp(self):
        self.engine = Siglip2Engine()

    def _install_fake_transformers(self, weights):
        class FakeImageProcessor:
            def to_dict(self):
                return {"size": {"height": 384, "width": 384}}

        class FakeProcessor:
            image_processor = FakeImageProcessor()

        class FakeModel:
            def to(self, device):
                return self

            def eval(self):
                return self

            def state_dict(self):
                return weights

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeProcessor())
        fake_transformers.AutoModel = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel())
        return fake_transformers

    def test_canary_logs_resolved_revision_and_a_real_weight_checksum(self):
        weights = {"layer.weight": _FakeTensor(np.array([1.0, 2.0, 3.0, 4.0]))}
        fake_transformers = self._install_fake_transformers(weights)
        with patch.object(engine_config, "SIGLIP2_ENABLED", True), patch.object(
            engine_config, "SIGLIP2_MODEL_REVISION", REAL_LOOKING_SHA
        ), patch.dict(sys.modules, {"transformers": fake_transformers}):
            with self.assertLogs("embedding_engines.siglip2_engine", level="INFO") as logs:
                self.engine.load()
            model_load_lines = [line for line in logs.output if '"event": "model_load"' in line]
            self.assertEqual(len(model_load_lines), 1)
            log_line = model_load_lines[0]
            self.assertIn(f'"modelRevision": "{REAL_LOOKING_SHA}"', log_line)
            self.assertIn('"weightChecksum": "', log_line)
            self.assertNotIn('"weightChecksum": null', log_line)
        self.assertIsNotNone(self.engine._weight_checksum)
        self.assertEqual(len(self.engine._weight_checksum), 16)


if __name__ == "__main__":
    unittest.main()
