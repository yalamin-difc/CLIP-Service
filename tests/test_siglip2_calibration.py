"""D-4: SigLIP2's calibrationStatus used to be a hardcoded constant --
"uncalibrated" forever, with no way for a real scripts/calibrate.py run
to ever be reflected. This proves:

  * embedding_engines.config's calibration-file loader/getters correctly
    load, validate, and fall back (missing file, malformed JSON, missing
    keys, unreadable values -- never a fabricated calibration).
  * SIGLIP2_MIN_SCORE/SIGLIP2_MIN_MARGIN (an explicit operator override)
    always wins over calibration.json when both are present.
  * Siglip2Engine.calibration_status flips from "uncalibrated" to
    "calibrated:<date>" purely based on the file's real presence/content
    -- the exact test D-4 asks for.
  * v2_router._decision_for_engine no longer hardcodes "uncalibrated" in
    its siglip2_v1 branch (a real bug this ticket also fixes) -- it now
    reports the engine's real calibration status either way.
  * CLIP's own CONF_MIN_SCORE/CONF_MIN_MARGIN and clip_v1's "calibrated"
    status are completely untouched by any of this.
"""
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from embedding_engines import config as engine_config
from embedding_engines.siglip2_engine import Siglip2Engine

import v2_router


def _write_calibration(path, min_score=0.55, min_margin=0.12, eval_date="2026-09-29", **overrides):
    payload = {"minScore": min_score, "minMargin": min_margin, "evalDate": eval_date}
    payload.update(overrides)
    pathlib.Path(path).write_text(json.dumps(payload), encoding="utf-8")


class LoadCalibrationFileTests(unittest.TestCase):
    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(pathlib.Path(tmp) / "does-not-exist.json")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", missing):
                self.assertIsNone(engine_config._load_siglip2_calibration_file())

    def test_valid_file_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / "calibration.json")
            _write_calibration(path, min_score=0.6, min_margin=0.2, eval_date="2026-01-15")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", path):
                loaded = engine_config._load_siglip2_calibration_file()
            self.assertEqual(loaded, {"minScore": 0.6, "minMargin": 0.2, "evalDate": "2026-01-15"})

    def test_malformed_json_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "calibration.json"
            path.write_text("not valid json {{{", encoding="utf-8")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", str(path)):
                self.assertIsNone(engine_config._load_siglip2_calibration_file())

    def test_missing_required_key_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "calibration.json"
            path.write_text(json.dumps({"minScore": 0.5}), encoding="utf-8")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", str(path)):
                self.assertIsNone(engine_config._load_siglip2_calibration_file())

    def test_non_numeric_score_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "calibration.json"
            path.write_text(json.dumps({"minScore": "high", "minMargin": 0.1, "evalDate": "2026-01-01"}), encoding="utf-8")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", str(path)):
                self.assertIsNone(engine_config._load_siglip2_calibration_file())

    def test_empty_eval_date_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "calibration.json"
            path.write_text(json.dumps({"minScore": 0.5, "minMargin": 0.1, "evalDate": "  "}), encoding="utf-8")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", str(path)):
                self.assertIsNone(engine_config._load_siglip2_calibration_file())

    def test_json_that_is_not_an_object_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "calibration.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", str(path)):
                self.assertIsNone(engine_config._load_siglip2_calibration_file())


class CalibrationStatusTests(unittest.TestCase):
    """The exact behavior D-4 asks for: 'uncalibrated' without the file,
    'calibrated:<date>' with it."""

    def test_uncalibrated_without_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(pathlib.Path(tmp) / "does-not-exist.json")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", missing):
                self.assertEqual(engine_config.get_siglip2_calibration_status(), "uncalibrated")

    def test_calibrated_with_date_when_file_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / "calibration.json")
            _write_calibration(path, eval_date="2026-03-04")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", path):
                self.assertEqual(engine_config.get_siglip2_calibration_status(), "calibrated:2026-03-04")

    def test_reverts_to_uncalibrated_if_file_is_removed(self):
        # Re-read on every call (not cached at import time) -- proves a
        # calibration.json that disappears while the process is running is
        # reflected immediately, never a stale "calibrated" forever.
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "calibration.json"
            _write_calibration(str(path))
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", str(path)):
                self.assertTrue(engine_config.get_siglip2_calibration_status().startswith("calibrated:"))
                path.unlink()
                self.assertEqual(engine_config.get_siglip2_calibration_status(), "uncalibrated")


class MinScoreMinMarginPrecedenceTests(unittest.TestCase):
    def test_none_when_neither_env_nor_file_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(pathlib.Path(tmp) / "does-not-exist.json")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", missing), patch.object(
                engine_config, "SIGLIP2_MIN_SCORE", None
            ):
                self.assertIsNone(engine_config.get_siglip2_min_score())
            self.assertEqual(engine_config.get_siglip2_min_margin(), 0.0)

    def test_file_value_used_when_env_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / "calibration.json")
            _write_calibration(path, min_score=0.61, min_margin=0.17)
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", path), patch.object(
                engine_config, "SIGLIP2_MIN_SCORE", None
            ), patch.object(engine_config, "SIGLIP2_MIN_MARGIN", None):
                self.assertEqual(engine_config.get_siglip2_min_score(), 0.61)
                self.assertEqual(engine_config.get_siglip2_min_margin(), 0.17)

    def test_explicit_env_override_wins_over_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / "calibration.json")
            _write_calibration(path, min_score=0.61, min_margin=0.17)
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", path), patch.object(
                engine_config, "SIGLIP2_MIN_SCORE", "0.9"
            ), patch.object(engine_config, "SIGLIP2_MIN_MARGIN", "0.4"):
                self.assertEqual(engine_config.get_siglip2_min_score(), 0.9)
                self.assertEqual(engine_config.get_siglip2_min_margin(), 0.4)


class Siglip2EngineCalibrationStatusTests(unittest.TestCase):
    """The literal test D-4 asks for, exercised through the real engine
    object (not just the config module)."""

    def setUp(self):
        self.engine = Siglip2Engine()

    def test_engine_reports_uncalibrated_without_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(pathlib.Path(tmp) / "does-not-exist.json")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", missing):
                self.assertEqual(self.engine.calibration_status, "uncalibrated")

    def test_engine_reports_calibrated_with_date_when_file_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / "calibration.json")
            _write_calibration(path, eval_date="2026-05-20")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", path):
                self.assertEqual(self.engine.calibration_status, "calibrated:2026-05-20")

    def test_provenance_reflects_the_same_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / "calibration.json")
            _write_calibration(path, eval_date="2026-05-20")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", path):
                self.assertEqual(self.engine.provenance().calibration_status, "calibrated:2026-05-20")


class DecisionForEngineCalibrationStatusTests(unittest.TestCase):
    """v2_router._decision_for_engine's siglip2_v1 branch used to hardcode
    "uncalibrated" in its response details even when a real threshold was
    already gating -- confirms it now reports the real status."""

    def test_clip_v1_is_untouched(self):
        # Sanity: CLIP's own branch never reads SIGLIP2_* config at all.
        with patch("app.CONF_MIN_SCORE", 0.22), patch("app.CONF_MIN_MARGIN", 0.03):
            decision = v2_router._decision_for_engine("clip_v1", 0.5, 0.1)
        self.assertNotIn("calibrationStatus", decision["details"])

    def test_uncalibrated_when_no_threshold_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(pathlib.Path(tmp) / "does-not-exist.json")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", missing), patch.object(
                engine_config, "SIGLIP2_MIN_SCORE", None
            ):
                decision = v2_router._decision_for_engine("siglip2_v1", 0.7, 0.4)
        self.assertEqual(decision["reason"], "UNCALIBRATED")
        self.assertEqual(decision["details"]["calibrationStatus"], "uncalibrated")

    def test_reports_real_calibrated_date_once_a_file_backed_threshold_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / "calibration.json")
            _write_calibration(path, min_score=0.5, min_margin=0.1, eval_date="2026-06-01")
            with patch.object(engine_config, "SIGLIP2_CALIBRATION_FILE", path), patch.object(
                engine_config, "SIGLIP2_MIN_SCORE", None
            ), patch.object(engine_config, "SIGLIP2_MIN_MARGIN", None):
                decision = v2_router._decision_for_engine("siglip2_v1", 0.9, 0.5)
        # This is the bug fix: it must NOT say "uncalibrated" once a real
        # calibration.json is actually gating the decision.
        self.assertEqual(decision["details"]["calibrationStatus"], "calibrated:2026-06-01")
        self.assertFalse(decision["noMatch"])


if __name__ == "__main__":
    unittest.main()
