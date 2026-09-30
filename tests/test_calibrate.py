"""D-4: scripts/calibrate.py's own correctness -- eval-set validation,
confusion-matrix/precision/recall/F1 arithmetic, grid construction, the
deterministic tie-break, the two winner-selection modes (max F1 vs.
target-precision), and the calibration.json this script writes. Uses the
exact same app.should_return_no_match the live service gates on, so these
tests also prove the script can never silently diverge from runtime
decision behaviour.
"""
import json
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import calibrate as calibrator  # noqa: E402

SYNTHETIC_SET_PATH = REPO_ROOT / "docs" / "eval" / "synthetic_calibration_set.json"


def _tiny_eval_set(**overrides):
    base = {
        "datasetVersion": "unit-test-v1",
        "isSynthetic": True,
        "cases": [
            {"caseId": "t1", "topScore": 0.9, "secondScore": 0.5, "label": True},
            {"caseId": "t2", "topScore": 0.8, "secondScore": 0.4, "label": True},
            {"caseId": "f1", "topScore": 0.2, "secondScore": 0.1, "label": False},
            {"caseId": "f2", "topScore": 0.3, "secondScore": 0.29, "label": False},
        ],
    }
    base.update(overrides)
    return base


class LoadEvalSetTests(unittest.TestCase):
    def _write(self, tmp_path, data):
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def test_loads_a_valid_synthetic_set(self):
        data = calibrator.load_eval_set(str(SYNTHETIC_SET_PATH))
        self.assertTrue(data["isSynthetic"])
        self.assertEqual(len(data["cases"]), 14)

    def test_rejects_missing_top_level_keys(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(pathlib.Path(tmp), {"isSynthetic": True})
            with self.assertRaises(ValueError) as ctx:
                calibrator.load_eval_set(path)
            self.assertIn("cases", str(ctx.exception))

    def test_rejects_non_synthetic_data(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(pathlib.Path(tmp), _tiny_eval_set(isSynthetic=False))
            with self.assertRaises(ValueError) as ctx:
                calibrator.load_eval_set(path)
            self.assertIn("isSynthetic", str(ctx.exception))

    def test_rejects_empty_cases(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(pathlib.Path(tmp), _tiny_eval_set(cases=[]))
            with self.assertRaises(ValueError):
                calibrator.load_eval_set(path)

    def test_rejects_case_missing_required_keys(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data = _tiny_eval_set(cases=[{"caseId": "bad", "topScore": 0.5}])
            path = self._write(pathlib.Path(tmp), data)
            with self.assertRaises(ValueError) as ctx:
                calibrator.load_eval_set(path)
            self.assertIn("label", str(ctx.exception))

    def test_rejects_non_boolean_label(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data = _tiny_eval_set(cases=[{"caseId": "bad", "topScore": 0.5, "label": "true"}])
            path = self._write(pathlib.Path(tmp), data)
            with self.assertRaises(ValueError) as ctx:
                calibrator.load_eval_set(path)
            self.assertIn("boolean", str(ctx.exception))


class ConfusionMatrixAndMetricsTests(unittest.TestCase):
    def test_confusion_matrix_matches_hand_computed_values(self):
        cases = _tiny_eval_set()["cases"]
        # min_score=0.5, min_margin=0.2: t1 (0.9/0.5, margin 0.4) accepted+true=TP;
        # t2 (0.8/0.4, margin 0.4) accepted+true=TP; f1 (0.2 < 0.5) rejected+false=TN;
        # f2 (0.3 < 0.5) rejected+false=TN.
        confusion = calibrator.confusion_matrix(cases, min_score=0.5, min_margin=0.2)
        self.assertEqual(confusion, {"tp": 2, "fp": 0, "fn": 0, "tn": 2})

    def test_confusion_matrix_with_a_false_positive(self):
        # A false case with a high score and wide margin is a real defect a
        # threshold can't fix by margin alone -- must show up as FP, not be
        # silently absorbed.
        cases = [{"caseId": "bad-fp", "topScore": 0.95, "secondScore": 0.3, "label": False}]
        confusion = calibrator.confusion_matrix(cases, min_score=0.5, min_margin=0.2)
        self.assertEqual(confusion, {"tp": 0, "fp": 1, "fn": 0, "tn": 0})

    def test_confusion_matrix_with_a_false_negative(self):
        cases = [{"caseId": "missed", "topScore": 0.3, "secondScore": 0.1, "label": True}]
        confusion = calibrator.confusion_matrix(cases, min_score=0.5, min_margin=0.2)
        self.assertEqual(confusion, {"tp": 0, "fp": 0, "fn": 1, "tn": 0})

    def test_no_second_score_only_gates_on_min_score(self):
        cases = [{"caseId": "solo", "topScore": 0.9, "secondScore": None, "label": True}]
        confusion = calibrator.confusion_matrix(cases, min_score=0.5, min_margin=0.2)
        self.assertEqual(confusion, {"tp": 1, "fp": 0, "fn": 0, "tn": 0})

    def test_metrics_perfect_separation_is_1_0_across_the_board(self):
        metrics = calibrator.compute_metrics({"tp": 5, "fp": 0, "fn": 0, "tn": 5})
        self.assertEqual(metrics, {"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0})

    def test_metrics_all_zero_confusion_never_divides_by_zero(self):
        metrics = calibrator.compute_metrics({"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        self.assertEqual(metrics, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0})


class BuildGridTests(unittest.TestCase):
    def test_inclusive_of_stop_value(self):
        grid = calibrator.build_grid(0.0, 0.2, 0.1)
        self.assertEqual(grid, [0.0, 0.1, 0.2])

    def test_rejects_non_positive_step(self):
        with self.assertRaises(ValueError):
            calibrator.build_grid(0.0, 1.0, 0.0)

    def test_no_floating_point_drift_in_output_values(self):
        grid = calibrator.build_grid(0.0, 1.0, 0.1)
        # A naive float accumulation would produce 0.30000000000000004 etc.
        for value in grid:
            self.assertEqual(value, round(value, 6))


class SweepTests(unittest.TestCase):
    def test_default_mode_maximises_f1_on_a_perfectly_separable_set(self):
        cases = json.loads(SYNTHETIC_SET_PATH.read_text(encoding="utf-8"))["cases"]
        score_grid = calibrator.build_grid(0.0, 1.0, 0.05)
        margin_grid = calibrator.build_grid(0.0, 0.3, 0.05)
        result = calibrator.sweep(cases, score_grid, margin_grid, target_precision=None)
        self.assertEqual(result["winner"]["metrics"]["f1"], 1.0)
        self.assertEqual(result["winner"]["confusion"], {"tp": 6, "fp": 0, "fn": 0, "tn": 8})

    def test_target_precision_mode_maximises_recall_among_qualifying_pairs(self):
        cases = json.loads(SYNTHETIC_SET_PATH.read_text(encoding="utf-8"))["cases"]
        score_grid = calibrator.build_grid(0.0, 1.0, 0.05)
        margin_grid = calibrator.build_grid(0.0, 0.3, 0.05)
        result = calibrator.sweep(cases, score_grid, margin_grid, target_precision=1.0)
        self.assertGreaterEqual(result["winner"]["metrics"]["precision"], 1.0)
        self.assertEqual(result["winner"]["metrics"]["recall"], 1.0)

    def test_unreachable_target_precision_raises_rather_than_falling_back(self):
        cases = _tiny_eval_set()["cases"]
        score_grid = calibrator.build_grid(0.0, 1.0, 0.1)
        margin_grid = calibrator.build_grid(0.0, 0.3, 0.1)
        with self.assertRaises(ValueError) as ctx:
            calibrator.sweep(cases, score_grid, margin_grid, target_precision=2.0)
        self.assertIn("2.0", str(ctx.exception))

    def test_tie_break_is_deterministic_across_repeated_runs(self):
        cases = _tiny_eval_set()["cases"]
        score_grid = calibrator.build_grid(0.0, 1.0, 0.1)
        margin_grid = calibrator.build_grid(0.0, 0.3, 0.1)
        first = calibrator.sweep(cases, score_grid, margin_grid, target_precision=None)
        second = calibrator.sweep(cases, score_grid, margin_grid, target_precision=None)
        self.assertEqual(first["winner"], second["winner"])


class CalibrateEndToEndTests(unittest.TestCase):
    def test_writes_a_calibration_file_with_the_expected_shape(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            output_path = str(pathlib.Path(tmp) / "calibration.json")
            result = calibrator.calibrate(
                str(SYNTHETIC_SET_PATH),
                output_path,
                (0.0, 1.0, 0.05),
                (0.0, 0.3, 0.05),
                target_precision=None,
            )
            written = json.loads(pathlib.Path(output_path).read_text(encoding="utf-8"))
            self.assertEqual(written, result["calibration"])
            for key in ("minScore", "minMargin", "evalDate", "evalSetPath", "datasetVersion", "isSynthetic", "caseCount", "optimizedFor", "metrics", "confusion"):
                self.assertIn(key, written)
            self.assertEqual(written["caseCount"], 14)
            self.assertTrue(written["isSynthetic"])
            self.assertRegex(written["evalDate"], r"^\d{4}-\d{2}-\d{2}$")

    def test_never_writes_a_calibration_that_fails_the_isSynthetic_guard(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            eval_set_path = pathlib.Path(tmp) / "real.json"
            eval_set_path.write_text(json.dumps(_tiny_eval_set(isSynthetic=False)), encoding="utf-8")
            output_path = str(pathlib.Path(tmp) / "calibration.json")
            with self.assertRaises(ValueError):
                calibrator.calibrate(str(eval_set_path), output_path, (0.0, 1.0, 0.1), (0.0, 0.3, 0.1), None)
            self.assertFalse(pathlib.Path(output_path).exists())


if __name__ == "__main__":
    unittest.main()
