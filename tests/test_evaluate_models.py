"""scripts/evaluate_models.py: the calibration/evaluation harness's own
correctness -- manifest validation, recall@K/MRR arithmetic, the
"never fabricate a metric for an unavailable engine" contract, and the
hard separation between real-engine results and the synthetic self-check
(the self-check must never be presentable as real matching quality).
"""
import json
import pathlib
import sys
import unittest
from unittest.mock import patch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import app as clip_service  # noqa: E402
import evaluate_models as harness  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "docs" / "eval" / "synthetic_eval_manifest.json"


def _tiny_manifest(**overrides):
    base = {
        "datasetVersion": "unit-test-v1",
        "isSynthetic": True,
        "tenantId": "t1",
        "siteId": "s1",
        "corpus": [
            {"candidateId": "c1", "syntheticEmbeddingSeed": 1, "title": "alpha"},
            {"candidateId": "c2", "syntheticEmbeddingSeed": 2, "title": "beta"},
        ],
        "queries": [
            {"queryId": "q1", "text": "alpha query", "expectedCandidateId": "c1"},
        ],
    }
    base.update(overrides)
    return base


class ManifestValidationTests(unittest.TestCase):
    def test_the_real_shipped_manifest_loads_and_validates(self):
        manifest = harness.load_manifest(str(MANIFEST_PATH))
        self.assertTrue(manifest["isSynthetic"])
        self.assertGreaterEqual(len(manifest["corpus"]), 1)
        self.assertGreaterEqual(len(manifest["queries"]), 1)

    def test_rejects_a_manifest_not_marked_synthetic(self):
        data = _tiny_manifest(isSynthetic=False)
        path = self._write_tmp(data)
        with self.assertRaisesRegex(ValueError, "isSynthetic"):
            harness.load_manifest(path)

    def test_rejects_a_manifest_missing_top_level_keys(self):
        data = _tiny_manifest()
        del data["siteId"]
        path = self._write_tmp(data)
        with self.assertRaisesRegex(ValueError, "siteId"):
            harness.load_manifest(path)

    def test_rejects_a_query_with_unknown_expected_candidate(self):
        data = _tiny_manifest()
        data["queries"][0]["expectedCandidateId"] = "does-not-exist"
        path = self._write_tmp(data)
        with self.assertRaisesRegex(ValueError, "unknown expectedCandidateId"):
            harness.load_manifest(path)

    def test_rejects_a_corpus_entry_missing_a_required_key(self):
        data = _tiny_manifest()
        del data["corpus"][0]["syntheticEmbeddingSeed"]
        path = self._write_tmp(data)
        with self.assertRaisesRegex(ValueError, "syntheticEmbeddingSeed"):
            harness.load_manifest(path)

    def _write_tmp(self, data) -> str:
        tmp_dir = REPO_ROOT / "tests" / "_tmp_manifests"
        tmp_dir.mkdir(exist_ok=True)
        path = tmp_dir / f"{self._testMethodName}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return str(path)


class RankMetricsTests(unittest.TestCase):
    def test_rank_1_hit_scores_all_bands_and_full_reciprocal_rank(self):
        metrics = harness._rank_metrics(["c1", "c2", "c3"], "c1")
        self.assertEqual(metrics["rank"], 1)
        self.assertTrue(metrics["hitAt1"])
        self.assertTrue(metrics["hitAt5"])
        self.assertEqual(metrics["reciprocalRank"], 1.0)

    def test_rank_3_hit_misses_at1_but_hits_at5_and_at10(self):
        metrics = harness._rank_metrics(["x", "y", "c1"], "c1")
        self.assertEqual(metrics["rank"], 3)
        self.assertFalse(metrics["hitAt1"])
        self.assertTrue(metrics["hitAt5"])
        self.assertAlmostEqual(metrics["reciprocalRank"], 1.0 / 3)

    def test_candidate_not_ranked_at_all_scores_zero(self):
        metrics = harness._rank_metrics(["x", "y"], "c1")
        self.assertIsNone(metrics["rank"])
        self.assertFalse(metrics["hitAt1"])
        self.assertFalse(metrics["hitAt10"])
        self.assertEqual(metrics["reciprocalRank"], 0.0)

    def test_aggregate_averages_recall_and_reciprocal_rank_correctly(self):
        per_query = [
            {"hitAt1": True, "hitAt5": True, "hitAt10": True, "reciprocalRank": 1.0},
            {"hitAt1": False, "hitAt5": True, "hitAt10": True, "reciprocalRank": 0.5},
            {"hitAt1": False, "hitAt5": False, "hitAt10": False, "reciprocalRank": 0.0},
        ]
        aggregate = harness._aggregate(per_query)
        self.assertEqual(aggregate["queriesEvaluated"], 3)
        self.assertAlmostEqual(aggregate["recallAt1"], 1 / 3, places=4)
        self.assertAlmostEqual(aggregate["recallAt5"], 2 / 3, places=4)
        self.assertAlmostEqual(aggregate["meanReciprocalRank"], 0.5, places=4)

    def test_aggregate_of_zero_queries_never_divides_by_zero(self):
        aggregate = harness._aggregate([])
        self.assertEqual(aggregate, {"queriesEvaluated": 0})


class RealEngineSkipTests(unittest.TestCase):
    """The harness must never report a fabricated metric for an engine it
    could not actually run -- this is the single most important contract
    in this file, since a silently-invented "0.95 recall" for an engine
    that never actually ran would be worse than no number at all."""

    def test_disabled_engine_is_reported_skipped_not_scored(self):
        manifest = _tiny_manifest()
        result = harness.evaluate_with_engine("siglip2_v1", manifest)
        self.assertEqual(result["status"], "skipped")
        self.assertIn("reason", result)
        self.assertNotIn("aggregate", result)
        self.assertNotIn("perQuery", result)

    def test_engine_with_unavailable_dependencies_is_skipped_not_scored(self):
        # clip_v1.is_enabled() is always True, so this forces the "loads
        # but can't actually run" case deterministically -- whether or not
        # torch/transformers happen to be installed in the environment
        # this test runs in (they are in CI, not in every dev sandbox) --
        # exactly the "cannot actually run" case this harness must never
        # paper over. Same load_model() patch technique used throughout
        # this suite (see tests/test_app.py) to force a real, not
        # environment-dependent, failure.
        manifest = _tiny_manifest()
        with patch.object(clip_service, "load_model", side_effect=RuntimeError("forced failure for this test")):
            result = harness.evaluate_with_engine("clip_v1", manifest)
        self.assertEqual(result["status"], "skipped")
        self.assertIn("reason", result)
        self.assertNotIn("aggregate", result)


class SyntheticSelfCheckTests(unittest.TestCase):
    def test_selfcheck_achieves_near_perfect_recall_on_its_own_construction(self):
        manifest = harness.load_manifest(str(MANIFEST_PATH))
        result = harness.evaluate_synthetic_selfcheck(manifest)
        self.assertEqual(result["mode"], "harness-selfcheck")
        self.assertEqual(result["aggregate"]["recallAt1"], 1.0)
        self.assertEqual(result["aggregate"]["meanReciprocalRank"], 1.0)

    def test_selfcheck_always_carries_its_own_non_accuracy_warning(self):
        manifest = harness.load_manifest(str(MANIFEST_PATH))
        result = harness.evaluate_synthetic_selfcheck(manifest)
        self.assertIn("warning", result)
        self.assertIn("not", result["warning"].lower())
        self.assertIn("real", result["warning"].lower())

    def test_selfcheck_is_deterministic_across_runs(self):
        manifest = harness.load_manifest(str(MANIFEST_PATH))
        first = harness.evaluate_synthetic_selfcheck(manifest)
        second = harness.evaluate_synthetic_selfcheck(manifest)
        self.assertEqual(first["aggregate"], second["aggregate"])


class ReportShapeTests(unittest.TestCase):
    def test_report_never_merges_real_engine_and_selfcheck_sections(self):
        report = harness.build_report(str(MANIFEST_PATH), ["clip_v1", "siglip2_v1"], top_k=10, include_selfcheck=True)
        self.assertIn("realEngineResults", report)
        self.assertIn("harnessSelfCheck", report)
        # The two must be distinct top-level sections -- never merged into
        # one "results" blob that could be mistaken for a single score.
        self.assertNotEqual(set(report["realEngineResults"].keys()), {"aggregate"})
        self.assertEqual(report["harnessSelfCheck"]["mode"], "harness-selfcheck")
        for engine_id, section in report["realEngineResults"].items():
            self.assertEqual(section["mode"], "real-engine")

    def test_no_selfcheck_flag_omits_the_section_entirely(self):
        report = harness.build_report(str(MANIFEST_PATH), ["clip_v1"], top_k=10, include_selfcheck=False)
        self.assertNotIn("harnessSelfCheck", report)

    def test_report_never_declares_a_winner_between_engines(self):
        report = harness.build_report(str(MANIFEST_PATH), ["clip_v1", "siglip2_v1"], top_k=10, include_selfcheck=True)
        serialized = json.dumps(report).lower()
        self.assertNotIn("winner", serialized)


if __name__ == "__main__":
    unittest.main()
