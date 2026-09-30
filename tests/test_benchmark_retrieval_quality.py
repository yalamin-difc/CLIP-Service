"""D-5: scripts/benchmark.py's retrieval-quality harness -- labelled-set
validation, the language-breakdown grouping, the precision/recall curve
arithmetic, the "never fabricate a real-engine number" contract (mirrors
evaluate_models.py's own), and the markdown report's honesty (never
presents a SKIPPED engine's absence-of-data as a real measurement, never
overwrites the pinned CLIP model revision). The original latency/
throughput benchmark (`run()`) is also spot-checked to prove this
extension didn't disturb it.
"""
import json
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import app as clip_service  # noqa: E402
import benchmark  # noqa: E402

LABELLED_SET_PATH = REPO_ROOT / "docs" / "eval" / "retrieval_quality_labelled_set.json"


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
            {"queryId": "q1", "language": "en", "text": "alpha query", "expectedCandidateId": "c1"},
            {"queryId": "q2", "language": "ar", "text": "استعلام بيتا", "expectedCandidateId": "c2"},
        ],
    }
    base.update(overrides)
    return base


class LoadLabelledSetTests(unittest.TestCase):
    def _write(self, tmp_path, data):
        path = tmp_path / "labelled.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def test_loads_the_bundled_real_dataset(self):
        manifest = benchmark.load_labelled_set(str(LABELLED_SET_PATH))
        self.assertTrue(manifest["isSynthetic"])
        self.assertEqual(len(manifest["corpus"]), 12)
        self.assertEqual(len(manifest["queries"]), 30)
        languages = {q["language"] for q in manifest["queries"]}
        self.assertEqual(languages, {"en", "ar", "mixed"})

    def test_rejects_missing_language(self):
        import tempfile

        data = _tiny_manifest()
        del data["queries"][0]["language"]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(pathlib.Path(tmp), data)
            with self.assertRaises(ValueError) as ctx:
                benchmark.load_labelled_set(path)
            self.assertIn("language", str(ctx.exception))

    def test_rejects_invalid_language(self):
        import tempfile

        data = _tiny_manifest()
        data["queries"][0]["language"] = "fr"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(pathlib.Path(tmp), data)
            with self.assertRaises(ValueError):
                benchmark.load_labelled_set(path)

    def test_still_rejects_non_synthetic_data(self):
        # Reused from evaluate_models.load_manifest -- proves the D-5
        # loader didn't accidentally loosen that guard.
        import tempfile

        data = _tiny_manifest(isSynthetic=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(pathlib.Path(tmp), data)
            with self.assertRaises(ValueError):
                benchmark.load_labelled_set(path)


class PrCurveTests(unittest.TestCase):
    def test_hand_computed_two_point_curve(self):
        # Two relevant pairs (scores 0.9, 0.4), two irrelevant (0.6, 0.1).
        pairs = [(0.9, True), (0.6, False), (0.4, True), (0.1, False)]
        curve = benchmark._pr_curve(pairs, [0.0, 0.5, 1.0])
        by_threshold = {p["threshold"]: p for p in curve}
        # threshold 0.0: everything retrieved -> tp=2, fp=2, fn=0
        self.assertEqual(by_threshold[0.0]["tp"], 2)
        self.assertEqual(by_threshold[0.0]["fp"], 2)
        self.assertEqual(by_threshold[0.0]["precision"], 0.5)
        self.assertEqual(by_threshold[0.0]["recall"], 1.0)
        # threshold 0.5: only 0.9 and 0.6 retrieved -> tp=1 (0.9), fp=1 (0.6), fn=1 (0.4)
        self.assertEqual(by_threshold[0.5]["tp"], 1)
        self.assertEqual(by_threshold[0.5]["fp"], 1)
        self.assertEqual(by_threshold[0.5]["fn"], 1)
        self.assertEqual(by_threshold[0.5]["precision"], 0.5)
        self.assertEqual(by_threshold[0.5]["recall"], 0.5)
        # threshold 1.0: nothing retrieved -> all misses
        self.assertEqual(by_threshold[1.0], {"threshold": 1.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 2})

    def test_empty_pairs_never_divides_by_zero(self):
        curve = benchmark._pr_curve([], [0.5])
        self.assertEqual(curve, [{"threshold": 0.5, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0}])


class SelfCheckTests(unittest.TestCase):
    """Deterministic synthetic vectors, no model -- proves the harness's
    own arithmetic (not real accuracy) is correct, exactly like
    evaluate_models.py's own self-check."""

    def test_perfect_recall_on_the_bundled_dataset(self):
        manifest = benchmark.load_labelled_set(str(LABELLED_SET_PATH))
        result = benchmark.evaluate_retrieval_quality_selfcheck(manifest)
        self.assertEqual(result["status"], "success")
        self.assertIn("warning", result)
        overall = result["byLanguage"]["overall"]
        self.assertEqual(overall["queriesEvaluated"], 30)
        self.assertEqual(overall["recallAt1"], 1.0)
        self.assertEqual(overall["recallAt5"], 1.0)
        self.assertEqual(overall["recallAt10"], 1.0)
        self.assertEqual(overall["meanReciprocalRank"], 1.0)
        self.assertEqual(overall["precisionAt1"], overall["recallAt1"])

    def test_per_language_breakdown_partitions_correctly(self):
        manifest = benchmark.load_labelled_set(str(LABELLED_SET_PATH))
        result = benchmark.evaluate_retrieval_quality_selfcheck(manifest)
        by_language = result["byLanguage"]
        self.assertEqual(by_language["en"]["queriesEvaluated"], 12)
        self.assertEqual(by_language["ar"]["queriesEvaluated"], 12)
        self.assertEqual(by_language["mixed"]["queriesEvaluated"], 6)
        self.assertEqual(
            by_language["en"]["queriesEvaluated"] + by_language["ar"]["queriesEvaluated"] + by_language["mixed"]["queriesEvaluated"],
            by_language["overall"]["queriesEvaluated"],
        )

    def test_precision_recall_curve_is_present_and_shaped_correctly(self):
        manifest = benchmark.load_labelled_set(str(LABELLED_SET_PATH))
        result = benchmark.evaluate_retrieval_quality_selfcheck(manifest)
        curve = result["byLanguage"]["overall"]["precisionRecallCurve"]
        self.assertGreater(len(curve), 1)
        for point in curve:
            self.assertIn("threshold", point)
            self.assertIn("precision", point)
            self.assertIn("recall", point)
            self.assertGreaterEqual(point["precision"], 0.0)
            self.assertLessEqual(point["precision"], 1.0)

    def test_never_labelled_as_a_real_measurement(self):
        manifest = benchmark.load_labelled_set(str(LABELLED_SET_PATH))
        result = benchmark.evaluate_retrieval_quality_selfcheck(manifest)
        self.assertEqual(result["mode"], "harness-selfcheck")
        self.assertIn("no model inference at all", result["warning"])
        self.assertIn("must never be presented as", result["warning"])
        self.assertNotIn("modelRevision", result)


class RealEngineNeverFabricatesTests(unittest.TestCase):
    def test_disabled_engine_is_skipped_not_fabricated(self):
        manifest = benchmark.load_labelled_set(str(LABELLED_SET_PATH))
        # siglip2_v1 is disabled by default in every test environment.
        result = benchmark.evaluate_retrieval_quality("siglip2_v1", manifest)
        self.assertEqual(result["status"], "skipped")
        self.assertIn("disabled", result["reason"])
        self.assertNotIn("byLanguage", result)

    def test_clip_v1_without_real_weights_is_skipped_not_fabricated(self):
        # This sandbox has no transformers/torch install reaching real
        # CLIP weights -- clip_v1.encode_text() must fail, and the harness
        # must report that honestly rather than inventing a score.
        manifest = benchmark.load_labelled_set(str(LABELLED_SET_PATH))
        result = benchmark.evaluate_retrieval_quality("clip_v1", manifest)
        if result["status"] == "success":
            # If a future environment DOES have real weights reachable,
            # the byLanguage section must actually be present and shaped
            # correctly -- this branch documents that expectation without
            # assuming the sandbox's current limitation is permanent.
            self.assertIn("byLanguage", result)
            self.assertIn("overall", result["byLanguage"])
        else:
            self.assertEqual(result["status"], "skipped")
            self.assertTrue(result["reason"])
            self.assertNotIn("byLanguage", result)


class BuildReportTests(unittest.TestCase):
    def test_report_never_overwrites_the_pinned_model_revision(self):
        report = benchmark.build_retrieval_quality_report(str(LABELLED_SET_PATH), ["clip_v1"], include_selfcheck=False)
        self.assertEqual(report["clipModelRevision"], clip_service.MODEL_REVISION)
        self.assertEqual(report["clipModelId"], clip_service.MODEL_NAME)
        # And the module-level constant itself is untouched by having run this.
        self.assertEqual(clip_service.MODEL_REVISION, "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")

    def test_report_records_real_dataset_size_and_date(self):
        report = benchmark.build_retrieval_quality_report(str(LABELLED_SET_PATH), ["clip_v1"], include_selfcheck=False)
        self.assertEqual(report["corpusSize"], 12)
        self.assertEqual(report["queryCount"], 30)
        self.assertRegex(report["generatedAt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_selfcheck_omitted_when_not_requested(self):
        report = benchmark.build_retrieval_quality_report(str(LABELLED_SET_PATH), ["clip_v1"], include_selfcheck=False)
        self.assertNotIn("harnessSelfCheck", report)


class RenderMarkdownTests(unittest.TestCase):
    def test_contains_the_bold_caveat(self):
        report = benchmark.build_retrieval_quality_report(str(LABELLED_SET_PATH), ["clip_v1"])
        markdown = benchmark.render_retrieval_quality_markdown(report)
        self.assertIn("**Measured retrieval quality", markdown)
        self.assertIn("distinct from unit-test pass counts", markdown)

    def test_records_dataset_size_model_revision_and_date(self):
        report = benchmark.build_retrieval_quality_report(str(LABELLED_SET_PATH), ["clip_v1"])
        markdown = benchmark.render_retrieval_quality_markdown(report)
        self.assertIn("Corpus size: 12 items", markdown)
        self.assertIn(clip_service.MODEL_REVISION, markdown)
        self.assertIn(report["generatedAt"], markdown)

    def test_skipped_engine_never_shows_fabricated_numbers(self):
        report = benchmark.build_retrieval_quality_report(str(LABELLED_SET_PATH), ["clip_v1"], include_selfcheck=False)
        markdown = benchmark.render_retrieval_quality_markdown(report)
        if report["realEngineResults"]["clip_v1"]["status"] == "skipped":
            self.assertIn("SKIPPED", markdown)
            self.assertIn("No number below this line is a real measurement", markdown)

    def test_self_check_section_clearly_labelled_as_not_real(self):
        report = benchmark.build_retrieval_quality_report(str(LABELLED_SET_PATH), ["clip_v1"])
        markdown = benchmark.render_retrieval_quality_markdown(report)
        self.assertIn("NOT a real accuracy measurement", markdown)


class OriginalLatencyBenchmarkStillWorksTests(unittest.TestCase):
    """Regression check: this extension only added new functions/imports
    -- run() itself (the original latency/throughput benchmark) must be
    completely unaffected."""

    def test_run_still_produces_the_original_shape(self):
        report = benchmark.run(candidate_count=1000, iterations=2, dimension=8)
        for key in (
            "candidateCount",
            "iterations",
            "dimension",
            "modelId",
            "modelRevision",
            "candidateRetrievalLatencyMs",
            "rankingLatencyMs",
            "totalResponseTimeMs",
            "throughputPerSecond",
            "errorRate",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["candidateCount"], 1000)
        self.assertEqual(report["modelRevision"], clip_service.MODEL_REVISION)


if __name__ == "__main__":
    unittest.main()
