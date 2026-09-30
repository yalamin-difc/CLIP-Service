#!/usr/bin/env python3
"""Festival benchmark: latency/throughput (original) plus a D-5 retrieval-
quality harness (precision@1, recall@k, MRR, and a precision/recall curve,
broken out by language) against a labelled true-pair dataset.

The two modes are independent and never blended:
  --candidates N      original latency/throughput benchmark (unchanged).
  --labelled-set PATH new retrieval-quality benchmark (D-5); writes a
                       filled docs/BENCHMARK_REPORT.md alongside the JSON
                       report, and NEVER overwrites app.MODEL_REVISION or
                       any other pinned configuration -- the report only
                       ever reads and echoes it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import resource
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import app
import evaluate_models
from embedding_engines.registry import get_engine
from embedding_engines.vector_index import NumpyVectorIndex


SUPPORTED_COUNTS = (1_000, 10_000, 50_000)


def milliseconds(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def gpu_memory_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def percentile(values: List[float], value: float) -> float:
    return float(np.percentile(np.asarray(values), value))


def run(candidate_count: int, iterations: int, dimension: int) -> Dict[str, Any]:
    rng = np.random.default_rng(2026)
    candidates = rng.normal(size=(candidate_count, dimension)).astype(np.float32)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
    query = rng.normal(size=(dimension,)).astype(np.float32)
    query /= np.linalg.norm(query)

    embedding_latencies: List[float] = []
    try:
        started = time.perf_counter()
        embedded = app.image_embedding_for(Image.new("RGB", (224, 224), "white"))
        embedding_latencies.append(milliseconds(started))
        query = np.asarray(embedded, dtype=np.float32)
        if query.shape[0] != dimension:
            query = rng.normal(size=(dimension,)).astype(np.float32)
            query /= np.linalg.norm(query)
    except Exception:
        # Ranking benchmarks remain useful in environments without model weights.
        pass

    retrieval_latencies: List[float] = []
    ranking_latencies: List[float] = []
    total_latencies: List[float] = []
    errors = 0
    wall_started = time.perf_counter()
    for _ in range(iterations):
        total_started = time.perf_counter()
        try:
            retrieval_started = time.perf_counter()
            retrieved = candidates[:candidate_count]
            retrieval_latencies.append(milliseconds(retrieval_started))

            ranking_started = time.perf_counter()
            scores = retrieved @ query
            np.argpartition(scores, -min(5, candidate_count))[-min(5, candidate_count) :]
            ranking_latencies.append(milliseconds(ranking_started))
        except Exception:
            errors += 1
        total_latencies.append(milliseconds(total_started))
    wall_seconds = max(time.perf_counter() - wall_started, 1e-9)

    return {
        "candidateCount": candidate_count,
        "iterations": iterations,
        "dimension": dimension,
        "algorithm": "exhaustive in-memory cosine scan",
        "device": app.INFERENCE_DEVICE,
        "modelId": app.MODEL_NAME,
        "modelRevision": app.MODEL_REVISION,
        "embeddingLatencyMs": embedding_latencies,
        "candidateRetrievalLatencyMs": {
            "mean": statistics.fmean(retrieval_latencies),
            "p95": percentile(retrieval_latencies, 95),
        },
        "rankingLatencyMs": {
            "mean": statistics.fmean(ranking_latencies),
            "p95": percentile(ranking_latencies, 95),
        },
        "totalResponseTimeMs": {
            "mean": statistics.fmean(total_latencies),
            "p95": percentile(total_latencies, 95),
        },
        "memoryMb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
        "gpuMemoryMb": gpu_memory_mb(),
        "throughputPerSecond": iterations / wall_seconds,
        "errorRate": errors / iterations,
    }


# ---------------------------------------------------------------------------
# D-5: retrieval-quality harness (precision@1, recall@k, MRR, PR curve, by
# language). Reuses evaluate_models.py's already-tested manifest loading and
# rank/aggregate arithmetic (_rank_metrics/_aggregate) rather than
# reimplementing it -- only the language breakdown and the PR curve are new.
# ---------------------------------------------------------------------------

LANGUAGES: Tuple[str, ...] = ("en", "ar", "mixed")
DEFAULT_K_VALUES: Tuple[int, ...] = (1, 5, 10)


def default_pr_thresholds() -> List[float]:
    return [round(float(t), 4) for t in np.arange(0.0, 1.0001, 0.05)]


def load_labelled_set(path: str) -> Dict[str, Any]:
    """Loads and validates a D-5 labelled true-pair dataset. Reuses
    evaluate_models.load_manifest for the corpus/query/isSynthetic
    validation it already does correctly, then additionally requires every
    query to carry a real 'language' in LANGUAGES -- the one thing a plain
    retrieval manifest doesn't need but a per-language retrieval-quality
    report does."""
    manifest = evaluate_models.load_manifest(path)
    for query in manifest["queries"]:
        language = query.get("language")
        if language not in LANGUAGES:
            raise ValueError(
                f"Query {query.get('queryId', '?')!r} has missing/invalid 'language' "
                f"(must be one of {LANGUAGES}): {language!r}"
            )
    return manifest


def _pr_curve(pairs: List[Tuple[float, bool]], thresholds: List[float]) -> List[Dict[str, Any]]:
    """Standard single-relevant-item precision/recall curve: at each score
    threshold, a (candidate, query) pair counts as 'retrieved' if its
    cosine score is >= threshold; 'relevant' if that candidate is the
    query's one correct match. Aggregated across every query/candidate
    pair in the group, never per-query-averaged (which would be undefined
    whenever a query retrieves zero candidates at a high threshold)."""
    points = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for score, relevant in pairs:
            retrieved = score >= threshold
            if retrieved and relevant:
                tp += 1
            elif retrieved and not relevant:
                fp += 1
            elif not retrieved and relevant:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        points.append(
            {"threshold": round(float(threshold), 4), "precision": round(precision, 4), "recall": round(recall, 4), "tp": tp, "fp": fp, "fn": fn}
        )
    return points


def _by_language_report(
    manifest_queries: List[Dict[str, Any]],
    per_query_metrics: List[Dict[str, Any]],
    ranked_by_query: Dict[str, List[Tuple[str, float]]],
    expected_by_query: Dict[str, str],
    pr_thresholds: List[float],
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {"overall": manifest_queries}
    for language in LANGUAGES:
        groups[language] = [q for q in manifest_queries if q["language"] == language]

    by_language: Dict[str, Any] = {}
    for group_name, group_queries in groups.items():
        query_ids = {q["queryId"] for q in group_queries}
        group_metrics = [m for m in per_query_metrics if m["queryId"] in query_ids]
        aggregate = evaluate_models._aggregate(group_metrics)
        pairs: List[Tuple[float, bool]] = []
        for query_id in query_ids:
            expected = expected_by_query[query_id]
            for candidate_id, score in ranked_by_query.get(query_id, []):
                pairs.append((float(score), candidate_id == expected))
        by_language[group_name] = {
            "queriesEvaluated": aggregate.get("queriesEvaluated", 0),
            # precision@1 and recall@1 are numerically identical here --
            # every query has exactly one correct candidate, so "was the
            # top-1 result correct" is the same question asked either way.
            # Both are reported under their own names since D-5 asks for
            # each by name.
            "precisionAt1": aggregate.get("recallAt1"),
            "recallAt1": aggregate.get("recallAt1"),
            "recallAt5": aggregate.get("recallAt5"),
            "recallAt10": aggregate.get("recallAt10"),
            "meanReciprocalRank": aggregate.get("meanReciprocalRank"),
            "precisionRecallCurve": _pr_curve(pairs, pr_thresholds),
        }
    return by_language


def evaluate_retrieval_quality(
    engine_id: str, manifest: Dict[str, Any], pr_thresholds: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Runs the labelled set through a real, enabled engine -- the exact
    same registry/vector-index code path the live service uses. Never
    fabricates a metric for an engine that could not actually load and
    run here (mirrors evaluate_models.evaluate_with_engine's contract)."""
    engine = get_engine(engine_id)
    section: Dict[str, Any] = {"engine": engine_id, "mode": "real-engine"}
    if not engine.is_enabled():
        section["status"] = "skipped"
        section["reason"] = f"{engine_id} is disabled by configuration (is_enabled() returned False)."
        return section

    tenant_id = manifest["tenantId"]
    site_id = manifest["siteId"]
    corpus_size = len(manifest["corpus"])

    try:
        rows = []
        for candidate in manifest["corpus"]:
            vector = engine.encode_text(candidate["title"])
            rows.append((candidate["candidateId"], tenant_id, site_id, vector))
    except Exception as exc:  # noqa: BLE001 - mirrors evaluate_models' own broad, honest catch
        section["status"] = "skipped"
        section["reason"] = f"Corpus encoding failed: {type(exc).__name__}: {exc}"
        return section

    index = NumpyVectorIndex()
    index.build(engine_id, rows)

    provenance = engine.provenance().to_dict()
    section["modelId"] = provenance["modelId"]
    section["modelRevision"] = provenance["modelRevision"]
    section["calibrationStatus"] = provenance["calibrationStatus"]

    expected_by_query: Dict[str, str] = {}
    ranked_by_query: Dict[str, List[Tuple[str, float]]] = {}
    per_query_metrics: List[Dict[str, Any]] = []
    for query in manifest["queries"]:
        expected_by_query[query["queryId"]] = query["expectedCandidateId"]
        try:
            query_vector = engine.encode_text(query["text"])
        except Exception as exc:  # noqa: BLE001
            ranked_by_query[query["queryId"]] = []
            per_query_metrics.append(
                {
                    "queryId": query["queryId"],
                    "language": query["language"],
                    "expectedCandidateId": query["expectedCandidateId"],
                    "rank": None,
                    "hitAt1": False,
                    "hitAt5": False,
                    "hitAt10": False,
                    "reciprocalRank": 0.0,
                    "error": str(exc),
                }
            )
            continue
        ranked = index.search(engine_id, tenant_id, [site_id], query_vector, top_k=corpus_size)
        ranked_by_query[query["queryId"]] = ranked
        metrics = evaluate_models._rank_metrics([cid for cid, _score in ranked], query["expectedCandidateId"])
        metrics["queryId"] = query["queryId"]
        metrics["language"] = query["language"]
        per_query_metrics.append(metrics)

    section["status"] = "success"
    section["corpusSize"] = corpus_size
    section["queryCount"] = len(manifest["queries"])
    section["byLanguage"] = _by_language_report(
        manifest["queries"], per_query_metrics, ranked_by_query, expected_by_query, pr_thresholds or default_pr_thresholds()
    )
    return section


def evaluate_retrieval_quality_selfcheck(
    manifest: Dict[str, Any], pr_thresholds: Optional[List[float]] = None, dim: int = 32, noise: float = 0.05
) -> Dict[str, Any]:
    """Deterministic, seeded synthetic vectors -- no model inference at
    all. Proves this file's own PR-curve/language-grouping arithmetic is
    correct (evaluate_models._rank_metrics/_aggregate are already proven
    by tests/test_evaluate_models.py); it is NOT, and must never be
    presented as, a measurement of real clip_v1 or siglip2_v1 retrieval
    quality -- see evaluate_models.SELFCHECK_WARNING."""
    tenant_id = manifest["tenantId"]
    site_id = manifest["siteId"]
    corpus_size = len(manifest["corpus"])
    candidate_seed = {c["candidateId"]: c["syntheticEmbeddingSeed"] for c in manifest["corpus"]}

    rows = []
    for candidate in manifest["corpus"]:
        vector = evaluate_models._synthetic_vector(candidate["syntheticEmbeddingSeed"], dim)
        rows.append((candidate["candidateId"], tenant_id, site_id, vector.tolist()))
    index = NumpyVectorIndex()
    index.build("harness-selfcheck", rows)

    expected_by_query: Dict[str, str] = {}
    ranked_by_query: Dict[str, List[Tuple[str, float]]] = {}
    per_query_metrics: List[Dict[str, Any]] = []
    for i, query in enumerate(manifest["queries"]):
        expected_id = query["expectedCandidateId"]
        expected_by_query[query["queryId"]] = expected_id
        base_vector = evaluate_models._synthetic_vector(candidate_seed[expected_id], dim)
        noise_vector = evaluate_models._synthetic_vector(candidate_seed[expected_id] * 1000 + i, dim)
        query_vector = base_vector + noise * noise_vector
        query_vector = query_vector / (np.linalg.norm(query_vector) or 1.0)
        ranked = index.search("harness-selfcheck", tenant_id, [site_id], query_vector.tolist(), top_k=corpus_size)
        ranked_by_query[query["queryId"]] = ranked
        metrics = evaluate_models._rank_metrics([cid for cid, _score in ranked], expected_id)
        metrics["queryId"] = query["queryId"]
        metrics["language"] = query["language"]
        per_query_metrics.append(metrics)

    return {
        "mode": "harness-selfcheck",
        "status": "success",
        "warning": evaluate_models.SELFCHECK_WARNING,
        "corpusSize": corpus_size,
        "queryCount": len(manifest["queries"]),
        "byLanguage": _by_language_report(
            manifest["queries"], per_query_metrics, ranked_by_query, expected_by_query, pr_thresholds or default_pr_thresholds()
        ),
    }


def build_retrieval_quality_report(labelled_set_path: str, engine_ids: List[str], include_selfcheck: bool = True) -> Dict[str, Any]:
    manifest = load_labelled_set(labelled_set_path)
    report: Dict[str, Any] = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "labelledSetPath": labelled_set_path,
        "datasetVersion": manifest.get("datasetVersion"),
        "isSynthetic": manifest.get("isSynthetic"),
        "corpusSize": len(manifest["corpus"]),
        "queryCount": len(manifest["queries"]),
        # Never overwritten by this script -- read straight from app.py's
        # pinned constant, exactly once, for the report to echo. Nothing
        # in this file ever assigns to app.MODEL_REVISION.
        "clipModelId": app.MODEL_NAME,
        "clipModelRevision": app.MODEL_REVISION,
        "realEngineResults": {},
    }
    for engine_id in engine_ids:
        report["realEngineResults"][engine_id] = evaluate_retrieval_quality(engine_id, manifest)
    if include_selfcheck:
        report["harnessSelfCheck"] = evaluate_retrieval_quality_selfcheck(manifest)
    return report


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_score(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_retrieval_quality_markdown(report: Dict[str, Any]) -> str:
    """Renders a filled docs/BENCHMARK_REPORT.md-shaped document -- see
    docs/BENCHMARK_REPORT_TEMPLATE.md for the section this extends.
    Distinguishes, unmistakably, real-engine measurements from the
    harness self-check, and never claims a real measurement for an
    engine that reported "skipped"."""
    lines: List[str] = []
    lines.append("# Festival benchmark report — retrieval quality (D-5)")
    lines.append("")
    lines.append(
        "**Measured retrieval quality — distinct from unit-test pass counts.** "
        "A passing test suite proves the code behaves as written; the numbers below "
        "(where actually measured against a real engine) prove how well that code "
        "retrieves the correct item for a real query. Never conflate the two."
    )
    lines.append("")
    lines.append("## Dataset and model")
    lines.append("")
    lines.append(f"- Generated: {report['generatedAt']}")
    lines.append(f"- Labelled set: `{report['labelledSetPath']}`")
    lines.append(f"- Dataset version: `{report.get('datasetVersion')}`")
    lines.append(f"- isSynthetic: `{report.get('isSynthetic')}`")
    lines.append(f"- Corpus size: {report['corpusSize']} items")
    lines.append(f"- Query count: {report['queryCount']} (see per-language breakdown below)")
    lines.append(f"- CLIP model ID: `{report['clipModelId']}`")
    lines.append(f"- CLIP model revision (pinned, read-only, never overwritten by this report): `{report['clipModelRevision']}`")
    lines.append("")

    for engine_id, section in report.get("realEngineResults", {}).items():
        lines.append(f"## Real-engine measurement: `{engine_id}`")
        lines.append("")
        if section.get("status") != "success":
            lines.append(f"**Status: SKIPPED — {section.get('reason', 'no reason recorded')}**")
            lines.append("")
            lines.append(
                "No number below this line is a real measurement for this engine. "
                "See 'Producing a real measurement' at the end of this report for how to "
                "complete this section from an environment with real model access."
            )
            lines.append("")
            continue
        lines.append(f"**Status: SUCCESS** (model revision actually loaded: `{section.get('modelRevision')}`)")
        lines.append("")
        _append_by_language_table(lines, section["byLanguage"])

    if "harnessSelfCheck" in report:
        selfcheck = report["harnessSelfCheck"]
        lines.append("## Harness self-check (arithmetic proof only — NOT a real accuracy measurement)")
        lines.append("")
        lines.append(f"> {selfcheck['warning']}")
        lines.append("")
        _append_by_language_table(lines, selfcheck["byLanguage"])

    lines.append("## Producing a real measurement")
    lines.append("")
    lines.append(
        "Real-engine sections above report `SKIPPED` in any environment without network "
        "access to the pinned model weights (this repository was authored in one — see "
        "`docs/P13_SIGLIP2_AB_IMPLEMENTATION.md`'s revision-pinning notes). From an "
        "environment with real access:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append(f"python scripts/benchmark.py --labelled-set {report['labelledSetPath']} --output docs/eval/retrieval_quality_report.json")
    lines.append("```")
    lines.append("")
    lines.append(
        "This regenerates this file with real `clip_v1` (and `siglip2_v1`, if "
        "`SIGLIP2_ENABLED=true`) precision/recall/MRR numbers in place of the SKIPPED "
        "sections above — the self-check section and this document's structure stay the same."
    )
    lines.append("")
    return "\n".join(lines)


def _append_by_language_table(lines: List[str], by_language: Dict[str, Any]) -> None:
    lines.append("| Language | Queries | Precision@1 | Recall@1 | Recall@5 | Recall@10 | MRR |")
    lines.append("|---|---|---|---|---|---|---|")
    for group_name in ("overall", "en", "ar", "mixed"):
        group = by_language.get(group_name)
        if not group:
            continue
        lines.append(
            f"| {group_name} | {group['queriesEvaluated']} | {_fmt_pct(group['precisionAt1'])} | "
            f"{_fmt_pct(group['recallAt1'])} | {_fmt_pct(group['recallAt5'])} | {_fmt_pct(group['recallAt10'])} | "
            f"{_fmt_score(group['meanReciprocalRank'])} |"
        )
    lines.append("")
    overall_curve = (by_language.get("overall") or {}).get("precisionRecallCurve") or []
    sampled = [point for point in overall_curve if round(point["threshold"] * 20) == point["threshold"] * 20]
    if sampled:
        lines.append("Precision/recall curve (overall, sampled thresholds):")
        lines.append("")
        lines.append("| Threshold | Precision | Recall |")
        lines.append("|---|---|---|")
        for point in sampled:
            lines.append(f"| {point['threshold']:.2f} | {_fmt_pct(point['precision'])} | {_fmt_pct(point['recall'])} |")
        lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, choices=SUPPORTED_COUNTS, help="Original latency/throughput benchmark.")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--dimension", type=int, default=app.EXPECTED_EMBEDDING_DIMENSION)
    parser.add_argument("--labelled-set", dest="labelled_set", help="D-5: run the retrieval-quality harness against this labelled set instead.")
    parser.add_argument(
        "--engine", dest="engines", action="append", choices=["clip_v1", "siglip2_v1"], help="Repeatable; default: clip_v1 only."
    )
    parser.add_argument("--no-selfcheck", action="store_true", help="Skip the harness-selfcheck section (retrieval-quality mode only).")
    parser.add_argument("--markdown-output", default="docs/BENCHMARK_REPORT.md", help="Where to write the filled report (retrieval-quality mode only).")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.labelled_set:
        engine_ids = args.engines or ["clip_v1"]
        report = build_retrieval_quality_report(args.labelled_set, engine_ids, include_selfcheck=not args.no_selfcheck)
        rendered = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
        markdown = render_retrieval_quality_markdown(report)
        pathlib.Path(args.markdown_output).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.markdown_output).write_text(markdown, encoding="utf-8")
        print(rendered)
        print(f"Wrote filled report to {args.markdown_output}", file=sys.stderr)
        return

    if args.candidates is None:
        parser.error("either --candidates or --labelled-set is required")
    report = run(args.candidates, max(1, args.iterations), args.dimension)
    rendered = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
