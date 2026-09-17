#!/usr/bin/env python3
"""Calibration/evaluation harness for clip_v1 and siglip2_v1.

Reads a dataset manifest of labelled synthetic query -> expected-candidate
pairs (see docs/eval/synthetic_eval_manifest.json and docs/eval/README.md)
and computes, PER ENGINE INDEPENDENTLY, retrieval quality metrics:
recall@1, recall@5, recall@10, and mean reciprocal rank (MRR).

Two distinct modes, never blended into one number:

  * "real-engine" -- runs the manifest's queries/corpus through an actual
    enabled, loadable embedding engine (clip_v1 or siglip2_v1) via the
    same registry/vector-index code path the service itself uses. If the
    engine cannot be enabled or loaded here (no SIGLIP2_ENABLED, no model
    weights reachable, missing dependencies, etc.), that engine's section
    is reported as "status": "skipped" with the reason -- never a
    fabricated or estimated number.

  * "harness-selfcheck" -- uses deterministic synthetic vectors (seeded
    per manifest entry, no model inference at all) purely to prove this
    script's own recall/MRR arithmetic is correct. This mode NEVER
    represents real matching quality for any model and must never be
    reported, logged, or displayed as if it did -- see the "warning"
    field always present on its section.

CLIP and SigLIP2 results are never averaged, compared as one score, or
used to declare a "winner" -- each engine's section stands alone, exactly
like /v2/ab/match's own evaluation semantics (see docs/P13_SIGLIP2_AB_IMPLEMENTATION.md).

Usage:
    python scripts/evaluate_models.py --manifest docs/eval/synthetic_eval_manifest.json \\
        --engine clip_v1 --engine siglip2_v1 --output docs/eval/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from embedding_engines.base import EngineUnavailableError  # noqa: E402
from embedding_engines.registry import get_engine  # noqa: E402
from embedding_engines.vector_index import NumpyVectorIndex  # noqa: E402

SELFCHECK_WARNING = (
    "harness-selfcheck uses deterministic synthetic vectors with no model "
    "inference at all. It validates this script's own recall/MRR "
    "arithmetic ONLY -- it is not, and must never be presented as, a "
    "measurement of real clip_v1 or siglip2_v1 matching quality."
)


def load_manifest(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required_top = {"corpus", "queries", "tenantId", "siteId"}
    missing = required_top - data.keys()
    if missing:
        raise ValueError(f"Manifest missing required top-level keys: {sorted(missing)}")
    if not data.get("isSynthetic", False):
        raise ValueError(
            "This harness only accepts manifests explicitly marked isSynthetic=true. "
            "It has no facility for handling real user data and must not be pointed at any."
        )
    for candidate in data["corpus"]:
        for key in ("candidateId", "syntheticEmbeddingSeed", "title"):
            if key not in candidate:
                raise ValueError(f"Corpus entry missing required key '{key}': {candidate}")
    for query in data["queries"]:
        for key in ("queryId", "text", "expectedCandidateId"):
            if key not in query:
                raise ValueError(f"Query entry missing required key '{key}': {query}")
        candidate_ids = {c["candidateId"] for c in data["corpus"]}
        if query["expectedCandidateId"] not in candidate_ids:
            raise ValueError(f"Query {query['queryId']} references unknown expectedCandidateId {query['expectedCandidateId']!r}")
    return data


def _rank_metrics(ranked_ids: List[str], expected_id: str) -> Dict[str, Any]:
    rank = ranked_ids.index(expected_id) + 1 if expected_id in ranked_ids else None
    return {
        "expectedCandidateId": expected_id,
        "rank": rank,
        "hitAt1": bool(rank is not None and rank <= 1),
        "hitAt5": bool(rank is not None and rank <= 5),
        "hitAt10": bool(rank is not None and rank <= 10),
        "reciprocalRank": (1.0 / rank) if rank else 0.0,
    }


def _aggregate(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(per_query)
    if n == 0:
        return {"queriesEvaluated": 0}
    return {
        "queriesEvaluated": n,
        "recallAt1": round(sum(1 for q in per_query if q["hitAt1"]) / n, 4),
        "recallAt5": round(sum(1 for q in per_query if q["hitAt5"]) / n, 4),
        "recallAt10": round(sum(1 for q in per_query if q["hitAt10"]) / n, 4),
        "meanReciprocalRank": round(sum(q["reciprocalRank"] for q in per_query) / n, 4),
    }


def evaluate_with_engine(engine_id: str, manifest: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
    """Runs the manifest through a real, enabled engine. Never fabricates a
    metric for an engine that could not actually load and run here."""
    engine = get_engine(engine_id)
    section: Dict[str, Any] = {"engine": engine_id, "mode": "real-engine"}

    if not engine.is_enabled():
        section["status"] = "skipped"
        section["reason"] = f"{engine_id} is disabled by configuration (is_enabled() returned False)."
        return section

    tenant_id = manifest["tenantId"]
    site_id = manifest["siteId"]

    try:
        rows = []
        for candidate in manifest["corpus"]:
            vector = engine.encode_text(candidate["title"])
            rows.append((candidate["candidateId"], tenant_id, site_id, vector))
    except EngineUnavailableError as exc:
        section["status"] = "skipped"
        section["reason"] = f"Corpus encoding failed: {exc} (errorCategory={exc.error_category})"
        return section
    except Exception as exc:  # pragma: no cover - depends on real model/network availability
        section["status"] = "skipped"
        section["reason"] = f"Corpus encoding failed: {type(exc).__name__}: {exc}"
        return section

    index = NumpyVectorIndex()
    index.build(engine_id, rows)

    provenance = engine.provenance().to_dict()
    section["modelId"] = provenance["modelId"]
    section["modelRevision"] = provenance["modelRevision"]
    section["calibrationStatus"] = provenance["calibrationStatus"]

    per_query: List[Dict[str, Any]] = []
    for query in manifest["queries"]:
        try:
            query_vector = engine.encode_text(query["text"])
        except (EngineUnavailableError, Exception) as exc:  # noqa: BLE001
            per_query.append(
                {
                    "queryId": query["queryId"],
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
        ranked = index.search(engine_id, tenant_id, [site_id], query_vector, top_k=top_k)
        ranked_ids = [candidate_id for candidate_id, _score in ranked]
        metrics = _rank_metrics(ranked_ids, query["expectedCandidateId"])
        metrics["queryId"] = query["queryId"]
        per_query.append(metrics)

    section["status"] = "success"
    section["perQuery"] = per_query
    section["aggregate"] = _aggregate(per_query)
    return section


def _synthetic_vector(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype("float64")
    return vector / (np.linalg.norm(vector) or 1.0)


def evaluate_synthetic_selfcheck(manifest: Dict[str, Any], dim: int = 32, noise: float = 0.05) -> Dict[str, Any]:
    """Proves the recall/MRR arithmetic above is correct using deterministic,
    seeded vectors -- no model inference, no real embeddings. A query's
    synthetic vector is its expected candidate's vector plus small seeded
    noise, so a correct implementation should recover near-perfect recall;
    this checks the harness's OWN CODE, not any model's matching quality."""
    tenant_id = manifest["tenantId"]
    site_id = manifest["siteId"]
    candidate_seed = {c["candidateId"]: c["syntheticEmbeddingSeed"] for c in manifest["corpus"]}

    rows = []
    for candidate in manifest["corpus"]:
        vector = _synthetic_vector(candidate["syntheticEmbeddingSeed"], dim)
        rows.append((candidate["candidateId"], tenant_id, site_id, vector.tolist()))

    index = NumpyVectorIndex()
    index.build("harness-selfcheck", rows)

    per_query: List[Dict[str, Any]] = []
    for i, query in enumerate(manifest["queries"]):
        expected_id = query["expectedCandidateId"]
        base_vector = _synthetic_vector(candidate_seed[expected_id], dim)
        noise_vector = _synthetic_vector(candidate_seed[expected_id] * 1000 + i, dim)
        query_vector = base_vector + noise * noise_vector
        query_vector = query_vector / (np.linalg.norm(query_vector) or 1.0)
        ranked = index.search("harness-selfcheck", tenant_id, [site_id], query_vector.tolist(), top_k=10)
        ranked_ids = [candidate_id for candidate_id, _score in ranked]
        metrics = _rank_metrics(ranked_ids, expected_id)
        metrics["queryId"] = query["queryId"]
        per_query.append(metrics)

    return {
        "mode": "harness-selfcheck",
        "status": "success",
        "warning": SELFCHECK_WARNING,
        "perQuery": per_query,
        "aggregate": _aggregate(per_query),
    }


def build_report(manifest_path: str, engine_ids: List[str], top_k: int, include_selfcheck: bool) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    report: Dict[str, Any] = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifestPath": manifest_path,
        "datasetVersion": manifest.get("datasetVersion"),
        "isSynthetic": manifest.get("isSynthetic"),
        "corpusSize": len(manifest["corpus"]),
        "queryCount": len(manifest["queries"]),
        "realEngineResults": {},
    }
    for engine_id in engine_ids:
        report["realEngineResults"][engine_id] = evaluate_with_engine(engine_id, manifest, top_k=top_k)
    if include_selfcheck:
        report["harnessSelfCheck"] = evaluate_synthetic_selfcheck(manifest)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibration/evaluation harness for clip_v1 and siglip2_v1.")
    parser.add_argument(
        "--manifest", default="docs/eval/synthetic_eval_manifest.json", help="Path to a dataset manifest (isSynthetic: true only)."
    )
    parser.add_argument(
        "--engine", dest="engines", action="append", choices=["clip_v1", "siglip2_v1"], help="Repeatable; default: both."
    )
    parser.add_argument("--top-k", type=int, default=10, help="Candidates retrieved per query.")
    parser.add_argument(
        "--no-selfcheck", action="store_true", help="Skip the harness-selfcheck section (real-engine results only)."
    )
    parser.add_argument("--output", default=None, help="Write the JSON report to this path in addition to stdout.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    engine_ids = args.engines or ["clip_v1", "siglip2_v1"]
    report = build_report(args.manifest, engine_ids, args.top_k, include_selfcheck=not args.no_selfcheck)
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
