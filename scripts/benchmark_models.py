#!/usr/bin/env python3
"""P13 section 21: CPU benchmark harness for clip_v1 and siglip2_v1.

Measures, for each enabled engine, only real, locally-obtained numbers:
cold-start (first encode after process start, including model load),
warm image-encoding latency, warm text-encoding latency (English, Arabic,
and mixed text), a warm match latency against a small synthetic corpus,
process memory (RSS) before/after load, and CPU utilization sampled
during warm runs. It never fabricates a number for an engine it could not
actually load and run in this environment -- such an engine's section of
the report is marked `"status": "skipped"` with the reason, rather than a
placeholder value.

Usage:
    python scripts/benchmark_models.py --engine clip_v1 --engine siglip2_v1 \\
        --iterations 10 --output docs/benchmarks/report.json

Requires SIGLIP2_ENABLED=true (and, in a real deployment, real model
access) to benchmark siglip2_v1 at all -- otherwise that engine's section
is reported as skipped, matching the service's own safe default.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embedding_engines.base import EngineUnavailableError  # noqa: E402
from embedding_engines.registry import get_engine  # noqa: E402

ENGLISH_TEXT = "black suitcase with a red ribbon on the handle"
ARABIC_TEXT = "حقيبة سفر سوداء عليها شريط أحمر على المقبض"
MIXED_TEXT = "Black Samsonite حقيبة with red ribbon"


def _rss_bytes() -> Optional[int]:
    try:
        import resource

        # ru_maxrss is KB on Linux, bytes on macOS.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak * 1024 if platform.system() != "Darwin" else peak
    except Exception:  # pragma: no cover - platform dependent
        return None


def _cpu_times() -> Optional[Dict[str, float]]:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {"userSeconds": usage.ru_utime, "systemSeconds": usage.ru_stime}
    except Exception:  # pragma: no cover
        return None


def _make_test_image():
    from PIL import Image

    return Image.new("RGB", (384, 384), (200, 60, 60))


def _timed(fn, *args, **kwargs) -> Dict[str, Any]:
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {"result": result, "latencyMs": round(elapsed_ms, 3)}


def benchmark_engine(engine_id: str, iterations: int) -> Dict[str, Any]:
    engine = get_engine(engine_id)
    section: Dict[str, Any] = {"engine": engine_id, "status": "success"}

    if not engine.is_enabled():
        section["status"] = "skipped"
        section["reason"] = f"{engine_id} is disabled by configuration (is_enabled() returned False)."
        return section

    rss_before = _rss_bytes()
    cpu_before = _cpu_times()

    try:
        cold_start = _timed(engine.encode_image, _make_test_image())
    except EngineUnavailableError as exc:
        section["status"] = "skipped"
        section["reason"] = f"Cold start failed: {exc} (errorCategory={exc.error_category})"
        return section
    except Exception as exc:  # pragma: no cover - depends on real model/network availability
        section["status"] = "skipped"
        section["reason"] = f"Cold start failed: {type(exc).__name__}: {exc}"
        return section

    rss_after_load = _rss_bytes()
    section["coldStartMs"] = cold_start["latencyMs"]
    section["embeddingDimension"] = len(cold_start["result"])
    section["expectedEmbeddingDimension"] = engine.expected_dimension
    section["device"] = engine.device
    provenance = engine.provenance().to_dict()
    section["modelId"] = provenance["modelId"]
    section["modelRevision"] = provenance["modelRevision"]
    section["calibrationStatus"] = provenance["calibrationStatus"]

    image_latencies_ms: List[float] = []
    for _ in range(iterations):
        image_latencies_ms.append(_timed(engine.encode_image, _make_test_image())["latencyMs"])

    text_latencies: Dict[str, List[float]] = {"english": [], "arabic": [], "mixed": []}
    for label, text in (("english", ENGLISH_TEXT), ("arabic", ARABIC_TEXT), ("mixed", MIXED_TEXT)):
        for _ in range(iterations):
            text_latencies[label].append(_timed(engine.encode_text, text)["latencyMs"])

    def _stats(values: List[float]) -> Dict[str, float]:
        sorted_values = sorted(values)
        n = len(sorted_values)
        return {
            "meanMs": round(sum(sorted_values) / n, 3),
            "minMs": round(sorted_values[0], 3),
            "maxMs": round(sorted_values[-1], 3),
            "p50Ms": round(sorted_values[n // 2], 3),
        }

    section["warmImageEncodeMs"] = _stats(image_latencies_ms)
    section["warmTextEncodeMs"] = {label: _stats(values) for label, values in text_latencies.items()}

    # A minimal "match" measurement: encode a query, score it against a small
    # synthetic in-memory corpus using the same NumpyVectorIndex the service
    # uses, so the number reflects retrieval cost, not just raw inference.
    from embedding_engines.vector_index import NumpyVectorIndex
    import numpy as np

    rng = np.random.default_rng(42)
    corpus_size = 500
    dim = section["embeddingDimension"]
    synthetic_vectors = rng.normal(size=(corpus_size, dim)).astype("float32")
    synthetic_vectors /= np.linalg.norm(synthetic_vectors, axis=1, keepdims=True)
    index = NumpyVectorIndex()
    index.build(
        engine_id,
        [(f"bench-{i}", "bench-tenant", "bench-site", synthetic_vectors[i].tolist()) for i in range(corpus_size)],
    )

    def _match_once():
        query_vector = engine.encode_text(ENGLISH_TEXT)
        return index.search(engine_id, "bench-tenant", ["bench-site"], query_vector, top_k=5)

    match_latencies_ms = [_timed(_match_once)["latencyMs"] for _ in range(iterations)]
    section["warmMatchMs"] = _stats(match_latencies_ms)
    section["syntheticCorpusSize"] = corpus_size

    rss_after_runs = _rss_bytes()
    cpu_after = _cpu_times()
    section["memory"] = {
        "rssBeforeLoadBytes": rss_before,
        "rssAfterLoadBytes": rss_after_load,
        "rssAfterRunsBytes": rss_after_runs,
    }
    if cpu_before is not None and cpu_after is not None:
        section["cpu"] = {
            "userSecondsDuringBenchmark": round(cpu_after["userSeconds"] - cpu_before["userSeconds"], 3),
            "systemSecondsDuringBenchmark": round(cpu_after["systemSeconds"] - cpu_before["systemSeconds"], 3),
        }
    return section


def build_report(engine_ids: List[str], iterations: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
        },
        "iterationsPerMeasurement": iterations,
        "engines": {},
    }
    for engine_id in engine_ids:
        report["engines"][engine_id] = benchmark_engine(engine_id, iterations)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P13: CPU benchmark harness for clip_v1 and siglip2_v1.")
    parser.add_argument(
        "--engine", dest="engines", action="append", choices=["clip_v1", "siglip2_v1"], help="Repeatable; default: both."
    )
    parser.add_argument("--iterations", type=int, default=10, help="Warm measurements per metric.")
    parser.add_argument("--output", default=None, help="Write the JSON report to this path in addition to stdout.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    engine_ids = args.engines or ["clip_v1", "siglip2_v1"]
    report = build_report(engine_ids, args.iterations)
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
