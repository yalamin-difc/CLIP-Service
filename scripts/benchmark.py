#!/usr/bin/env python3
"""Synthetic festival benchmark for the current exhaustive-scan matcher."""

from __future__ import annotations

import argparse
import json
import pathlib
import resource
import statistics
import sys
import time
from typing import Any, Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import app


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, choices=SUPPORTED_COUNTS, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--dimension", type=int, default=app.EXPECTED_EMBEDDING_DIMENSION)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run(args.candidates, max(1, args.iterations), args.dimension)
    rendered = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
