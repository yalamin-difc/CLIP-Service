# Festival benchmark report — retrieval quality (D-5)

**Measured retrieval quality — distinct from unit-test pass counts.** A passing test suite proves the code behaves as written; the numbers below (where actually measured against a real engine) prove how well that code retrieves the correct item for a real query. Never conflate the two.

## Dataset and model

- Generated: 2026-09-30T06:43:41Z
- Labelled set: `docs/eval/retrieval_quality_labelled_set.json`
- Dataset version: `d5-retrieval-quality-v1`
- isSynthetic: `True`
- Corpus size: 12 items
- Query count: 30 (see per-language breakdown below)
- CLIP model ID: `openai/clip-vit-base-patch32`
- CLIP model revision (pinned, read-only, never overwritten by this report): `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`

## Real-engine measurement: `clip_v1`

**Status: SKIPPED — Corpus encoding failed: HTTPException: 503: {'code': 'model_unavailable', 'message': 'CLIP model dependencies are not installed.'}**

No number below this line is a real measurement for this engine. See 'Producing a real measurement' at the end of this report for how to complete this section from an environment with real model access.

## Harness self-check (arithmetic proof only — NOT a real accuracy measurement)

> harness-selfcheck uses deterministic synthetic vectors with no model inference at all. It validates this script's own recall/MRR arithmetic ONLY -- it is not, and must never be presented as, a measurement of real clip_v1 or siglip2_v1 matching quality.

| Language | Queries | Precision@1 | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|---|
| overall | 30 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 |
| en | 12 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 |
| ar | 12 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 |
| mixed | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 |

Precision/recall curve (overall, sampled thresholds):

| Threshold | Precision | Recall |
|---|---|---|
| 0.00 | 15.3% | 100.0% |
| 0.05 | 17.9% | 100.0% |
| 0.10 | 21.0% | 100.0% |
| 0.15 | 27.5% | 100.0% |
| 0.20 | 37.0% | 100.0% |
| 0.25 | 46.2% | 100.0% |
| 0.30 | 60.0% | 100.0% |
| 0.35 | 81.1% | 100.0% |
| 0.40 | 100.0% | 100.0% |
| 0.45 | 100.0% | 100.0% |
| 0.50 | 100.0% | 100.0% |
| 0.55 | 100.0% | 100.0% |
| 0.60 | 100.0% | 100.0% |
| 0.65 | 100.0% | 100.0% |
| 0.70 | 100.0% | 100.0% |
| 0.75 | 100.0% | 100.0% |
| 0.80 | 100.0% | 100.0% |
| 0.85 | 100.0% | 100.0% |
| 0.90 | 100.0% | 100.0% |
| 0.95 | 100.0% | 100.0% |
| 1.00 | 0.0% | 0.0% |

## Producing a real measurement

Real-engine sections above report `SKIPPED` in any environment without network access to the pinned model weights (this repository was authored in one — see `docs/P13_SIGLIP2_AB_IMPLEMENTATION.md`'s revision-pinning notes). From an environment with real access:

```bash
python scripts/benchmark.py --labelled-set docs/eval/retrieval_quality_labelled_set.json --output docs/eval/retrieval_quality_report.json
```

This regenerates this file with real `clip_v1` (and `siglip2_v1`, if `SIGLIP2_ENABLED=true`) precision/recall/MRR numbers in place of the SKIPPED sections above — the self-check section and this document's structure stay the same.
