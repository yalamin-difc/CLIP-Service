# Calibration / evaluation harness

`scripts/evaluate_models.py` computes retrieval-quality metrics
(recall@1, recall@5, recall@10, mean reciprocal rank) for `clip_v1` and
`siglip2_v1`, **independently, never blended into one score**, matching
`/v2/ab/match`'s own rule of never declaring a "winner" between engines
(see `docs/P13_SIGLIP2_AB_IMPLEMENTATION.md`).

## What this is (and is not) proof of

This harness ships with exactly one dataset:
`docs/eval/synthetic_eval_manifest.json` — 8 hand-authored, entirely
fabricated lost-and-found items ("red leather wallet", "navy backpack",
...) and 9 queries against them (English, Arabic, and one mixed-language
query), each with a known `expectedCandidateId`.

**This manifest is synthetic fixture data, not a real-world accuracy
benchmark.** A perfect score against it proves the harness's plumbing
works — that embeddings get produced, indexed, ranked, and scored
correctly — it does **not** demonstrate that `clip_v1` or `siglip2_v1`
will perform well against real festival lost-and-found reports. Do not
cite a `realEngineResults` number from this manifest as a production
accuracy claim. Building a real evaluation set requires real, reviewed,
labelled recovery cases — out of scope for this harness itself.

## Two separate report sections

1. **`realEngineResults`** — runs the manifest through an actual engine
   (`clip_v1` or `siglip2_v1`) via `embedding_engines.registry.get_engine()`,
   the exact same code path the live service uses. If the engine can't be
   enabled or loaded in the environment the harness runs in (no
   `SIGLIP2_ENABLED`, no reachable model weights, missing `torch`/
   `transformers`, etc.), that engine's section is `"status": "skipped"`
   with a `"reason"` — **the harness never invents a number for an engine
   it could not actually run.**

2. **`harnessSelfCheck`** — uses deterministic, seeded synthetic vectors
   with **no model inference at all**, to prove the recall/MRR arithmetic
   in this script is itself correct. It always carries an explicit
   `"warning"` field. This section must never be read, logged, or quoted
   as if it measured real model quality — it measures whether this
   Python file's own ranking math is right.

These two sections are never merged, averaged, or displayed as a single
number (enforced by `tests/test_evaluate_models.py`).

## Running it

```bash
# Both engines, full report to stdout + file
python scripts/evaluate_models.py \
  --manifest docs/eval/synthetic_eval_manifest.json \
  --engine clip_v1 --engine siglip2_v1 \
  --output docs/eval/report.json

# Real-engine results only (skip the self-check section)
python scripts/evaluate_models.py --no-selfcheck
```

Running `siglip2_v1` for real requires `SIGLIP2_ENABLED=true`, a
network-reachable `google/siglip2-so400m-patch14-384` at the configured
`SIGLIP2_MODEL_REVISION`, and `torch`/`transformers` installed — none of
which is available in this development sandbox (see
`docs/P13_SIGLIP2_AB_IMPLEMENTATION.md`'s revision-pinning notes). In
this sandbox, both engines correctly report `"status": "skipped"`.

## Manifest format

```jsonc
{
  "datasetVersion": "...",
  "isSynthetic": true,        // required -- the loader refuses anything else
  "demoData": true,
  "tenantId": "...",
  "siteId": "...",
  "corpus": [
    {"candidateId": "...", "syntheticEmbeddingSeed": 101, "title": "...", "labels": [...], "ocrText": "...", "barcodeValues": [...]}
  ],
  "queries": [
    {"queryId": "...", "modality": "text", "language": "en", "text": "...", "expectedCandidateId": "..."}
  ]
}
```

`syntheticEmbeddingSeed` is only used by the `harnessSelfCheck` section
(a real-engine run always encodes the actual `title`/`text` text through
the real model). `load_manifest()` validates required keys and rejects
any manifest not explicitly marked `isSynthetic: true` — this harness has
no facility for handling real user data and must never be pointed at any.

## Extending toward a real evaluation

To turn this into a genuine accuracy benchmark:

1. Assemble a real, reviewed dataset of confirmed recovery cases (report
   text/image → the item it actually matched), with PII already removed
   or handled per the platform's PII policy.
2. Build a manifest in the same shape, but with `isSynthetic: false` —
   which will require loosening `load_manifest()`'s current hard refusal,
   deliberately, once real-data handling (consent, retention, redaction)
   is designed for this specific script.
3. Resolve and set a real, pinned `SIGLIP2_MODEL_REVISION` from an
   environment with HuggingFace network access (see
   `docs/P13_SIGLIP2_AB_IMPLEMENTATION.md`).
4. Run `evaluate_with_engine` for both engines against that manifest, and
   feed the resulting `recallAt*`/`meanReciprocalRank` numbers into a real
   calibration pass for `SIGLIP2_MIN_SCORE`/`SIGLIP2_MIN_MARGIN`
   (`embedding_engines/config.py`) — never reusing CLIP's own thresholds.
