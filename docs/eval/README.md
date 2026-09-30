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
   `scripts/calibrate.py` (below) is that calibration pass.

## Retrieval-quality benchmark (D-5)

`scripts/benchmark.py --labelled-set PATH` extends the original latency/
throughput benchmark (`--candidates N`, unchanged) with a retrieval-
quality harness: precision@1, recall@k (k=1, 5, 10), mean reciprocal
rank, and a precision/recall curve, **broken out by language** (`en`,
`ar`, `mixed`, plus `overall`). It reuses `evaluate_models.py`'s own
`load_manifest()`/`_rank_metrics()`/`_aggregate()` rather than
reimplementing that arithmetic, and follows the exact same
`realEngineResults` (never-fabricated, `"status": "skipped"` with a
reason when a real engine can't run) vs. `harnessSelfCheck`
(deterministic, no model inference) separation as above.

### Labelled-set format

Same shape as the retrieval manifest, with one addition: every query
must carry a `language` in `{"en", "ar", "mixed"}`.

```jsonc
{
  "datasetVersion": "...",
  "isSynthetic": true,
  "tenantId": "...",
  "siteId": "...",
  "corpus": [{"candidateId": "...", "syntheticEmbeddingSeed": 101, "title": "..."}],
  "queries": [
    {"queryId": "...", "language": "en", "text": "...", "expectedCandidateId": "..."},
    {"queryId": "...", "language": "ar", "text": "...", "expectedCandidateId": "..."},
    {"queryId": "...", "language": "mixed", "text": "...", "expectedCandidateId": "..."}
  ]
}
```

`docs/eval/retrieval_quality_labelled_set.json` is the bundled example:
12 hand-authored lost-and-found items, each with an English query, an
Arabic query, and — for half of them — a code-switched EN/AR query (30
queries total). **This is fixture data, real vocabulary and real
grammar, but not real recovered-case data** — the same honesty rule as
`synthetic_eval_manifest.json` above applies: a perfect self-check score
against it proves the harness's arithmetic is correct, never that
`clip_v1`/`siglip2_v1` will perform well on real festival reports.

### Running it

```bash
python scripts/benchmark.py \
  --labelled-set docs/eval/retrieval_quality_labelled_set.json \
  --engine clip_v1 --engine siglip2_v1 \
  --output docs/eval/retrieval_quality_report.json \
  --markdown-output docs/BENCHMARK_REPORT.md
```

Writes both the full JSON report and a filled `docs/BENCHMARK_REPORT.md`
(see `docs/BENCHMARK_REPORT_TEMPLATE.md` for the original latency-only
template this extends). The markdown report leads with **"Measured
retrieval quality — distinct from unit-test pass counts"** and never
presents a `SKIPPED` real-engine section's absence of data as a
measurement.

### Why this repo's own committed report says SKIPPED for clip_v1

Same root cause as `siglip2_v1` above: this repository was authored in a
sandbox with no outbound network access to `huggingface.co` (confirmed —
see `docs/P13_SIGLIP2_AB_IMPLEMENTATION.md`), so `openai/clip-vit-base-
patch32`'s real weights could never be downloaded here either, and
`transformers`/`torch` are not installed. The committed
`docs/BENCHMARK_REPORT.md` and `docs/eval/retrieval_quality_report.json`
are exactly what `scripts/benchmark.py` produced in this environment —
the `harnessSelfCheck` section proves the precision/recall/MRR/PR-curve
arithmetic is correct, and the `clip_v1` section honestly reports
`"status": "skipped"` with the real reason, rather than a fabricated
number. Re-running the same command from an environment with real
network access to the pinned `MODEL_REVISION` regenerates both files
with genuine measured numbers in place of the skipped section — nothing
else about the format changes. `scripts/benchmark.py` never overwrites
`app.MODEL_REVISION`; the report only ever reads and echoes it.

## Calibration (D-4)

`scripts/calibrate.py` sweeps a grid of `(min_score, min_margin)` pairs
against a **labelled** eval set — cases of `(topScore, secondScore, label)`
where `label` is whether the top-ranked candidate really was the correct
match — and writes the pair that maximises F1 (or, with
`--target-precision`, the pair with the highest recall among those at or
above that precision) to `calibration.json`. It reuses `app.py`'s own
`should_return_no_match()` to classify every case, so a calibration run
can never silently diverge from what the live `/v2/match` decision
actually does.

**This is a different eval-set shape than `synthetic_eval_manifest.json`
above.** That manifest is retrieval-shaped (one query → one expected
candidate, used for recall@K/MRR). Calibration needs an explicit
true/false label *per decision* — including genuine negatives (a
plausible-looking but wrong top candidate) — which a single-expected-
candidate retrieval manifest doesn't carry. See
`docs/eval/synthetic_calibration_set.json` for the format:

```jsonc
{
  "datasetVersion": "...",
  "isSynthetic": true,   // required, same hard refusal as evaluate_models.py
  "cases": [
    {"caseId": "...", "topScore": 0.85, "secondScore": 0.55, "label": true},
    {"caseId": "...", "topScore": 0.30, "secondScore": 0.28, "label": false}
  ]
}
```

`secondScore` may be omitted/`null` for a query with only one candidate.

### What the bundled example set does (and does not) prove

`docs/eval/synthetic_calibration_set.json` is 14 hand-authored, entirely
fabricated `(topScore, secondScore, label)` triples, deliberately
constructed to be perfectly separable. Running `scripts/calibrate.py`
against it (see `tests/test_calibrate.py`) proves the sweep, confusion-
matrix, and F1/precision/recall arithmetic are correct — **it is not, and
must never be cited as, a real SigLIP2 accuracy measurement**, for exactly
the same reason `harnessSelfCheck` above isn't one.

**D-4 depends on D-5's eval set.** A real calibration run needs real
`topScore`/`secondScore` values from real SigLIP2 inference against real,
reviewed, labelled recovery cases — which do not exist in this repository
yet. `docs/eval/retrieval_quality_labelled_set.json` (above) is fixture
data, not that reviewed real-case dataset, so it does not change this:
until a real, reviewed labelled set exists, `docs/eval/calibration.json`
does not exist in this repo, and `siglip2_v1`'s `calibrationStatus`
correctly reports `"uncalibrated"` in every environment that hasn't run a
real calibration itself.

### Running it

```bash
# Self-check only (proves the arithmetic, not real accuracy):
python scripts/calibrate.py --eval-set docs/eval/synthetic_calibration_set.json \
  --output /tmp/calibration-selfcheck.json

# A real calibration run, once a real reviewed labelled set exists:
python scripts/calibrate.py --eval-set path/to/real_labelled_set.json \
  --output docs/eval/calibration.json --target-precision 0.9
```

### How it's loaded at runtime

`embedding_engines/config.py`'s `get_siglip2_calibration_status()` /
`get_siglip2_min_score()` / `get_siglip2_min_margin()` read
`SIGLIP2_CALIBRATION_FILE` (default `docs/eval/calibration.json`,
overridable via env var) on every call — no caching, no restart required
for a newly-written file to take effect. An explicit `SIGLIP2_MIN_SCORE`/
`SIGLIP2_MIN_MARGIN` environment variable, when set, always overrides the
file (an operator's explicit choice wins). `Siglip2Engine.calibration_status`
and `/v2/match`'s per-decision `calibrationStatus` both call these same
functions, so they can never disagree with each other or with what
actually gated the decision. **CLIP's `CONF_MIN_SCORE`/`CONF_MIN_MARGIN`
and its `"calibrated"` status are completely untouched by any of this.**
