# P13 — SigLIP2 alongside CLIP: dual embeddings, model-specific indexing, A/B testing

Status: implemented on branch `claude/lucid-feynman-i3g5u9`, **SigLIP2 disabled by default**
(`SIGLIP2_ENABLED=false`), **A/B testing disabled by default** (`AB_TEST_ENABLED=false`,
`AB_UI_ENABLED=false`). Nothing in this document changes the default behaviour of the
production CLIP service.

## 1. Architecture

Before this change, `app.py` hard-coded a single model (CLIP ViT-B/32) throughout: one set of
module-level globals (`model`, `processor`, `embedding_dimension`, ...), one embedding field on
every corpus item (`embedding`/`clipEmbedding`), and one `/match` endpoint that could only ever
score CLIP vectors against CLIP vectors.

This change adds a **model-engine abstraction** alongside that code, without moving or rewriting
it:

```
embedding_engines/
  base.py            EmbeddingEngine interface, EngineProvenance/EngineReadiness value objects,
                      EngineUnavailableError (structured, never a raw traceback)
  config.py          env-var configuration for the multi-engine surface (see section 2)
  clip_engine.py      ClipEngine: a thin adapter that DELEGATES to app.py's existing,
                      byte-for-byte unmodified CLIP globals/functions (load_model,
                      image_embedding_for, text_embedding_for, runtime_provenance, ...)
  siglip2_engine.py   Siglip2Engine: an independent implementation using
                      transformers.AutoProcessor/AutoModel, its own device handling, its own
                      preprocessing fingerprint, lazy loading, torch.inference_mode()
  registry.py         ModelRegistry-equivalent: get_engine(id), list_engine_ids(),
                      describe_models() (backs GET /v2/models)
  vector_index.py     VectorIndex interface + NumpyVectorIndex (see section 11)

v2_router.py          The /v2/* API surface (FastAPI APIRouter), mounted into app.py at the very
                      end of the file via `app.include_router(...)`, after every legacy name it
                      needs already exists.

scripts/
  reembed_model.py     CLI for administrative batch re-embedding (section 12)
  benchmark_models.py  CPU benchmark harness for both engines (see section 14)
```

**Why `clip_v1` delegates instead of being rewritten:** the legacy endpoints, their Prometheus
metrics, their audit events, and 135 existing tests all depend on `app.py`'s current CLIP
globals and functions (`load_model`, `image_embedding_for`, `MODEL_NAME`, `MODEL_REVISION`,
`embedding_dimension`, `preprocessing_version`, ...). Moving that code into
`embedding_engines/clip_engine.py` and rewriting `app.py` to call it would have meant re-proving
every one of those 135 tests and every consumer's assumption about `app.py`'s module state was
still true. Instead, `ClipEngine` is a **thin, testable adapter**: every method reads or calls
through to `app.py`'s own live state via a deferred `import app` (evaluated at call time, not at
module import time, avoiding a circular import). `clip_v1`'s behaviour is therefore, by
construction, exactly whatever the legacy code already does — there is no second implementation
to drift out of sync.

**Why the import is deferred everywhere:** `app.py` needs `embedding_engines`/`v2_router` (to
mount `/v2/models`, `/v2/match`, ...), and `embedding_engines.clip_engine`/`v2_router` need
`app.py`'s functions. Rather than fight that circular dependency with import reordering,
`clip_engine.py`, `siglip2_engine.py`, and `v2_router.py` each define a local `_app()` helper
that does `import app as app_module; return app_module` **inside** each function body. By the
time any of those functions actually runs, `app.py` has finished importing (it imports
`v2_router` at the very bottom of the file), so the deferred import always sees a fully
initialized module.

## 2. Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `DEFAULT_EMBEDDING_ENGINE` | `clip_v1` | Engine used by `/v2/*` endpoints when the caller doesn't specify `engine`. Never affects legacy endpoints (always CLIP). |
| `SIGLIP2_ENABLED` | `false` | Master switch. When `false`, `siglip2_v1` reports `enabled: false` everywhere, `/v2/embeddings/*`, `/v2/match`, and `/v2/items/{id}/embeddings/siglip2_v1` return `503 engine_disabled` for it, and it is never loaded. |
| `SIGLIP2_MODEL_ID` | `google/siglip2-so400m-patch14-384` | HF model id. |
| `SIGLIP2_MODEL_REVISION` | `main` | **See "Pinning the SigLIP2 revision" below — `main` is a development-time placeholder, not a production value.** |
| `SIGLIP2_DEVICE` | `cpu` | `cpu` or `cuda`. Never silently falls back: requesting `cuda` without a CUDA-capable torch raises a structured `EngineUnavailableError`, it does not silently run on CPU. |
| `SIGLIP2_LOAD_ON_START` | `false` | If `true` *and* `SIGLIP2_ENABLED=true`, a best-effort warmup runs during startup. A warmup failure is caught and logged; it never fails process startup and never affects CLIP. |
| `SIGLIP2_EXPECTED_DIMENSION` | `1152` | Used for validating every SigLIP2 vector before it is stored or scored. |
| `SIGLIP2_TEXT_MAX_TOKENS` | `64` | Fallback fixed-length token count for SigLIP2 text padding, used only if the loaded tokenizer's own `model_max_length` isn't in a sane range. Never CLIP's 77-token constant. |
| `SIGLIP2_MIN_SCORE` / `SIGLIP2_MIN_MARGIN` | unset | Optional, evaluation-derived SigLIP2 thresholds (see section 9, Calibration). Unset by default — an unset value means "uncalibrated, don't gate," never CLIP's `CONF_MIN_SCORE`/`CONF_MIN_MARGIN`. |
| `AB_TEST_ENABLED` | `false` | Master switch for `POST /v2/ab/match`. `false` → `503 ab_testing_disabled`. |
| `AB_UI_ENABLED` | `false` | Master switch for the Festival console's "AI Model Comparison" card (see section 13). `false` → the section is not present in the rendered HTML at all. |

None of these variables are read by any legacy endpoint. `CONF_MIN_SCORE`, `CONF_MIN_MARGIN`,
`CONF_TEMPERATURE`, `MODEL_NAME`, `MODEL_REVISION`, and `EXPECTED_EMBEDDING_DIMENSION` in
`app.py` are completely untouched.

### Pinning the SigLIP2 revision

This feature was built in a network-restricted sandbox with no outbound access to
`huggingface.co` (confirmed: `curl https://huggingface.co/api/models/...` returned a 403 from
the environment's egress policy). A specific commit SHA for
`google/siglip2-so400m-patch14-384` could not be resolved from here, and inventing one would
violate the explicit requirement not to fabricate a revision hash.

`SIGLIP2_MODEL_REVISION` therefore defaults to `"main"` — a real, resolvable git ref, not a
made-up hash — used only as a development-time placeholder. **Before setting
`SIGLIP2_ENABLED=true` in any staging or production environment**, an operator with real network
access must:

```bash
# From a machine with access to huggingface.co:
curl -s https://huggingface.co/api/models/google/siglip2-so400m-patch14-384 | python3 -c \
  "import json,sys; print(json.load(sys.stdin)['sha'])"
# or:
python3 -c "from huggingface_hub import HfApi; print(HfApi().model_info('google/siglip2-so400m-patch14-384').sha)"
```

and set `SIGLIP2_MODEL_REVISION` to that commit SHA, so the model can never silently drift
underneath a running deployment. `MODEL_REVISION` for CLIP (already pinned to
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` in `app.py`) is untouched.

## 3. API endpoints

### Legacy (unchanged contracts, unchanged defaults)

`/encode-image`, `/encode-text`, `/similarity`, `/analyze-image`, `/items`,
`/items/{id}/release`, `/items/{id}/re-embed`, `/match`, `/health`, `/health/live`,
`/health/ready`, `/docs`, `/redoc`, `/metrics`, `/audit-logs`, `/` (Festival console) — all
retained verbatim. `/health`'s top-level response gained one additive key: `engines` (per-engine
readiness, see section 10); existing keys (`status`, `dependencies.model`, `dependencies.store`)
are unchanged.

### New (V2)

| Method | Path | Auth action | Notes |
|---|---|---|---|
| GET | `/v2/models` | `health:read` | Non-secret registry snapshot for every engine. |
| POST | `/v2/embeddings/image` | `match:execute` | Form fields: `file`/`image`, `engine` (default `DEFAULT_EMBEDDING_ENGINE`). Same non-vector-exposing response shape as legacy `/encode-image`, plus `engine`/`preprocessingVersion`/`calibrationStatus`. |
| POST | `/v2/embeddings/text` | `match:execute` | Form fields: `text`/`queryText`, `engine`. |
| POST | `/v2/items/{item_id}/embeddings/{engine}` | `corpus:write` | Per-item, per-engine re-embed. `engine` is a path segment (`clip_v1` or `siglip2_v1`). |
| POST | `/v2/match` | `match:execute` | Form fields mirror legacy `/match` (`text`, `file`, `topK`, `ocrText`, `barcodeValues`, `labels`, `doOcr`, `doBarcode`) plus `engine` and (permission-gated) `expectedCandidateId`. |
| POST | `/v2/ab/match` | `match:execute` | Runs `clip_v1` and `siglip2_v1` independently against the same query; `503 ab_testing_disabled` unless `AB_TEST_ENABLED=true`. |

All V2 endpoints reuse the exact same `internal_auth.require_identity` JWT contract, the same
`X-Request-Id` handling, the same `error_response`/`problem_detail` shape, and the same
tenant/site scoping as the legacy endpoints — nothing new was invented for auth or error
formatting.

`expectedCandidateId` (ground-truth evaluation, see section 8) requires the identity to also carry
the `match:evaluate` action; without it, supplying `expectedCandidateId` returns `403
action_not_permitted`. This keeps ground-truth evaluation an internal/admin capability that a
normal citizen-facing integration's permission set will not have, without adding a separate
endpoint to protect.

## 4. Storage schema

Every corpus item keeps its existing top-level fields untouched:
`embedding`, `clipEmbedding`, `embeddingDimension`, `embeddingModality`, `modelId`,
`modelRevision` — always the CLIP vector, written only by the legacy code path (`POST /items`,
`POST /items/{id}/re-embed`) exactly as before.

A new, additive `embeddings` object holds per-engine blocks:

```json
{
  "embeddings": {
    "clip_v1": {
      "vector": [ ... 512 floats ... ],
      "modelId": "openai/clip-vit-base-patch32",
      "modelRevision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
      "embeddingDimension": 512,
      "preprocessingVersion": "…",
      "embeddingModality": "image",
      "createdAt": "2026-...Z",
      "updatedAt": "2026-...Z"
    },
    "siglip2_v1": {
      "vector": [ ... 1152 floats ... ],
      "modelId": "google/siglip2-so400m-patch14-384",
      "modelRevision": "…",
      "embeddingDimension": 1152,
      "preprocessingVersion": "…",
      "embeddingModality": "image",
      "createdAt": "2026-...Z",
      "updatedAt": "2026-...Z"
    }
  }
}
```

Writing rules (`v2_router._write_embedding_block`, mirrored in `scripts/reembed_model.py`):

- `embeddings.<engine>` is written **only** through the V2 API (`POST
  /v2/items/{id}/embeddings/{engine}`) or the admin CLI — never through `POST /items`, which
  explicitly rejects a caller-supplied `embeddings` field (`400 system_fields_forbidden`), so a
  client can never inject a fake vector into a corpus item.
- For `engine == "clip_v1"` **only**, the write also refreshes the legacy `embedding` /
  `clipEmbedding` / `embeddingDimension` / `embeddingModality` fields — same vector space, so
  this keeps them current exactly as the legacy `/items/{id}/re-embed` endpoint already would.
- For every other engine (`siglip2_v1` today), the legacy fields are **never** touched.

`GET /items`, `GET /items/{id}`, and legacy `/match` responses are unchanged (they go through
the original `serialize_item`, which was not modified). V2 responses go through
`_serialize_item_v2`, which adds an `embeddings` summary with the `vector` key stripped out —
raw vectors are never returned by any endpoint.

MongoDB gets two new, additive indexes (`embeddings.clip_v1.modelId`,
`embeddings.siglip2_v1.modelId`) alongside the existing ones — `create_index` is idempotent and
requires no migration.

## 5. Validation and cross-model rejection

`v2_router._candidate_vector(candidate, engine)` is the single place that decides whether a
stored item has a usable vector for a given engine:

- It only ever reads `embeddings.<engine.engine_id>` (or, for `clip_v1` only, falls back to the
  legacy `embedding` field for items that predate the V2 API).
- It rejects (returns `None`, silently excluding the candidate from ranking — never raises)
  anything whose length doesn't match `engine.expected_dimension`, or whose norm is zero/non-finite.
- It therefore can never return a `clip_v1` candidate's vector when scoring under `siglip2_v1`,
  or vice versa — there is no code path that reads one engine's block while scoring another.

`tests/test_v2_api.py::CrossModelRejectionTests` proves this both at the unit level
(`_candidate_vector` directly) and end-to-end (`POST /v2/match?engine=siglip2_v1` excludes an
item that only has a CLIP embedding).

## 6. Model-specific index (`embedding_engines/vector_index.py`)

`NumpyVectorIndex` stores one contiguous `float32` matrix per engine (`build`/`refresh` fully
replace it; `add_or_update`/`remove` mutate it in place) and answers `search(engine, tenant_id,
permitted_site_ids, query_vector, top_k)` with brute-force cosine (a plain dot product, since
vectors are pre-normalized) restricted to rows recorded under that exact `(tenant_id, site_id)`
pair. Isolation is enforced **inside** the index, not left to the caller to filter afterward —
see `tests/test_engines.py::NumpyVectorIndexTests` for the isolation proofs.

For this first implementation, `v2_router._execute_engine_match` rebuilds the index from the
repository's current candidate set on every request (`list_matchable_items` → filter/validate
per engine → `index.build(...)` → `index.search(...)`), exactly like the legacy `/match`
endpoint's own per-request ranking loop. This keeps correctness simple (no cache-invalidation
bugs to get wrong) at the cost of rebuilding the matrix each request — acceptable at the
`CORPUS_LIMIT` scale (50,000 by default) this service already assumes for legacy `/match`.

The `VectorIndex` abstract base class is the seam for a future FAISS- or MongoDB Atlas Vector
Search-backed implementation: it only needs to implement `build`/`add_or_update`/`remove`/
`search`/`stats` with the same signatures, and nothing in `v2_router.py` would need to change.
No new mandatory external infrastructure was introduced for this prompt.

## 7. Re-embedding

- **Per item** (interactive/small scale): `POST /v2/items/{item_id}/embeddings/{engine}`.
- **Bulk / administrative** (large corpus migration): `scripts/reembed_model.py`.

```bash
# Preview only -- no writes, no image access required:
python scripts/reembed_model.py --engine siglip2_v1 --tenant acme --dry-run

# Real run, 25 items this invocation, images supplied by the operator
# (the CLIP service never stores raw image bytes -- see the script's own
# docstring):
python scripts/reembed_model.py --engine siglip2_v1 --tenant acme --site downtown \
  --batch-size 25 --manifest ./images.json

# Resume a prior run after item "item-00042":
python scripts/reembed_model.py --engine siglip2_v1 --tenant acme \
  --batch-size 25 --manifest ./images.json --resume-after item-00042
```

`needs_reembedding(item, engine_id, model_id, model_revision, expected_dimension)` is the pure,
independently unit-tested (`tests/test_v2_api.py::ProvenanceAndAuditTests
::test_stale_revision_triggers_only_matching_engine_reembed`) function deciding staleness: it
only ever inspects `engine_id`'s own block, so a stale `clip_v1` revision never triggers a
`siglip2_v1` re-embed and vice versa. `--force` re-embeds every scanned item regardless.

**Known limitation:** the CLIP service's database has never stored raw image bytes (only
encoded vectors and lightweight `image_info` metadata) — this was true before P13 and remains
true after it. Bulk re-embedding therefore requires the operator to supply images from wherever
the Backend originally stored them, via `--manifest` (an `itemId -> path` JSON map) or
`--image-dir`. Items with no available image source are reported under
`skippedNoImageSource`, never silently dropped or fabricated.

## 8. A/B testing (`POST /v2/ab/match`)

Runs `clip_v1` and `siglip2_v1` independently (never averaging or comparing their vectors) and
returns:

```json
{
  "clip": {"status": "success", "provenance": {...}, "latencyMs": 42, "candidates": [...], "top1Score": 0.83, "top1Top2Margin": 0.11, "decision": {...}},
  "siglip2": {"status": "unavailable", "errorCode": "engine_disabled", "message": "..."},
  "comparison": {"topKOverlap": 2, "sameTop1": false, "rankChanges": [...], "latencyDifferenceMs": null}
}
```

- **No winner is ever declared.** `comparison` never contains a `winner` field, and nothing in
  `v2_router.py` compares the two engines' raw cosine scores against each other to pick one —
  their scales are not interchangeable (see sections 8-9).
- **Failure isolation**: each engine's block is computed inside its own `try/except
  EngineUnavailableError` (and a generic `HTTPException` catch for malformed per-engine
  requests). A SigLIP2 failure — disabled, load failure, timeout, dimension mismatch — produces
  `{"status": "unavailable", "errorCode": "<category>"}` for that engine only; the CLIP side is
  computed and returned normally. No Python traceback is ever included in the response.
  `tests/test_v2_api.py::AbMatchTests::test_ab_match_isolates_siglip2_failure_from_clip` proves
  this.
- **Ground-truth evaluation**: pass `expectedCandidateId` (requires the
  `match:evaluate` action) to get `expectedRank`/`recallAt1`/`recallAt5`/`recallAt10`/
  `reciprocalRank` for each engine that succeeded, alongside its candidates.

## 9. Calibration

`clip_v1` keeps using CLIP's existing, production-calibrated `CONF_MIN_SCORE`/`CONF_MIN_MARGIN`/
`CONF_TEMPERATURE` — completely untouched. `siglip2_v1`'s `calibrationStatus` is always
`"uncalibrated"` until an operator sets `SIGLIP2_MIN_SCORE`/`SIGLIP2_MIN_MARGIN` from real
evaluation data; until then, `/v2/match`'s and `/v2/ab/match`'s no-match decision for
`siglip2_v1` is always `{"noMatch": false, "reason": "UNCALIBRATED", ...}` — it never gates
candidates, and it never represents a cosine score as an ownership probability (no endpoint
formats a score as "N% certain this belongs to the user").

## 10. Health and observability

`GET /health`'s response gained one additive `engines` key:

```json
{
  "status": "ok",
  "dependencies": {"model": {...}, "store": {...}},
  "engines": {
    "clip_v1": {"engine": "clip_v1", "enabled": true, "loaded": true, "ready": true, "device": "cpu", "embeddingDimension": 512, "expectedEmbeddingDimension": 512, "warmupCompleted": true, "lastLoadErrorCategory": null, ...},
    "siglip2_v1": {"engine": "siglip2_v1", "enabled": false, "loaded": false, "ready": false, ...}
  }
}
```

A disabled/unavailable SigLIP2 engine never flips `status` or `/health/ready`'s readiness —
those two checks are computed exactly as before, from CLIP + storage only.

New Prometheus metrics (all engine-labeled; `engine` only ever takes the small fixed set of
registered engine ids, so cardinality stays low):

- `clip_service_engine_requests_total{engine,modality,status}` — modality is `image`/`text`/`match`.
- `clip_service_engine_inference_latency_seconds{engine,modality}`
- `clip_service_engine_match_latency_seconds{engine}`
- `clip_service_engine_load_seconds{engine}` — currently only populated for `siglip2_v1` (see
  "Known limitations" below).

## 11. Auditing

`/v2/match` writes a `match_execution_v2` audit event (`engine`, `modelId`, `modelRevision`,
`preprocessingVersion`, `scoringVersion`, `topK`, `resultIds`, `scores`, `latencyMs`,
`decisioning`) via the same `get_repository().add_audit_log(...)` legacy `/match` already uses.
`/v2/ab/match` writes an `ab_match_execution` event (both engines' status + the comparison
block). `/v2/items/{id}/embeddings/{engine}` writes `re_embedding_v2`. None of these ever
include a raw embedding vector, an auth token, or unnecessary PII — only ids, scores, and
provenance, matching the legacy audit events' existing shape.

## 12. Deployment

### CPU (staging validation on the Azure `Standard_D8as_v5` VM)

```bash
docker build -f Dockerfile.cpu -t clip-service:p13-cpu .
docker run --rm -p 8080:8080 \
  -e STORAGE_MODE=mongodb -e MONGODB_URI=... -e INTERNAL_JWT_SECRET=... \
  -e INFERENCE_DEVICE=cpu \
  -e SIGLIP2_ENABLED=false \
  clip-service:p13-cpu
```

To validate SigLIP2 on staging (never production by default):

```bash
docker run --rm -p 8080:8080 \
  -e STORAGE_MODE=mongodb -e MONGODB_URI=... -e INTERNAL_JWT_SECRET=... \
  -e INFERENCE_DEVICE=cpu \
  -e SIGLIP2_ENABLED=true \
  -e SIGLIP2_MODEL_REVISION=<real pinned commit SHA -- see "Pinning" above> \
  -e SIGLIP2_DEVICE=cpu \
  -e SIGLIP2_LOAD_ON_START=false \
  -e AB_TEST_ENABLED=true \
  -e AB_UI_ENABLED=true \
  clip-service:p13-cpu
```

`Dockerfile.cpu` and `requirements.txt` are unchanged — `transformers`/`torch` were already
present for CLIP and are reused for SigLIP2 (both are `AutoModel`/`AutoProcessor`-based in
`transformers`, no new PyPI dependency was needed).

### GPU

`Dockerfile.gpu`/`requirements-gpu.txt` are unchanged. Setting `SIGLIP2_DEVICE=cuda` (with CUDA
actually available) runs SigLIP2 on GPU; CUDA is never assumed or forced — requesting `cuda`
without a working CUDA torch raises a structured, catchable `EngineUnavailableError`, never a
silent CPU fallback and never a crash.

### Concurrency

`Siglip2Engine` uses its own dedicated `anyio.CapacityLimiter(1)` for inference (see
`encode_image_async`/`encode_text_async`), independent of CLIP's `INFERENCE_CONCURRENCY`
limiter — so a slow SigLIP2 request can never starve CLIP's own concurrency budget, and only
one SigLIP2 inference runs at a time by default (avoids spawning multiple concurrent copies of
the large model's forward pass). `SIGLIP2_LOAD_ON_START=false` by default so a single
`uvicorn`/`gunicorn` worker never eagerly loads the ~3.5GB SigLIP2 So400M model unless asked to.

## 13. A/B testing / Festival demo instructions

1. Set `AB_TEST_ENABLED=true` (API) and, for the console, `AB_UI_ENABLED=true`.
2. `POST /v2/ab/match` with a signed internal identity (same JWT contract as every other
   protected endpoint), a `text` and/or `file`, and optional `topK`/`labels`/`ocrText`/
   `barcodeValues`.
3. The Festival console (`/`) gains an "AI Model Comparison" card when `AB_UI_ENABLED=true`.
   Consistent with the console's existing, tested security model (it **never** signs a JWT or
   calls a protected endpoint from the browser — see `tests/test_festival_console.py
   ::test_console_js_never_signs_a_jwt`), this card is explanatory for the general public: it
   states the comparison is available to "authorized demonstrators" through a server-side facade
   the Backend team would provide (exactly the same pattern the console's existing "Image
   Analysis" card already uses), rather than embedding a JWT in client-side JavaScript. It always
   shows the required label — "AI-assisted candidate retrieval — human verification required." —
   and the uncalibrated-SigLIP2 warning, and never renders a raw vector.

## 14. Benchmarking

```bash
python scripts/benchmark_models.py --iterations 10 --output docs/benchmarks/report.json
```

See `docs/benchmarks/README.md` for what could and could not be measured from this development
sandbox, and exact instructions for producing a real report on the target Azure VM.

## 15. Rollback

Disabling this feature is a pure configuration change:

```bash
export SIGLIP2_ENABLED=false
export AB_TEST_ENABLED=false
export AB_UI_ENABLED=false
# restart the staging service (e.g.):
systemctl restart clip-service   # or: docker restart <container>
```

- `siglip2_v1` immediately reports `enabled: false` from `/v2/models` and `/health`.
- `/v2/embeddings/*`, `/v2/match?engine=siglip2_v1`, and `/v2/items/{id}/embeddings/siglip2_v1`
  return `503 engine_disabled`.
- `/v2/ab/match` returns `503 ab_testing_disabled`.
- The Festival console's "AI Model Comparison" card disappears from the rendered HTML.
- Legacy CLIP endpoints (`/match`, `/items`, ...) are completely unaffected — they never read any
  of these three flags.
- **No database rollback is required.** Any `embeddings.siglip2_v1` data already written stays
  in place, dormant, and does not affect legacy `embedding`/`clipEmbedding` reads or scoring.

## 16. Known limitations

- **SigLIP2 revision is unpinned in this sandbox build.** `SIGLIP2_MODEL_REVISION` defaults to
  `"main"`. An operator must resolve and set a real commit SHA before enabling SigLIP2 anywhere
  beyond local experimentation — see section 2.
- **No real SigLIP2 (or even CLIP) inference was run in this development sandbox.** The sandbox
  this feature was built in has no outbound network access to `huggingface.co` (confirmed via a
  403 from the environment's egress policy) and does not have `torch`/`transformers` installed
  (their installation is also blocked by the same egress policy, since `requirements.txt` points
  at `download.pytorch.org`). All engine-level behaviour (loading, dimension validation, device
  handling, failure isolation, preprocessing fingerprinting) is proven with the same
  mock-the-model-not-the-code approach `tests/test_app.py` already established for CLIP
  (`FakeModel`/`FakeProcessor` standing in for `transformers.CLIPModel`/`CLIPProcessor`, and an
  equivalent stub for SigLIP2's `AutoModel`/`AutoProcessor`) — see `tests/test_engines.py` and
  `tests/test_v2_api.py`. **A real download-and-run validation of `google/siglip2-so400m-patch14-384`
  on the target Azure VM is a required staging step before enabling it for any real traffic** —
  see `docs/benchmarks/README.md`.
- **Per-engine model-load-time metric is only wired up for `siglip2_v1`.** `clip_v1`'s load path
  (`app.py`'s existing `load_model()`) was intentionally left untouched to avoid any risk to the
  production CLIP path; it does not yet emit `clip_service_engine_load_seconds{engine="clip_v1"}`.
- **The vector index is rebuilt per request**, not cached and incrementally maintained (see
  section 6) — acceptable at this service's existing `CORPUS_LIMIT` scale, but a future
  optimization (or a FAISS/Atlas-backed `VectorIndex`) would remove the rebuild cost.
- **Bulk re-embedding requires operator-supplied images** (`scripts/reembed_model.py`'s
  `--manifest`/`--image-dir`) because the service has never stored raw image bytes — this is a
  pre-existing architectural characteristic of the service, not something introduced by P13.

## 17. Security considerations

- `/v2/*` endpoints reuse the exact same signed-JWT (`internal_auth.require_identity`),
  tenant/site scoping, and rate-limiting middleware as every legacy endpoint — no new auth
  mechanism was introduced.
- `embeddings` is a system-controlled field: `POST /items` explicitly rejects a caller-supplied
  `embeddings` key (`400 system_fields_forbidden`), so a client can never inject an arbitrary
  vector that would later be scored as if a real model had produced it.
- No endpoint (`/v2/models`, `/v2/match`, `/v2/ab/match`, or item responses) ever returns a raw
  embedding vector — `_serialize_item_v2` strips the `vector` key from every engine block, and
  `describe_models()` never includes one either. `scripts/ci_static_checks.py`'s existing
  `check_no_raw_embedding_fixtures` guard (unmodified) continues to enforce this for committed
  test fixtures.
- Ground-truth evaluation (`expectedCandidateId`) requires the separate `match:evaluate` action,
  keeping it out of reach of a normal citizen-facing integration's permission set without adding
  a second endpoint to secure.
- `EngineUnavailableError` always carries a short, enumerable `error_category` string (e.g.
  `dependency_missing`, `dimension_mismatch`, `timeout`) — never a raw Python exception message
  or traceback — and every place it's caught (`v2_router.py`) turns it into a structured
  `problem_detail(...)` response.

## 18. Calibration warning (repeated for visibility)

**SigLIP2's cosine similarity is not an ownership probability, and it is not comparable to
CLIP's score.** A `0.91` SigLIP2 score does not mean "91% certain this belongs to the user," and
it cannot be compared numerically against a CLIP score of `0.91` — the two models were trained
with different objectives and different score distributions. Every SigLIP2-bearing response
carries `"calibrationStatus": "uncalibrated"` for exactly this reason. Production thresholds for
SigLIP2 must come from real evaluation data (`SIGLIP2_MIN_SCORE`/`SIGLIP2_MIN_MARGIN`), gathered
using `/v2/ab/match`'s ground-truth evaluation mode and/or an offline evaluation script run
against a labeled dataset — never copied from CLIP's `CONF_MIN_SCORE`/`CONF_MIN_MARGIN`.
