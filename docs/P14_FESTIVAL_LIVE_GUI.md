# P14 — Dubai AI Festival Live Multimodal Asset Recovery GUI

Status: implemented on branch `claude/lucid-feynman-i3g5u9`, **demo facade disabled by
default** (`DEMO_FACADE_ENABLED=false`). Nothing here changes the default behaviour of the
production CLIP service, `/v2/*`, or any existing protected endpoint. See section 14 for the
P14.1 rebrand ("Multimodal AL Engine") and the intended new public hostname
(`ai-engine.al-amentech.io`).

## 1. Architecture

P14 adds a third layer on top of the P13 model-engine work, without touching it:

```
demo_config.py    Standalone env-var config for the public demo facade (DEMO_*), fails
                   closed via demo_config_ready() when any required value is missing.

demo_router.py    The public-facing facade: GET /demo/status, POST /demo/match,
                   GET /demo/assets/candidates/{filename}. Mounted into app.py exactly like
                   v2_router.py (always registered, gated at call time), via
                   app.include_router(demo_router).

templates/clip-console.html   Redesigned Festival landing page: new "Live AI Demo" card,
static/clip-console.css       two compact model-status cards, updated pipeline/governance
static/clip-console.js        panels, full EN/AR translations -- all pre-existing
                               P13-era sections (provenance card, AB_UI_ENABLED explanatory
                               card, service status card) are kept byte-identical so every
                               existing Festival console test keeps passing unmodified.
```

### Why the facade is a separate module that forwards, not a reimplementation

`demo_router.py` never computes a similarity score, never touches MongoDB, and never loads a
model directly. Every real answer comes from an actual HTTP call (`async httpx`, loopback
only) into this same process's own, already-hardened `/v2/match` / `/v2/ab/match` endpoints,
authenticated with a short-lived (<=120s) internal identity minted server-side for that one
call and discarded immediately after. This means:

- Every existing protection on `/v2/match`/`/v2/ab/match` (tenant/site isolation, model/
  dimension validation, audit logging, calibration status, failure isolation) applies to demo
  traffic automatically -- there is no second, parallel matching implementation to keep in
  sync or that could silently diverge in behaviour.
- The demo facade's own code is small and easy to review: identity minting, request
  forwarding, response re-shaping into a smaller public schema, rate limiting, and safe asset
  serving. Nothing else.

### Why every `/demo/*` route is always registered, gated at call time

Like `SIGLIP2_ENABLED` in `embedding_engines/siglip2_engine.py` and `AB_TEST_ENABLED` in
`v2_router.py`, `demo_router.py`'s routes are registered unconditionally in `app.py` (`from
demo_router import router as demo_router; app.include_router(demo_router)`), and every single
handler starts with `demo_config.demo_config_ready()` (aliased in this file as
`_require_demo_ready()`), raising `404` if it's false. This is functionally identical to "only
mounted when `DEMO_FACADE_ENABLED=true`" from the outside (every `/demo/*` path 404s exactly
as if it didn't exist), while staying trivially testable with `unittest.mock.patch.object`
instead of needing a process restart with different environment variables per test -- the same
pattern already used and tested throughout the P13 work.

## 2. Security boundary

**Nothing changed about how the browser authenticates** -- it still never does, and never can:

- The browser never supplies `tenantId`, `siteId`, `datasetVersion`, `demoData`, `actions`, a
  JWT, or an engine's underlying model ID. `POST /demo/match`'s only accepted fields are
  `file`, `text`, and `mode` (`clip`/`siglip2`/`compare`) -- FastAPI drops any other form field
  silently, so a client attempting to smuggle e.g. `tenantId=attacker-tenant` has no effect
  (`tests/test_demo_facade.py::ServerControlledIdentityTests
  ::test_4_client_cannot_select_tenant_site_dataset_or_engine`).
- `demo_router._mint_demo_identity()` is the only place a JWT for demo traffic is ever created.
  It calls the existing `internal_auth.sign_internal_token()` helper (unmodified) with
  hardcoded claims: `service=festival-demo-facade`, the configured `DEMO_TENANT_ID`/
  `DEMO_SITE_ID`/`DEMO_DATASET_VERSION`, `demoData=true`, `actions=["match:execute"]` only, and
  `exp` no more than 120 seconds out. The token is used for exactly one internal HTTP call and
  is never written to a response, a log line, or anywhere the browser can read it.
- `static/clip-console.js` never builds a bearer credential header and never signs anything --
  it only ever calls the two public `/demo/*` endpoints. `tests/test_festival_console.py
  ::test_console_js_never_signs_a_jwt` and `tests/test_festival_ab_ui.py
  ::test_console_js_still_never_signs_a_jwt_or_calls_v2` (both pre-existing, unmodified) keep
  passing against the redesigned console.
- Every existing protected endpoint (`/items`, `/match`, `/v2/models`, `/v2/match`,
  `/v2/ab/match`, `/v2/items/*`, `/metrics`) is untouched: same `internal_auth.require_identity`
  contract, same tenant/site scoping, same 401/403 behaviour. `tests/test_festival_console.py`'s
  `test_items_still_protected`/`test_match_still_protected`/`test_analyze_image_still_protected`
  keep passing unmodified.
- No raw embedding vector is ever included in a `/demo/*` response -- `_safe_candidate()` only
  ever copies `candidateId`, `rank`, `title`, `description`, `score`, `imageUrl`, and a short
  `explanation` string out of the (already vector-free, per P13) `/v2/match`/`/v2/ab/match`
  response.

## 3. The demo facade

### `GET /demo/status`

Public, unauthenticated, safe-only:

```json
{
  "enabled": true,
  "clip": {"ready": true, "modelId": "openai/clip-vit-base-patch32", "embeddingDimension": 512},
  "siglip2": {"ready": true, "modelId": "google/siglip2-so400m-patch14-384", "embeddingDimension": 1152, "calibrationStatus": "uncalibrated"},
  "datasetVersion": "daf-2026-v1",
  "mode": "demo"
}
```

Reads live readiness/provenance directly from the P13 model registry
(`embedding_engines.registry.get_engine(...)`) -- never hardcoded, never a placeholder.

### `POST /demo/match`

`multipart/form-data`: `file` (optional image), `text` (optional string), `mode`
(`clip`/`siglip2`/`compare`, default `clip`). Requires at least one of `file`/`text` (`400
missing_query` otherwise). The facade itself supplies `topK=DEMO_TOP_K` -- the client cannot
raise it, and there is no `expectedCandidateId` field in this public form at all (ground-truth
evaluation stays admin-only, exactly as P13 left it).

For `clip`/`siglip2`, forwards to `POST /v2/match?engine=<clip_v1|siglip2_v1>` and returns:

```json
{"requestId": "...", "mode": "clip", "status": "success", "latencyMs": 42,
 "model": {"modelId": "...", "embeddingDimension": 512}, "candidates": [...]}
```

For `compare`, forwards to `POST /v2/ab/match` (itself `503 ab_testing_disabled` unless the
operator has also set `AB_TEST_ENABLED=true`) and returns the shape specified in the P14 brief:
`clip`/`siglip2` sides each with `status`/`latencyMs`/`model`/`candidates` (`siglip2` also
carries `calibrationStatus`), plus a `comparison` block (`sameTop1`, `topKOverlap`,
`rankChanges`, `latencyDifferenceMs`) copied straight from `/v2/ab/match`'s own comparison,
which never declares a winner and never compares the two engines' raw cosine scores against
each other (their score spaces are independent -- see the P13 doc's calibration warning,
repeated in the console's UI copy).

Every candidate is reduced to `candidateId`, `rank`, `title`, `description`, `score`,
`imageUrl`, `explanation` -- nothing else, no vectors.

`Cache-Control: no-store` and the request's `X-Request-Id` are set on every response.

### `GET /demo/assets/candidates/{filename}`

Serves a single, safety-checked image file from `DEMO_ASSET_DIR/candidates/`:

1. Rejects anything containing `/`, `\`, or equal to `.`/`..` outright.
2. Requires `Path(filename).name == filename` (no residual path components after
   normalization).
3. Requires a `.jpg`/`.jpeg`/`.png`/`.webp` extension.
4. Resolves the final path and requires it to still be inside `DEMO_ASSET_DIR/candidates`
   (`Path.relative_to`, which raises on anything that escaped via a symlink or a clever
   filename) -- otherwise `404`.
5. `404` (never a stack trace) for anything missing or rejected; no directory listing is ever
   possible since the route only ever serves one named file.

`imageUrl` in a candidate is built from `item.candidateFilename` (falling back to
`item.metadata.candidateFilename`), itself re-validated by `_candidate_image_url()` the same
way (basename only, allowed extension) before being turned into a URL -- a malicious or
malformed `candidateFilename` in Mongo can never produce a path-escaping URL.

## 4. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DEMO_FACADE_ENABLED` | `false` | Master switch. `false` -> every `/demo/*` route 404s. |
| `DEMO_TENANT_ID` | *(empty)* | Server-controlled tenant for every demo request. Required. |
| `DEMO_SITE_ID` | *(empty)* | Server-controlled site. Required. |
| `DEMO_DATASET_VERSION` | *(empty)* | Server-controlled dataset version. Required. |
| `DEMO_ASSET_DIR` | *(empty)* | Directory containing `candidates/` (and `queries/`, unused by this endpoint) with demo images. Required. Mounted read-only in the container. |
| `DEMO_RATE_LIMIT_PER_MINUTE` | `12` | Per-IP limit on `POST /demo/match` (uses `request.client.host`; `X-Forwarded-For` is never trusted, matching `app.py`'s own existing rate limiter). |
| `DEMO_TOP_K` | `3` | Candidates requested per engine; not client-overridable. |

`demo_config.demo_config_ready()` is `True` only when `DEMO_FACADE_ENABLED` **and** all four
other required values are non-empty -- a half-configured facade (flag on, one value missing)
behaves exactly like a disabled one, never falling back to a blank tenant/site.

Not an operator-facing variable (internal only): the facade always calls back into its own
process over `http://127.0.0.1:{PORT}` (or `DEMO_INTERNAL_BASE_URL` if explicitly set), and its
minted JWTs are hardcoded to a 120-second TTL, per the spec.

## 5. Rate limiting and resource protection

- A dedicated per-IP limiter (`demo_router._enforce_demo_rate_limit`, using the same
  windowed-counter pattern as `app.py`'s own global limiter) caps `POST /demo/match` at
  `DEMO_RATE_LIMIT_PER_MINUTE` (default 12) requests per client IP per minute, returning `429
  demo_rate_limit_exceeded`. This runs *in addition to* `app.py`'s existing global
  `RATE_LIMIT_PER_MINUTE` (120/minute) middleware, which already covers every route including
  `/demo/*`.
- `X-Forwarded-For` is never trusted (no reverse-proxy trust configuration exists in this
  service); the client key is always `request.client.host`.
- Upload size and accepted MIME types reuse `app.py`'s existing `read_image_upload()` verbatim
  (`MAX_UPLOAD_BYTES`, `MAX_IMAGE_PIXELS`, `APPROVED_IMAGE_MIME_TYPES`) -- no separate demo-only
  limit was introduced or could drift out of sync.
- `DEMO_TOP_K` bounds candidates per engine; the client cannot override it.
- SigLIP2 inference concurrency reuses the existing per-engine `anyio.CapacityLimiter(1)` inside
  `Siglip2Engine` (P13) -- the demo facade adds no separate inference path or extra concurrency.
- Every error response is a `problem_detail(...)` JSON object with a safe code and message;
  no Python traceback is ever returned to the browser.

## 6. Auditing

Every `POST /demo/match` call writes one `festival_demo_match` audit event via the existing
`get_repository().add_audit_log(...)` (same store as every other audit event in this service):

```json
{
  "mode": "clip", "tenantId": "festival-demo", "siteId": "dubai-ai-festival",
  "datasetVersion": "daf-2026-v1", "hasImage": false, "hasText": true,
  "clipStatus": "success", "siglip2Status": null, "latency": 42,
  "resultCandidateIds": ["item-1", "item-2"]
}
```

Never includes: the raw uploaded image, an embedding vector, a JWT, `INTERNAL_JWT_SECRET`, or
a MongoDB URI.

## 7. Landing page redesign

New, prominent sections (all in `templates/clip-console.html`):

- Header tagline "Urban Intelligence AI Lab" under the AL-AMEN TECHNOLOGY wordmark.
- Hero: "Multimodal Asset Recovery AI" / subheading / "CLIP ViT-B/32 | SigLIP2 So400M" badge
  (kept `dir="ltr"` even inside the Arabic layout, since model IDs stay Latin).
- **Live AI Demo** card (`#live-demo-card`): CLIP/SigLIP2/Compare mode tabs, drag-and-drop
  image upload with immediate in-browser preview, an optional English/Arabic/mixed-text
  description field, an "Analyze Item" button (disabled until an image or text is present), an
  animated (no fake percentage) processing indicator, structured error handling
  (timeout/429/503/invalid-file/network -> distinct EN/AR messages), and results that stay
  visible (dimmed via `.is-stale`) while a new request is in flight rather than disappearing.
- Two compact model-status cards (`#model-status-cards`), populated from `GET /demo/status`
  (falls back to a static "Unavailable" state if the facade is disabled or unreachable).
- Updated AI Pipeline list: `Image / Text -> Preprocessing -> CLIP / SigLIP2 -> Model-specific
  embeddings -> Tenant/site isolated retrieval -> OCR + Barcode -> Ranked candidates -> Urban
  Intelligence fusion -> Human verification`, plus an explicit footnote that CLIP's 512-D and
  SigLIP2's 1152-D embeddings are never stored or searched in the same index.
- Governance checklist gained two new, required items: "No PII required for model
  demonstration" and "Production data inaccessible from demo facade."
- Permanent notice "AI-assisted candidate retrieval — human verification required." and a
  calibration tooltip stating SigLIP2 scores are uncalibrated retrieval signals, not ownership
  probabilities, and are not comparable across engines.

**Every pre-existing, test-covered element is unchanged**: the literal strings "CLIP AI
Engine" (now only in the `<title>` tag and the de-emphasized header product line -- the hero
headline itself is new), "Dubai AI Festival 2026", "DEMO DATA"/"بيانات تجريبية", the Arabic
`STRINGS.ar.title` value, the live `{{MODEL_ID}}`/`{{MODEL_REVISION}}`/`{{SCORING_VERSION}}`
provenance card, the CSP scoping, and the disabled-by-default `AB_UI_ENABLED` explanatory card
from P13 (kept byte-identical; the new *interactive* Compare mode in the Live AI Demo card is
additive, gated by `DEMO_FACADE_ENABLED` instead, and does not replace it). The old, always-
disabled "Image Analysis" placeholder card (which could never call a protected endpoint) was
removed, since the new Live AI Demo card supersedes it with a real, safe, working equivalent;
no test referenced its markup.

All new strings have full English and Arabic translations (`static/clip-console.js`'s
`STRINGS.en`/`STRINGS.ar`), toggled by the existing `data-i18n`/`applyLanguage()` mechanism
(extended with a `data-i18n-placeholder` variant for the textarea's placeholder text). RTL
layout (`dir="rtl"` on `#doc-root`) is unaffected and continues to work exactly as before.

## 8. Deployment

### Staging (the current Azure canary)

```bash
docker build -f Dockerfile.cpu -t clip-service:p14-demo .
docker run -d --name clip-service-canary -p 8080:8080 \
  -e STORAGE_MODE=mongodb -e MONGODB_URI=$MONGODB_URI -e INTERNAL_JWT_SECRET=$INTERNAL_JWT_SECRET \
  -e INFERENCE_DEVICE=cpu \
  -e SIGLIP2_ENABLED=true -e SIGLIP2_MODEL_REVISION=e8e487298228002f3d8a82e0cd5c8ea9c567f57f \
  -e SIGLIP2_DEVICE=cpu -e SIGLIP2_LOAD_ON_START=false \
  -e AB_TEST_ENABLED=true \
  -e DEMO_FACADE_ENABLED=true \
  -e DEMO_TENANT_ID=festival-demo \
  -e DEMO_SITE_ID=dubai-ai-festival \
  -e DEMO_DATASET_VERSION=daf-2026-v1 \
  -e DEMO_ASSET_DIR=/opt/demo-data \
  -e DEMO_RATE_LIMIT_PER_MINUTE=12 \
  -e DEMO_TOP_K=3 \
  -v /opt/demo-data:/opt/demo-data:ro \
  clip-service:p14-demo
```

`DEMO_ASSET_DIR` must be mounted **read-only** and contain `candidates/` (with the 10 released
demo items' image files, named to match each item's `candidateFilename` metadata) and
`queries/` (unused by this endpoint; reserved for future sample-query features). nginx
configuration is unchanged by this work.

### Enabling SigLIP2 startup warmup (`SIGLIP2_LOAD_ON_START=true`)

The canary command above deliberately keeps `SIGLIP2_LOAD_ON_START=false`, its default: SigLIP2
then lazy-loads on the *first* real request to it, which pays the model's one-time load cost as
a cold-start hit on whichever visitor happens to send that first request. `Siglip2Engine.warmup()`
exists specifically to move that cost to process startup instead, but nothing called it until
this change wired it into `initialize_runtime()` (`app.py`'s `@app.on_event("startup")` handler,
via the new `warmup_optional_engines()`) -- so turning the flag on is now enough:

```bash
docker run -d --name clip-service-canary -p 8080:8080 \
  -e STORAGE_MODE=mongodb -e MONGODB_URI=$MONGODB_URI -e INTERNAL_JWT_SECRET=$INTERNAL_JWT_SECRET \
  -e INFERENCE_DEVICE=cpu \
  -e SIGLIP2_ENABLED=true -e SIGLIP2_MODEL_REVISION=e8e487298228002f3d8a82e0cd5c8ea9c567f57f \
  -e SIGLIP2_DEVICE=cpu -e SIGLIP2_LOAD_ON_START=true \
  -e AB_TEST_ENABLED=true \
  -e DEMO_FACADE_ENABLED=true \
  -e DEMO_TENANT_ID=festival-demo \
  -e DEMO_SITE_ID=dubai-ai-festival \
  -e DEMO_DATASET_VERSION=daf-2026-v1 \
  -e DEMO_ASSET_DIR=/opt/demo-data \
  -e DEMO_RATE_LIMIT_PER_MINUTE=12 \
  -e DEMO_TOP_K=3 \
  -v /opt/demo-data:/opt/demo-data:ro \
  clip-service:p14-demo
```

**What changes, and what does not:**

- `DEFAULT_EMBEDDING_ENGINE` stays `clip_v1` and CLIP's own required warmup (`warmup_model()`)
  runs first, exactly as before -- this flag only adds a second, independent, best-effort
  warmup step for `siglip2_v1` immediately after it. If that step fails for any reason (model
  download error, dependency problem, a bad `SIGLIP2_MODEL_REVISION`), it is caught and logged;
  it can never fail container startup and can never affect CLIP's readiness. `siglip2_v1` then
  simply falls back to lazy-loading on first use, i.e. exactly today's `SIGLIP2_LOAD_ON_START
  =false` behavior for that one engine, while CLIP keeps working normally throughout.
- Startup now blocks until *both* warmups finish (CLIP's, then SigLIP2's) before the process
  starts accepting connections -- this is intentional: the goal is for SigLIP2 to already be
  ready before any visitor traffic arrives, not merely to start faster and warm up in the
  background while early requests risk a cold-load 504.

**Startup grace time -- do not assume a fixed number.** SigLIP2 So400M is a large model
(~3.5GB of weights); loading it (first-ever pull from Hugging Face, or from a warm local cache
on a redeployed host) plus one real inference pass is a one-time cost noticeably larger than
the steady-state ~2-3 second per-query CPU inference time quoted in section 12 below, and it
was not measured from this sandbox (no network access to actually download the model here --
see `docs/AZURE_CANARY_VERIFICATION.md`). Do not hardcode a grace period from this document.
Instead, on the very first canary start with this flag on:

1. Increase the platform's container/process startup timeout (Azure's health-check initial
   delay, or the orchestrator's equivalent) generously beyond CLIP's already-known warmup time,
   since it must now also cover SigLIP2's one-time load.
2. Read the actual duration from the structured log line `Siglip2Engine.warmup()` emits on
   completion -- `{"event": "engine_warmup", "engine": "siglip2_v1", "outcome": "success",
   "durationSeconds": ...}` (or `"outcome": "failed"` with an `errorType`, never a stack trace
   or secret, if it didn't complete) -- and use that *observed* number, with margin, to size the
   grace period for subsequent restarts of that same host/image.
3. **Verify before routing demo traffic**, per `docs/AZURE_CANARY_VERIFICATION.md` Check 4: a
   real `POST /v2/embeddings/image` and `POST /v2/embeddings/text` against the just-started
   canary (a real file/text, a valid internal JWT, `engine=siglip2_v1`) must both return HTTP
   200 with a populated `embeddingDimension`/`preprocessingVersion` -- not a `503
   engine_unavailable`/`engine_disabled` -- before pointing festival traffic (or `AB_TEST_ENABLED`
   comparisons) at this instance. `GET /v2/models` (or `/health`'s `engines.siglip2_v1` block)
   showing `"ready": true` is the same signal, cheaper to poll in a loop while waiting.
4. Never widen `SIGLIP2_INFERENCE_TIMEOUT_MS`, `INFERENCE_TIMEOUT_MS`, or any other per-request
   timeout to paper over a slow or failed startup warmup -- those govern individual request
   latency, not process startup, and loosening them would only let a genuinely broken engine
   hang requests longer instead of failing fast. If warmup keeps failing, fix the underlying
   cause (model revision, network egress, disk space for the weights) instead.

### Rollback

```bash
docker run -d --name clip-service-canary -p 8080:8080 \
  -e STORAGE_MODE=mongodb -e MONGODB_URI=$MONGODB_URI -e INTERNAL_JWT_SECRET=$INTERNAL_JWT_SECRET \
  -e INFERENCE_DEVICE=cpu -e SIGLIP2_ENABLED=false -e AB_TEST_ENABLED=false \
  -e DEMO_FACADE_ENABLED=false \
  clip-service:p14-demo
```

Setting `DEMO_FACADE_ENABLED=false` (or simply omitting the `DEMO_*` variables) makes every
`/demo/*` route 404 immediately -- no database change, no image volume unmount, and no nginx
change is required. The redesigned console still renders (its Live AI Demo card shows the
"not enabled" notice, exactly like the pre-existing P13 AB card does when `AB_UI_ENABLED` is
off), and every legacy/`/v2` endpoint is completely unaffected either way.

## 9. Test procedure

```bash
pip install -r requirements.txt   # torch/transformers not required for the test suite --
                                   # every model call is mocked, same as the existing suite
python -m pytest tests/ -v
python scripts/ci_static_checks.py
```

`tests/test_demo_facade.py` covers all 22 P14 scenarios (disabled-by-default, fail-closed
config, tenant/site/dataset/engine cannot be client-selected, `demoData=true` + configured
`datasetVersion` in the minted identity, no JWT/no raw vectors in any response, path-traversal
and unsupported-MIME rejection, rate limiting, CLIP/SigLIP2/Compare modes, failure isolation
between engines, safe candidate image URLs, Arabic/mixed-text/text-only/image-only/neither
query shapes) using a mocked `demo_router._forward` -- no real network call to `/v2/match` or
`/v2/ab/match` is made, and SigLIP2 is never downloaded. `tests/test_festival_console.py` and
`tests/test_festival_ab_ui.py` (both pre-existing, unmodified) prove the redesigned console
still satisfies every P13-era security/content guarantee. The full suite -- 227 tests, plus 7
pre-existing tesseract-only skips -- passes together.

## 10. Known limitations

- **Live real-model smoke testing of the demo facade itself was not performed** in this
  sandbox (no network access to download CLIP/SigLIP2 weights, as documented in the P13 doc).
  All facade-level tests mock `demo_router._forward`, proving the facade's own logic (identity
  minting, rate limiting, response shaping, asset serving) is correct, but not that a live
  Azure canary round-trip through the real `/v2/match`/`/v2/ab/match` produces the exact
  response shape assumed here. **Before enabling `DEMO_FACADE_ENABLED=true` on the Azure
  canary, run the manual curl smoke tests below against the real deployment.**
- The self-loopback HTTP call (`demo_router._forward` -> `http://127.0.0.1:{PORT}/v2/match`)
  assumes a single async-capable process (uvicorn/gunicorn with an async worker) that can
  accept a new inbound connection while a request handler is awaiting I/O elsewhere. This is
  the deployment model this service already uses; a WSGI-style synchronous single-threaded
  worker would deadlock on this pattern and must not be used.
- `DEMO_ASSET_DIR` must be populated by the operator (image files matching each demo item's
  `candidateFilename`) -- this service has never stored raw image bytes itself (see the P13
  doc), so there is no automatic way to backfill these from existing corpus data.
- The demo facade's own rate limiter (like `app.py`'s global one) is in-process memory, not
  shared across multiple worker processes -- with more than one worker, the effective limit is
  `DEMO_RATE_LIMIT_PER_MINUTE * worker_count`. Acceptable for a single-VM festival canary;
  would need a shared store (e.g. Redis) for a multi-instance deployment.

## 11. SigLIP2 uncalibrated status (repeated)

`siglip2` responses (single-mode and compare) always carry `"calibrationStatus":
"uncalibrated"`. The console's calibration tooltip and the P13 doc's calibration warning both
apply unchanged: SigLIP2's similarity scores are retrieval signals, not ownership
probabilities, and are never directly comparable to CLIP's scores.

## 12. CPU latency expectation

Per the task's Azure validation, SigLIP2 CPU inference is expected to take roughly 2-3 seconds
per query on the `Standard_D8as_v5` canary. The console's processing indicator is a plain CSS
spinner with static text ("Analyzing multimodal evidence…" / "جارٍ تحليل الأدلة متعددة
الوسائط…") -- it never fabricates a completion percentage, since no such signal exists.

## 13. Human-review requirement (repeated)

Every result view -- single-engine and compare -- permanently displays "AI-assisted candidate
retrieval — human verification required." A match is retrieval evidence for a human reviewer,
never an automatic confirmation of ownership or identity.

## 14. P14.1 — rebrand ("Multimodal AL Engine") and new public hostname

P14.1 is a branding/content/UI refinement only: no matching logic, embedding dimension, model
configuration, authentication, demo facade security control, MongoDB configuration, API
contract, tenant/site isolation, rate limit, production default engine (`clip_v1` stays
`DEFAULT_EMBEDDING_ENGINE`), or deployment/nginx/DNS configuration changed. Only
`templates/clip-console.html`, `static/clip-console.css`, `static/clip-console.js`, and the two
pre-existing branding-string test assertions in `tests/test_festival_console.py` (which
literally asserted the old brand strings this rebrand removes) changed.

### New product name

The console's visible product/site name is now **Multimodal AL Engine** (Arabic: `محرك الذكاء
متعدد الوسائط`), replacing the old "CLIP AI Engine" name. The company wordmark "AL-AMEN
TECHNOLOGY" / "Al-Amen Technology" no longer appears anywhere in the rendered page or its
static assets -- the header now reads "Multimodal AL Engine" / "Urban Intelligence AI Lab"
instead of "AL-AMEN TECHNOLOGY" / "Urban Intelligence AI Lab" / "CLIP AI Engine". The browser
tab title is now `Multimodal AL Engine — Urban Intelligence` (was `CLIP AI Engine — Al-Amen
Technology`). `tests/test_rebrand_p14_1.py` enforces all of this.

Internal, non-visible identifiers are explicitly unchanged, per the P14.1 brief: `clip_v1`,
`siglip2_v1`, `/demo/match`, `/v2/match`, `/v2/ab/match`, every Python class/module name, every
environment variable, and every field in every API response.

### New public hostname (documentation only -- no infrastructure change)

**Intended new public hostname:** `ai-engine.al-amentech.io`
**Previous hostname:** `clip.al-amentech.io`

Planned eventual traffic path (not yet implemented):

```
ai-engine.al-amentech.io
        |
      nginx
        |
  festival canary
```

Recommended later infrastructure action (**not implemented by this commit**): configure nginx
to answer `https://clip.al-amentech.io` with an `HTTP 301` redirect to
`https://ai-engine.al-amentech.io`, once DNS for the new hostname is live and validated. This
change is explicitly out of scope here -- no nginx configuration, DNS record, or redirect was
added, modified, or deployed as part of P14.1. The application itself has no hostname-specific
logic (CORS/allowed origins in `app.py` are unrelated to the Festival console's own static
hostname and were not touched), so it continues to work correctly under either hostname without
any code change once DNS/nginx are updated separately.

### Rollback

Identical to the P14 rollback (section 8): this is a pure static-asset/template content change
with no new environment variable and no behavior gated by a flag. To revert the rebrand
specifically, redeploy the previous commit's `templates/clip-console.html`,
`static/clip-console.css`, and `static/clip-console.js` -- no database, config, or
infrastructure change is involved either way.
