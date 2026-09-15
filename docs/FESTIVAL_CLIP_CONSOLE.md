# Festival CLIP AI Engine console

## Purpose

`GET /` used to return a bare `{"status": "ok"}` JSON body. This console gives the
Dubai AI Festival 2026 a public, bilingual (English/Arabic) demonstration surface
that explains the service and shows its real, live public state — without adding a
single new capability an unauthenticated visitor didn't already have, and without
touching any protected route's contract.

## Architecture

```
Browser (GET /)
    |
FastAPI "/" route -- renders templates/clip-console.html once per request,
    |                 interpolating real, live, non-secret provenance
    |                 constants (see "Model provenance" below).
    |
static/clip-console.css, static/clip-console.js -- served by a StaticFiles
    mount at /static, no build step, no framework.
    |
Browser JS -- polls the existing PUBLIC GET /health/ready every 15s for the
              Service Status panel. Never calls a protected route.
```

The console is a plain server-rendered HTML page plus two static assets. There is
no template engine dependency (no Jinja2): the handler does a small, fixed set of
`str.replace()` substitutions against server-controlled constants only — never
against request input — so there is no injection surface.

## Routes

| Route | Behavior | Change |
| --- | --- | --- |
| `GET /` | Renders the HTML console | **Changed** from `{"status": "ok"}` JSON to HTML |
| `GET /static/*` | Serves `clip-console.css` / `clip-console.js` | **New** (StaticFiles mount) |
| `GET /docs` | FastAPI Swagger UI | Unchanged |
| `GET /redoc` | FastAPI ReDoc | Unchanged |
| `GET /openapi.json` | OpenAPI spec | Unchanged |
| `GET /health/live` | Public liveness | Unchanged |
| `GET /health/ready` | Public readiness | Unchanged (console *reads* it, never modifies its shape) |
| `GET /ui` | Legacy bare endpoint listing | Unchanged, left as-is (not in scope) |
| `POST /items`, `GET /items`, `POST /items/{id}/release`, `DELETE /items/{id}`, `POST /items/{id}/re-embed`, `GET /audit-logs`, `POST /match`, `POST /analyze-image`, `GET /health`, `GET /metrics` | Protected, signed-identity required | **Unchanged** |

No existing endpoint's request/response contract changed. `GET /` is the only
route whose *response format* changed, and nothing in the repository, the CI
workflow, or the Dockerfiles depended on it returning JSON (see "Audit findings"
below) — automated health checks already target `/health/live` and
`/health/ready`, not `/`.

## Security model

- **CSP is scoped, not global.** `Content-Security-Policy: default-src 'self'; script-src 'self'; ...` and `X-Frame-Options: DENY` are only added to `GET /` and `GET /static/*` responses. `/docs`, `/redoc`, `/openapi.json`, and every API route are untouched, so Swagger/ReDoc's own asset loading is never affected. `X-Content-Type-Options: nosniff` and `Referrer-Policy: strict-origin-when-cross-origin` are safe to apply to every response and are applied globally.
- **No inline script or style.** The console's CSS and JS live in `static/clip-console.css` / `static/clip-console.js`; the CSP's `script-src 'self'` and `style-src 'self'` would block an inline `<script>`/`<style>` if one were ever added by mistake.
- **The console never authenticates as a service.** `static/clip-console.js` never imports, reads, or constructs an `Authorization` header, never signs an HS256 token, and never reads `INTERNAL_JWT_SECRET` or `MONGODB_URI` — those values do not exist in a browser process. `tests/test_festival_console.py` asserts all of this directly (see "How to test").
- **No new public endpoint into protected functionality.** The console's Image Analysis / OCR / Barcode / Matching panels are rendered in an explicit, visibly-disabled "unavailable in this public demonstration" state (see "Known limitations" below) rather than calling `/analyze-image` or `/match` without a signed identity.

## Why protected CLIP endpoints remain protected

`internal_auth.require_identity()` verifies an HS256-signed internal JWT carrying
issuer, audience, service name, tenant ID, site ID (checked against a signed
allow-list), a signed action allow-list, an expiry, a request ID that must match
the `X-Request-Id` header, and the `demoData`/`datasetVersion` governance claims
(see [`docs/C01_BACKEND_INTEROPERABILITY.md`](./C01_BACKEND_INTEROPERABILITY.md)
and [`docs/SECURITY_NOTES.md`](./SECURITY_NOTES.md)). The signing secret
(`INTERNAL_JWT_SECRET`) must never exist outside the Backend and this service. A
browser cannot hold that secret without exposing it to every visitor, so no
browser page — this one included — can ever call `/items`, `/match`, or
`/analyze-image` directly. This is by design and predates this change; the
console does not weaken it and does not attempt to route around it with an
unauthenticated proxy.

## How Backend authentication works

The Urban Intelligence Backend signs a short-lived HS256 JWT per request using
its own copy of `INTERNAL_JWT_SECRET`, sets it as `Authorization: Bearer <token>`,
and sets `X-Request-Id` to the same value the token's `requestId` claim carries.
`CLIP-Service` verifies the signature, issuer, audience, expiry, request-ID match,
site/action permissions, and the demo/production governance claims before running
any protected operation. The Backend — never the browser, never CLIP — is the
system that is allowed to hold the signing secret.

## Demo-data controls

`demoData` and `datasetVersion` are signed claims on the internal JWT, verified
independently of any request body, and stamped onto every corpus item from the
verified identity (never from client input). The console's "Trusted AI" panel
surfaces this as "Synthetic demo-data separation" and the page banner displays
"DEMO DATA / بيانات تجريبية" — this is a UI label, not a governance mechanism; the
actual enforcement is unchanged and lives entirely in `internal_auth.py`.

## English/Arabic UX

- A visible `English | العربية` toggle swaps every `data-i18n`-tagged string via a
  JS dictionary (`STRINGS.en` / `STRINGS.ar` in `static/clip-console.js`) and sets
  `lang`/`dir` on the document root (`dir="rtl"` for Arabic).
- The choice is remembered per-browser via `localStorage` (best-effort; wrapped in
  try/catch since private-mode/blocked storage must never break the toggle).
- No OCR/barcode/model output is ever machine-translated — Section 8 of the
  originating prompt is explicit that detected text must render exactly as
  returned. Because live analysis is not enabled in this console (see below),
  this currently only matters as a standing rule for any future enablement.

## Model provenance panel

`Model ID`, `Model revision`, and `Scoring version` are the same fixed,
non-secret constants (`MODEL_NAME`, `MODEL_REVISION`, `SCORING_VERSION`) already
published in [`docs/MODEL_CARD.md`](./MODEL_CARD.md) and already returned by the
protected `/health` and `/analyze-image` responses. `Embedding dimension` and
`Preprocessing version` are the service's real, live module state
(`embedding_dimension`, `preprocessing_version`), set once model warm-up actually
succeeds — if warm-up has not completed, the console renders `Unavailable` rather
than a fabricated number. **`GET /health/ready`'s public response body was not
changed** to source these fields; they are read directly from the same in-process
values `/health` and `/analyze-image` already expose to authenticated callers, at
render time, by the `/` route handler itself (see `render_console_html()` in
`app.py`). This keeps `/health/ready`'s existing, tested, monitored contract
completely untouched.

## Service Status panel

Sourced entirely from the existing public `GET /health/ready` response, polled
client-side every 15 seconds. The panel maps:

- `checks.modelLoaded` → "Model" row
- `checks.ocrDependencyReady` → "OCR" row
- `checks.databaseReady` → "Storage" row
- `checks.device` → "Device" row
- overall `status === "ready"` → green; a known-but-not-ready core check → amber;
  otherwise (including a network failure) → red

Nothing here is fabricated: a field `/health/ready` doesn't currently expose is
displayed as "Unavailable", never guessed.

## Known limitations

**Live image analysis, OCR, barcode, and matching are not enabled in this public
console.** They require a signed internal service identity, which can never be
issued to a browser (see "Why protected CLIP endpoints remain protected"). The
console's Image Analysis section explains this directly to the visitor and lets
them preview a locally-chosen image in the browser only — the file is never
uploaded anywhere, because there is no browser-safe endpoint to send it to.

### What a secure enablement would require (not implemented here)

If a live, in-browser demo becomes a festival requirement, the safe path is a new
**Backend-mediated, browser-session-authenticated facade** — not a public,
unauthenticated proxy to CLIP. Concretely, it would need:

1. A new Backend route (e.g. `POST /api/festival/demo-analyze`) that requires the
   visitor's existing Backend session/auth, is rate-limited per visitor, and is
   scoped to a demo tenant/site.
2. That route calls `CLIP-Service`'s existing `/analyze-image` (and optionally
   `/match`) exactly as the Backend already does for real item ingestion, signing
   its own internal JWT server-side with `datasetVersion` fixed to a festival
   demo dataset.
3. The Backend response to the browser must be sanitized: only safe fields
   (OCR text, barcode value/type, similarity scores, candidate title/category/
   thumbnail) — never an embedding, never another visitor's item, never
   reporter/contact data.
4. This console's JS would then call that one Backend endpoint instead of
   rendering the current "unavailable" notice.

This repository intentionally does **not** implement that Backend route — it is
out of scope for CLIP-Service and belongs in `ai-lost-and-found-ver-2` (the
Backend), same as every other browser-facing feature in this platform.
(`test/unit/legacyClipEndpointsRetired.test.js` in that repository confirms the
Backend already retired its own former public on-demand analyze/match endpoints
as part of the C-01 retrieval-contract hardening, so no such facade currently
exists there either.)

## How to test

```bash
cd CLIP-Service
python -m unittest discover -s tests -v
# or:
pytest tests/
# Console-specific tests only:
pytest tests/test_festival_console.py -v
```

`tests/test_festival_console.py` covers: `GET /` returns 200 `text/html`; the
page contains "CLIP AI Engine" and "Dubai AI Festival 2026"; it identifies demo
data in both languages; Arabic/RTL resources exist; `/docs` and `/openapi.json`
still work; `/health/live` and `/health/ready` are unchanged; `/items`, `/match`,
and `/analyze-image` remain protected (401 without a signed identity); the
rendered page and the console's JS never contain the JWT secret value, a Mongo
connection string, or a raw embedding vector; the console's JS never signs a
token or sets an `Authorization` header; the CSP header is present on `/` but
absent on `/docs` and `/openapi.json`; and the model-provenance panel renders the
service's real `MODEL_NAME`/`MODEL_REVISION`/`SCORING_VERSION` constants. It also
spot-checks (alongside the full existing suite) that the demo-governance claim
contract, tenant/site scoping, and the OCR/barcode response shape from
`/analyze-image` are unchanged.

## How to deploy

No Dockerfile changes were required: `COPY . .` in both `Dockerfile.cpu` and
`Dockerfile.gpu` already copies `templates/` and `static/` into the image, and
neither file is excluded by `.dockerignore`. Deploy exactly as before:

```bash
docker build -f Dockerfile.cpu -t clip-service:latest .
docker run -e INTERNAL_JWT_SECRET=... -e MONGODB_URI=... -e INFERENCE_DEVICE=cpu \
  -e STORAGE_MODE=mongodb -p 8080:8080 clip-service:latest
```

`https://clip.al-amentech.io/` will serve the console once this branch is merged
and deployed; `/docs` and `/health/ready` continue to work exactly as before.

## How to roll back

Revert the merge commit for `feature/festival-clip-console` (or redeploy the
prior image tag). `GET /` returns to `{"status": "ok", "requestId": ...}`; no
other route, data, or credential is affected either way, because nothing outside
`GET /`, the new `/static` mount, and the two new files this PR adds was changed.

## Known limitations (summary)

- No live in-browser image analysis/OCR/barcode/matching demo (see above) —
  requires a new Backend-side facade this repository does not implement.
- `Embedding dimension` / `Preprocessing version` on the console show
  "Unavailable" until the service has completed a real model warm-up (i.e. in any
  environment without `torch`/`transformers` installed and reachable, such as a
  minimal CI or sandbox host) — this is the same real state `/health`'s protected
  response already reflects, never a fabricated value.
- This branch's Docker build/run could not be executed in the environment this
  change was authored in (no Docker daemon available there); the existing
  `cpu-image` CI job (`.github/workflows/clip-ci.yml`) builds `Dockerfile.cpu`,
  starts the container against real MongoDB, and smoke-tests it on every push —
  that job is the authoritative build/serve verification and will run on this
  PR.
