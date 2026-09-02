# C-01 Backend interoperability evidence

## Scope

This evidence covers only the hardened CLIP-Service contract consumed by a trusted
Backend service identity. No CLIP-Service runtime behavior was changed for C-01.

## Backend inspection status

The requested repository `yalamin-difc/Backend` was inspected through the
authenticated GitHub CLI before test changes were made. GitHub returned
`Could not resolve to a Repository with the name 'yalamin-difc/Backend'`, and the
repository was absent from the authenticated principal's accessible repository
list. The current Backend C-01 adapter therefore could not be reviewed or executed
from this run.

This is a cross-repository verification blocker. Passing CLIP-Service tests alone
does not fully close C-01.

## Exact tested sequence

`tests/test_backend_contract_c01.py` acts as the trusted Backend caller and uses
the current endpoint schemas without compatibility shims:

1. `POST /items` as `corpus:write` with a synthetic multipart image.
2. `POST /items/{item_id}/release` as `corpus:release`.
3. `POST /analyze-image` as `match:execute`.
4. `POST /match` as `match:execute`.
5. Verify only released candidates for the JWT tenant and permitted sites are
   evaluated.

Foreign records are created and released for another tenant and another site
before matching. The authorized match still evaluates and returns only the
intended tenant/site record.

## Signed identity contract

The tests issue an HS256 JWT with:

- configured issuer and `clip-service` audience;
- Backend service identity;
- `tenantId`, selected `siteId`, and permitted `siteIds`;
- one endpoint-specific action;
- expiry;
- signed request ID.

The same request ID is sent in `X-Request-Id` and is required on every
authenticated internal call. Tests reject invalid signatures, issuer, audience,
missing identity context, expiry, disallowed site, missing action, missing or
blank `X-Request-Id`, and request-ID mismatch with the existing structured
401/403 contract. Public `/health/live` and `/health/ready` remain unauthenticated.

## Response evidence

The flow verifies:

- model ID and pinned model revision;
- scoring version and server-controlled thresholds;
- released-only governance;
- evaluated candidate count;
- candidate tenant/site provenance;
- request ID in response header and body;
- absence of `embedding` and `clipEmbedding` at every response depth.

## Retrieval contract (branch `cursor/c01-retrieval-contract-086c`)

CLIP performs corpus management and candidate retrieval. The CLIP no-match
decision is advisory for Backend consumers. Backend remains authoritative for
application-level multimodal fusion and final MatchDecision.

Prior to this branch, `POST /match` withheld `topK` entirely whenever CLIP's
own image-similarity gate (`should_return_no_match`, based on top score and
score margin) decided `noMatch=true`. That prevented a trusted Backend
consumer from ever seeing a ranked candidate to evaluate with its own OCR,
barcode, location, time, category, and evidence-quality signals — CLIP was
acting as a second decision engine instead of retrieval infrastructure.

Changes made, strictly scoped to this:

- `POST /match` now always returns ranked, authorized, released, model-
  compatible retrieval candidates in `topK`, regardless of `decision.noMatch`.
  `decision` (`noMatch`, `reason`, `details`) is unchanged and remains
  advisory metadata for the caller — a returned candidate is retrieval
  evidence, not an accepted match.
- `explanation.similarity` gained three scalar fields — `imageCosine` (present
  only when an image query was supplied), `textCosine` (present only when a
  text query was supplied), and `combinedCosine` (the existing retrieval
  score; equals the single modality's cosine when only one modality is
  present). The existing `cosine`/`band` fields are unchanged for backward
  compatibility. No raw embedding vectors are exposed at any point.

Explicitly **not** changed, per scope lock: internal JWT authentication,
`minScore`/`minMargin`/`temperature`/model-control client overrides (still
rejected by `parse_match_request`'s `forbidden_controls` check — unmodified),
OCR engine, barcode engine, MongoDB architecture, model/revision, tenant/site
security, and no OCR-provider changes.

## Resolution status

CLIP-Service's retrieval contract is now proven to satisfy the "CLIP retrieves,
Backend decides" architecture: `topK` is available to a trusted caller
regardless of CLIP's own advisory no-match judgment, and that caller can
compute image/text-only or combined similarity components without ever
touching a raw vector.

This closes the **CLIP-Service side** of C-01. Overall C-01 status remains
**PARTIALLY RESOLVED pending Backend integration** — Backend still needs to
adopt this contract (sign internal JWTs, register/release corpus items, call
`/match` with a permissive-by-design request, and map the decomposed
`explanation` fields into its existing multimodal fusion and `MatchDecision`)
before the finding can be marked fully resolved end-to-end.

## Validation results

- Focused C-01 suite: 3 tests passed (unchanged).
- Retrieval-contract suite (`MatchRetrievalContractTests`, this branch): 10
  tests passed — low-score no-match still retrieves ranked candidates;
  client-supplied thresholds still rejected; image-only/text-only/combined
  cosine reporting; no raw vectors in any response; tenant isolation; site
  isolation; unreleased-candidate exclusion; wrong-action authorization
  rejection.
- Complete CLIP-Service suite: 44 tests total — all newly added tests pass;
  1 pre-existing failure and 2 pre-existing errors (missing `zxingcpp`
  module and a metrics-permission assertion) reproduce identically on the
  unmodified base branch (`ef04041`) and are unrelated to this change; 5
  skipped (require local Tesseract language data, unrelated to this change).
- Service source changes: `app.py` only (`/match` handler, `match_explanation`,
  `MatchExplanationSimilarityModel`) — no other endpoint, model, OCR, barcode,
  storage, or auth code touched.
- Raw embeddings remain protected.
- Signed internal JWT authentication remains mandatory.
- `minScore`/`minMargin`/`temperature` remain service-controlled; client
  attempts to set them are still rejected with `system_controls_forbidden`.
