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
- signed request ID;
- `demoData` (boolean) and `datasetVersion` — see
  [Demo/production governance contract (F-02)](#demoproduction-governance-contract-f-02)
  below for the exact required shape of these two claims.

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

## Demo/production governance contract (F-02)

**Historical defect (now removed).** `internal_auth.require_identity()` used
to silently default a missing/absent `demoData` claim to `True` and a
missing/absent `datasetVersion` claim to the hardcoded literal
`"festival-2026"`:

```python
# Removed — never do this again.
dataset_version=str(claims.get("datasetVersion") or "festival-2026"),
demo_data=bool(claims.get("demoData", True)),
```

A Backend signing bug that omitted these claims was therefore invisible: the
request was accepted, and every item/query it touched was silently
mislabeled as festival demo data under a stale, hardcoded dataset version —
with no error, no audit trail, and no way for an operator to tell it had
happened.

**Current contract.** `demoData` and `datasetVersion` are required, signed
claims verified from the JWT payload only — never defaulted, never read from
request JSON, headers other than the identity token itself, or query
parameters. Backend's `utils/internalJwt.js`
(`normalizeClipGovernanceClaims`) signs claims in this exact shape; CLIP
independently re-verifies every field rather than trusting Backend to have
done so correctly:

| `demoData` claim | `datasetVersion` claim | Result |
|---|---|---|
| absent | (any) | `401 missing_demo_data` |
| present, not a JSON boolean (e.g. the string `"true"`) | (any) | `401 invalid_demo_data` |
| `true` | absent, `null`, or blank/whitespace-only string | `401 invalid_dataset_version` |
| `true` | non-empty string | **valid** — `ServiceIdentity.demo_data=True`, `dataset_version=<that string>` |
| `false` | present as a non-null value (any string, including a real dataset version) | `401 contradictory_demo_claims` |
| `false` | absent or `null` | **valid** — `ServiceIdentity.demo_data=False`, `dataset_version=None` |

`dataset_version=None` is the correct, valid value for production traffic —
it is never a placeholder and never coerced to a string. This mirrors
Backend's own signing contract exactly: `demoData=false` always signs
`datasetVersion: null`, precisely so production traffic can never carry (or
have CLIP infer) a festival dataset version.

**Corpus registration, release, and re-embed (`POST /items`,
`POST /items/{id}/release`, `POST /items/{id}/re-embed`).** A client cannot
supply `demoData`/`datasetVersion` in the request body at all — both are on
`parse_item_payload`'s `forbidden` field list and any attempt to set them
is rejected with `400 system_fields_forbidden` before storage is ever
touched. Stored corpus metadata for these two fields is copied from
`ServiceIdentity` unconditionally (`prepare_item_for_storage`), never from
the request body.

That closes the payload-vs-JWT channel, but not the JWT-vs-stored-state
channel: updating (`POST /items` on an existing id), releasing, or
re-embedding an item is now checked against the item's *already-stored*
`demoData`/`datasetVersion` before the write proceeds
(`assert_dataset_scope_consistent`). A production identity
(`demoData=False`) can never release, re-embed, or update an item that was
created as demo data (or vice versa), and an identity signed for one
`datasetVersion` can never touch an item stored under a different one. A
mismatch fails closed with `409 governance_mismatch` and is recorded as a
`governance_mismatch_rejected` audit-log event (existing/identity
demoData and datasetVersion values, item id, tenant id) — a rejected,
contradictory write is never silent.

**Candidate retrieval (`POST /match`).** `list_released_items` (both the
in-memory and MongoDB repository implementations) now filters candidates by
the signed identity's dataset scope, in addition to the pre-existing
tenant/site/`released`/`eligibleForMatching`/`modelId`/`modelRevision`/
`embeddingDimension` filters (all preserved unchanged):

- `demoData=True` identities only match candidates with
  `demoData=True` **and** the identical `datasetVersion` — a demo query can
  never unintentionally retrieve candidates seeded under a different
  festival dataset version.
- `demoData=False` identities only match candidates with `demoData` not
  `True` — production queries never inherit `festival-2026` (or any other
  demo dataset's) candidates, and have no `datasetVersion` to filter by
  since production items never carry one.

**Response schema.** `MatchItemModel.datasetVersion` is `Optional[str]`
(previously a required `str`) so a matched production candidate — which
legitimately has `datasetVersion=None` — serializes correctly instead of
failing Pydantic response validation.

**Unaffected.** HS256 signature verification, issuer/audience, service
identity, expiry, tenant scoping, site membership, action scope, and the
`X-Request-Id`-equals-signed-`requestId` binding are all independently
verified exactly as before this change — none of those checks were
weakened, reordered relative to each other, or made to depend on the new
governance checks. Public `/health/live` and `/health/ready` remain
unauthenticated.

**Migration/deployment order.** This CLIP-Service change is a hard
prerequisite: Backend main must already sign `demoData`/`datasetVersion` on
every internal CLIP JWT (`utils/internalJwt.js`'s
`normalizeClipGovernanceClaims`) before this branch is deployed, or every
Backend→CLIP call will be rejected with `401 missing_demo_data`. Backend's
`signInternalClipToken()` itself already enforces the same shape at sign
time (throws if `demoData` isn't a boolean, or if `datasetVersion` is
missing/blank when `demoData=true`, or non-null when `demoData=false`), but
independently confirm every call site that signs a CLIP token
(`utils/clipMatchClient.js`, `utils/clipCorpusClient.js`) actually threads
`demoData`/`datasetVersion` through to `signInternalClipToken()` before
relying on this — a caller that doesn't pass them will hit Backend's own
sign-time rejection first (never Backend's JWT reaching CLIP at all), which
is Backend's fail-closed behavior working as intended, not a CLIP-Service
defect.
