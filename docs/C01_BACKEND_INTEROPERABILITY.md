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

The same request ID is sent in `X-Request-Id`. Tests reject invalid signatures,
issuer, audience, missing identity context, expiry, disallowed site, missing
action, and request-ID mismatch with the existing structured 401/403 contract.

## Response evidence

The flow verifies:

- model ID and pinned model revision;
- scoring version and server-controlled thresholds;
- released-only governance;
- evaluated candidate count;
- candidate tenant/site provenance;
- request ID in response header and body;
- absence of `embedding` and `clipEmbedding` at every response depth.

## Resolution status

CLIP-Service contract behavior is proven locally. Overall C-01 status is
**PARTIALLY RESOLVED** until the updated Backend repository is made accessible and
its actual adapter is run against this hardened service flow.

## Validation results

- Focused C-01 suite: 3 tests passed.
- Complete CLIP-Service suite: 34 tests passed.
- Service source changes for C-01: none.
- Raw embeddings remain protected.
- Signed internal JWT authentication remains mandatory.
