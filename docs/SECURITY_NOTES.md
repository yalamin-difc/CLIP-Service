# Security notes

## Trust boundary

Corpus and matching APIs accept only HS256-signed internal JWTs. Tokens require
issuer, audience, service name, tenant ID, selected site ID, permitted site IDs,
actions, expiry, request ID, and two demo/production governance claims:
`demoData` (an explicit boolean — never defaulted) and `datasetVersion` (a
required non-empty string when `demoData` is `true`; must be absent/null
when `demoData` is `false`). A token missing or misshaping either governance
claim is rejected before any protected operation runs — see
[`docs/C01_BACKEND_INTEROPERABILITY.md`](./C01_BACKEND_INTEROPERABILITY.md#demoproduction-governance-contract-f-02)
for the exact contract. Authenticated internal requests must include
`X-Request-Id`, and that header must exactly match the signed request ID claim.
Signing keys must come from a managed secret store, contain at least 256 bits of
entropy, and be rotated outside the service image.

### Issuer configuration (F1)

| Variable | Default | Meaning |
| --- | --- | --- |
| `INTERNAL_JWT_ISSUER` | `lost-found-backend` (previously `dubai-festival-backend`) | The issuer this service expects. It must match the Backend's `CLIP_INTERNAL_JWT_ISSUER`. A blank value uses the default. |
| `INTERNAL_JWT_ACCEPTED_ISSUERS` | empty | A comma-separated list of extra issuers that are accepted **only during an issuer rollout**, for example `dubai-festival-backend` while the Backend is redeployed. Leave it empty at all other times. |
| `INTERNAL_JWT_AUDIENCE` | `clip-service` | It must match the Backend's `CLIP_INTERNAL_JWT_AUDIENCE`. |

Tokens this service mints for itself (the demo facade) always use
`INTERNAL_JWT_ISSUER`. A token from any issuer that is not configured is
rejected with `401 invalid_identity`.

### Index counts per tenant and site (F2)

`GET /admin/index-counts` reports how many items are indexed for the
signed identity's tenant and for each of its permitted sites. Use it
before go-live to check that a pilot's index is empty.

- **Off by default.** It runs only when `ADMIN_INDEX_COUNTS_ENABLED=true`.
  When the setting is off, the route returns 404, exactly like an unknown
  path.
- **Needs its own action, `index:count`.** A token with only this action
  cannot read, write, release or match corpus items. A token without it is
  rejected with 403.
- **Scoped by the token, never by the request.** It counts only the
  token's `tenantId` and `siteIds`. Query parameters cannot widen that
  scope, and another tenant's items are never counted, even on a site with
  the same name.
- **Returns counts only.** Each site reports `total`, `released`, `demo`,
  `nonDemo` and `byStatus`. No item IDs, titles, descriptions or embeddings
  are returned. Demo and production items are both counted, whatever the
  token's `demoData` claim says, because the aim is to see whether anything
  is indexed at all.
- **Read-only and audited.** Each call adds an
  `admin_index_counts_read` audit entry with the tenant, the site IDs and
  the total.

The Backend's `npm run pilot:verify-clean -- --check-clip` signs an
`index:count` token and calls this endpoint.

### Pilot deployments

Leave `CONSOLE_ENABLED` unset or `false` (the default). Then `/`, `/ui` and
`/static/*` answer exactly like an unknown path: a 404 with
`{"detail": "Not Found"}` and without the console's CSP headers.
`tests/test_console_flag_f4.py` covers this, and also checks that the
console is unchanged when the flag is on.

In a pilot, keep `DEMO_FACADE_ENABLED=false` (the default). Then every
`/demo/*` route returns 404, even when the other `DEMO_*` values are set.
`tests/test_pilot_config_f1.py` lists the app's routes and checks each
`/demo/*` route, so any `/demo/*` route added later is checked too. Keep
`INTERNAL_JWT_ACCEPTED_ISSUERS` empty. Turn on `ADMIN_INDEX_COUNTS_ENABLED`
only while checking that the pilot index is empty, and turn it off again
afterwards. The full list of pilot settings is
in the Backend's `docs/PILOT_CONFIGURATION.md`.

Required actions are `corpus:write`, `corpus:release`, `corpus:read`,
`match:execute`, `health:read`, `metrics:read`, and `index:count` (F2, below). Browser clients must call the
trusted backend; CORS is not an authentication control.

## Data protection

- Mongo queries include tenant and permitted-site predicates.
- Candidate queries additionally require release and pinned-model compatibility,
  and are scoped to the signed identity's demo/production dataset: a demo
  identity only matches candidates in its own signed `datasetVersion`; a
  production identity only matches non-demo candidates and never inherits a
  festival dataset.
- Client input cannot set `demoData`/`datasetVersion` on a corpus item; both
  are stamped from the signed identity and validated against any
  already-stored item before an update, release, or re-embed is applied
  (fail-closed `409` on a mismatch, with an audit-log entry).
- API responses never contain embeddings.
- Client input cannot set release state, status, thresholds, embeddings, model
  metadata, tenant metadata, or embedding dimensions.
- Audit records contain identifiers, outcome metadata, content hashes, and sizes;
  they exclude images, embeddings, query text, and OCR evidence.

## Operational controls

Uploads enforce MIME allowlisting, byte and decoded-pixel limits, decoding
validation, and decompression-bomb handling. Request/OCR concurrency, rate,
inference concurrency, request timeout, inference timeout, and OCR timeout are
bounded by environment configuration. CLIP inference runs in a bounded worker
thread so model execution cannot block the FastAPI event loop.
Metrics and detailed health require signed operational identities.

## Deployment requirements

Festival, staging, and production require `STORAGE_MODE=mongodb`, reachable
MongoDB, `INTERNAL_JWT_SECRET`, and an explicit `INFERENCE_DEVICE`. Restrict
network access to the trusted backend and operations plane. Use TLS for MongoDB
and all service traffic.

## Residual risks

HS256 uses a shared secret; workload identity or asymmetric JWT verification is
preferred when the showcase platform supplies it. Rate limiting is process-local,
so the gateway must provide distributed enforcement for multi-replica deployments.
Container bases are pinned to immutable multi-platform manifest digests; deployment
promotion should additionally verify the selected platform digest and signature.
