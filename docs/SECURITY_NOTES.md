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

Required actions are `corpus:write`, `corpus:release`, `corpus:read`,
`match:execute`, `health:read`, and `metrics:read`. Browser clients must call the
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
