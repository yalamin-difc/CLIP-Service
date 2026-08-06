# CLIP Festival acceptance

## Implemented changes

- Fixed OCR worker dispatch by binding keyword-only `lang` and `psm` arguments
  before invoking AnyIO's worker thread.
- Added real-image English, Arabic, mixed-language, numeric identifier, and Code
  128 barcode tests, plus invalid/oversized image, timeout, disabled, and missing
  OCR dependency coverage.
- Added signed internal service identities, per-action authorization, tenant/site
  predicates, model-compatible released candidate filtering, and protected corpus.
- Added explicit memory/MongoDB modes, startup dependency checks, CPU/CUDA
  selection, model warmup, safe liveness/readiness, protected metrics, upload
  controls, request/OCR concurrency, rate limiting, and timeouts.
- Removed raw embeddings from responses and rejected client-controlled embeddings,
  release/status fields, thresholds, model metadata, and tenant context.
- Added CPU/GPU images, pinned dependencies/model revision, benchmark tooling,
  CycloneDX SBOM, model card, dependency manifest, and security notes.

## Changed endpoints

- `GET /health/live`: public minimal liveness.
- `GET /health/ready`: public safe readiness checks.
- `GET /health`: detailed dependency status; requires `health:read`.
- `GET /metrics`: requires `metrics:read`.
- `POST /items`: create/update corpus data; requires `corpus:write`.
- `GET /items`, `GET /items/{id}`: scoped reads; require `corpus:read`.
- `POST /items/{id}/release`: requires `corpus:release`.
- `DELETE /items/{id}`, `POST /items/{id}/re-embed`: require `corpus:write`.
- `POST /match`, `/analyze-image`, `/encode-image`, `/encode-text`, `/similarity`:
  require `match:execute`. Encode endpoints return model metadata, not vectors.

## Authentication model

The service verifies HS256 internal JWTs using `INTERNAL_JWT_SECRET`. Required
claims are issuer, audience, service, tenantId, siteId, siteIds, actions, expiry,
and requestId. `siteId` must be included in `siteIds`; the JWT request ID must
match `X-Request-Id`. Browsers do not receive signing credentials and must use
the authenticated backend.

## Storage requirements

Set `STORAGE_MODE=memory` only for explicitly configured local development or
tests. Festival, staging, and production require `STORAGE_MODE=mongodb` and
`MONGODB_URI`. Startup fails when MongoDB is missing or unreachable; there is no
memory fallback.

## Tenant-isolation tests

The suite verifies mandatory identity, action rejection, signed provenance,
forbidden browser system fields, tenant-scoped item reads, permitted-site
filtering, released-only matching, model compatibility, and protected metrics.
MongoDB indexes and queries include tenant/site fields.

## OCR test results

On Tesseract 5.3.4 with English and Arabic data, all 29 service tests pass,
including actual-image English, Arabic, mixed Arabic-English, numeric identifier,
and barcode extraction. Real OCR and barcode scans also pass through the HTTP
execution and serialization paths. Timeout, disabled OCR, missing dependency,
invalid image, and oversized image cases pass.

Run:

```bash
python3 -m unittest discover -s tests -v
```

## Model and device information

- Model: `openai/clip-vit-base-patch32`
- Revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`
- Expected dimension: 512
- Device: explicitly set `INFERENCE_DEVICE=cpu` or `cuda`
- Warmup: performed during startup; readiness stays false until successful

Detailed authorized health exposes the effective device, model, revision,
dimension, and warmup state. CUDA configuration fails rather than reporting a
GPU when CUDA is unavailable.

## Benchmark commands

```bash
python3 scripts/benchmark.py --candidates 1000 --iterations 100 --output benchmark-1000.json
python3 scripts/benchmark.py --candidates 10000 --iterations 100 --output benchmark-10000.json
python3 scripts/benchmark.py --candidates 50000 --iterations 100 --output benchmark-50000.json
```

Record deployment results in `docs/BENCHMARK_REPORT_TEMPLATE.md`.

## Known limitations

- Matching retrieves up to 50,000 compatible released records and exhaustively
  scans their embeddings. Ranking is O(n × 512), with linear memory/network cost.
- OCR/barcode evidence is returned for trusted backend use but is not persisted in
  match audit events.
- Process-local rate limiting must be complemented by gateway limits across
  replicas.
- This phase does not introduce a vector database; benchmark results should drive
  that decision after the festival target is measured.
- GPU image execution requires an NVIDIA runtime and compatible host driver.

## Rollback

1. Stop routing traffic to the new revision.
2. Redeploy the previous known-good image and configuration.
3. Keep MongoDB records; do not migrate or copy them into memory.
4. Rotate the internal JWT secret if identity handling contributed to rollback.
5. Verify previous liveness and backend contract checks before restoring traffic.
6. Preserve audit and benchmark artifacts for incident review; never export raw
   images or embeddings.
