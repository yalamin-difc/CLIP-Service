# CLIP-Service

Multimodal (CLIP + OCR + barcode) retrieval service for Al-Amen Technology's AI
Lost & Found / Urban Intelligence Platform, built for the Dubai AI Festival 2026.
CLIP-Service is retrieval infrastructure: it ranks candidates for an authenticated
Backend caller. The Urban Intelligence Backend remains the final match-decision
authority; see [`docs/C01_BACKEND_INTEROPERABILITY.md`](docs/C01_BACKEND_INTEROPERABILITY.md).

## Festival deployment

The public console (`/`, `/ui` and its `/static/*` assets) is served only
when **`CONSOLE_ENABLED=true`**. The default is `false`, and then those
paths return the same 404 as an unknown path. The festival deployment must
set `CONSOLE_ENABLED=true`; pilot deployments leave it unset. Health,
`/docs` and the authenticated API do not depend on this setting.

- Festival Console: https://clip.al-amentech.io/
- Swagger: https://clip.al-amentech.io/docs
- Readiness: https://clip.al-amentech.io/health/ready

See [`docs/FESTIVAL_CLIP_CONSOLE.md`](docs/FESTIVAL_CLIP_CONSOLE.md) for the
console's architecture, security model, and limitations.

## Documentation

- [`docs/SECURITY_NOTES.md`](docs/SECURITY_NOTES.md) — trust boundary, signed internal JWT contract, data protection
- [`docs/C01_BACKEND_INTEROPERABILITY.md`](docs/C01_BACKEND_INTEROPERABILITY.md) — the retrieval/match contract with the Backend
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — model ID, revision, embedding dimension, intended use
- [`docs/DEPENDENCY_MANIFEST.md`](docs/DEPENDENCY_MANIFEST.md) — pinned runtime dependencies
- [`docs/BENCHMARK_REPORT_TEMPLATE.md`](docs/BENCHMARK_REPORT_TEMPLATE.md) — latency/throughput benchmark template
- [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md) — filled retrieval-quality report (precision@1, recall@k, MRR, PR curve, by language) — see `docs/eval/README.md`, "Retrieval-quality benchmark (D-5)"
- [`README-DEPLOY-GPU-VM.md`](README-DEPLOY-GPU-VM.md) — GPU VM deployment notes
- [`docs/P13_SIGLIP2_AB_IMPLEMENTATION.md`](docs/P13_SIGLIP2_AB_IMPLEMENTATION.md) — SigLIP2 alongside CLIP: dual embeddings, the `/v2` API, A/B testing, rollback

## SigLIP2 / A/B testing (experimental, disabled by default)

This service also supports an experimental `google/siglip2-so400m-patch14-384` engine
(`siglip2_v1`) alongside the production CLIP engine (`clip_v1`), for internal evaluation via a
new `/v2/*` API surface. It is fully disabled by default (`SIGLIP2_ENABLED=false`,
`AB_TEST_ENABLED=false`, `AB_UI_ENABLED=false`) and never changes any legacy endpoint's
behaviour. See [`docs/P13_SIGLIP2_AB_IMPLEMENTATION.md`](docs/P13_SIGLIP2_AB_IMPLEMENTATION.md).

## Development

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/ci_static_checks.py
```

Docker images: `Dockerfile.cpu` (default) and `Dockerfile.gpu` (CUDA). Both
require `INTERNAL_JWT_SECRET` (>= 32 bytes), `MONGODB_URI` when
`STORAGE_MODE=mongodb`, and an explicit `INFERENCE_DEVICE` (`cpu` or `cuda`).

The expected internal JWT issuer is `INTERNAL_JWT_ISSUER`. Its default is
`lost-found-backend`, which matches the Backend's `CLIP_INTERNAL_JWT_ISSUER`
default. See [`docs/SECURITY_NOTES.md`](docs/SECURITY_NOTES.md#issuer-configuration-f1)
for the rollout setting `INTERNAL_JWT_ACCEPTED_ISSUERS`. For pilot
deployments, keep `DEMO_FACADE_ENABLED=false` and `CONSOLE_ENABLED=false`
(both are the default), so that every `/demo/*` route and the public console
return 404.

To check that a pilot's index is empty before go-live, set
`ADMIN_INDEX_COUNTS_ENABLED=true` for the check. This turns on the
read-only `GET /admin/index-counts` endpoint, which needs the `index:count`
action and returns per-site counts only. See
[`docs/SECURITY_NOTES.md`](docs/SECURITY_NOTES.md#index-counts-per-tenant-and-site-f2).
