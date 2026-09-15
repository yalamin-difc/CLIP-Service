# CLIP-Service

Multimodal (CLIP + OCR + barcode) retrieval service for Al-Amen Technology's AI
Lost & Found / Urban Intelligence Platform, built for the Dubai AI Festival 2026.
CLIP-Service is retrieval infrastructure: it ranks candidates for an authenticated
Backend caller. The Urban Intelligence Backend remains the final match-decision
authority; see [`docs/C01_BACKEND_INTEROPERABILITY.md`](docs/C01_BACKEND_INTEROPERABILITY.md).

## Festival deployment

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
- [`README-DEPLOY-GPU-VM.md`](README-DEPLOY-GPU-VM.md) — GPU VM deployment notes

## Development

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/ci_static_checks.py
```

Docker images: `Dockerfile.cpu` (default) and `Dockerfile.gpu` (CUDA). Both
require `INTERNAL_JWT_SECRET` (>= 32 bytes), `MONGODB_URI` when
`STORAGE_MODE=mongodb`, and an explicit `INFERENCE_DEVICE` (`cpu` or `cuda`).
