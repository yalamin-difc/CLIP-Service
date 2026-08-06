# Deploy CLIP-Service on an NVIDIA VM

Build the pinned CUDA target:

```bash
docker build -f Dockerfile.gpu -t clip-service:festival-gpu .
```

Required runtime configuration:

```bash
ENV=festival
STORAGE_MODE=mongodb
MONGODB_URI='mongodb+srv://...'
INTERNAL_JWT_SECRET='<at-least-32-random-bytes>'
INFERENCE_DEVICE=cuda
OCR_ENABLED=true
OCR_LANGUAGES=eng+ara
```

Run with an NVIDIA container runtime and inject secrets from the platform secret
manager:

```bash
docker run --rm --gpus all --env-file /secure/clip-service.env \
  -p 8080:8080 clip-service:festival-gpu
```

Public probes:

```bash
curl --fail http://127.0.0.1:8080/health/live
curl --fail http://127.0.0.1:8080/health/ready
```

Detailed health and metrics require signed internal JWTs carrying `health:read`
or `metrics:read`. Readiness remains false until MongoDB, OCR, model loading, the
512-dimensional embedding contract, explicit CUDA execution, and model warmup are
all verified. Startup fails rather than falling back to CPU or in-memory storage.

Operational and rollback requirements are in `CLIP_FESTIVAL_ACCEPTANCE.md`.
