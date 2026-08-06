# Dependency manifest

Runtime Python dependencies are fully version-pinned in:

- `requirements.txt` for CPU/PyTorch CPU wheels;
- `requirements-gpu.txt` for CUDA 12.4-compatible PyTorch wheels.

The OpenAI CLIP source dependency is pinned to commit
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`.

System dependencies in both images:

- Tesseract OCR;
- English and Arabic Tesseract language data;
- Git and CA certificates;
- CUDA 12.4.1 and cuDNN runtime in the GPU target.

Container targets:

```bash
docker build -f Dockerfile.cpu -t clip-service:festival-cpu .
docker build -f Dockerfile.gpu -t clip-service:festival-gpu .
```

The machine-readable CycloneDX SBOM is `sbom.cdx.json`. Regenerate it after every
dependency change:

```bash
cyclonedx-py requirements requirements.txt --output-format JSON \
  --output-file sbom.cdx.json
```
