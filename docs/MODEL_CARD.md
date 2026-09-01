# CLIP-Service model card

## Model

- Model ID: `openai/clip-vit-base-patch32`
- Model revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`
- Embedding dimension: 512
- Similarity: normalized cosine similarity
- Intended data: synthetic Dubai AI Festival lost-and-found demonstration data

## Intended use

The service ranks already released corpus items within one authenticated tenant and
an allowed set of sites. It is a retrieval aid, not an automated identity,
ownership, safety, or enforcement decision system. A low score or low margin
produces an explicit no-match result.

## Runtime

`INFERENCE_DEVICE` must be explicitly `cpu` or `cuda`. CUDA startup fails when
PyTorch cannot access a CUDA device. Startup performs an image warmup, discovers
the effective embedding dimension, and keeps readiness false until it succeeds.

## Limitations

- CLIP may perform poorly on fine-grained text, serial numbers, culturally specific
  objects, and image domains unlike its training data.
- OCR and barcode signals are explanatory aids; they do not alter the current CLIP
  cosine ranking.
- Retrieval is an exhaustive scan, so latency and memory rise linearly with corpus
  size.
- Festival data must remain synthetic. This model card does not authorize use with
  personal, biometric, or sensitive operational data.

## Evaluation

Run `scripts/benchmark.py` at 1,000, 10,000, and 50,000 candidates and record the
deployment-specific results in `docs/BENCHMARK_REPORT_TEMPLATE.md`.
