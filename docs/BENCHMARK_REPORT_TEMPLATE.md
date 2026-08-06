# Festival benchmark report

## Build and environment

- Git commit:
- Container target: CPU / GPU
- Host CPU and memory:
- GPU and driver:
- MongoDB topology and region:
- Replica count:
- Model ID: `openai/clip-vit-base-patch32`
- Model revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`
- Embedding dimension: 512

## Commands

```bash
python3 scripts/benchmark.py --candidates 1000 --iterations 100 --output benchmark-1000.json
python3 scripts/benchmark.py --candidates 10000 --iterations 100 --output benchmark-10000.json
python3 scripts/benchmark.py --candidates 50000 --iterations 100 --output benchmark-50000.json
```

## Results

Record mean and p95 for:

- embedding latency;
- candidate retrieval latency;
- ranking latency;
- total response time.

Also record:

- process peak memory;
- peak GPU memory when available;
- throughput;
- error rate.

## Acceptance observations

- Warmup duration:
- Readiness transition:
- OCR English/Arabic result:
- Tenant-isolation validation:
- Saturation point:
- Errors or throttling:
- Operational recommendation:

The current repository fetches up to 50,000 compatible released candidates and
performs an exhaustive cosine scan. This is O(n × embedding dimension) per match.
The benchmark does not represent MongoDB network latency unless supplemented with
deployment-level HTTP load testing.
