# P13 benchmarks

`scripts/benchmark_models.py` measures, per engine: cold-start latency, warm image-encoding
latency, warm text-encoding latency (English/Arabic/mixed), a warm match latency against a
500-item synthetic in-memory corpus, process RSS before/after load, and CPU time during the
benchmark. It never fabricates a number — an engine it could not actually load and run is
reported with `"status": "skipped"` and a reason, not a placeholder value.

## What is in this directory

- `sandbox_dry_run.json` — the **actual, real output** of running
  `python scripts/benchmark_models.py --iterations 5` in the sandbox this feature was developed
  in. Both engines report `"status": "skipped"`:
  - `clip_v1`: `transformers`/`torch` could not be installed in this sandbox — the environment's
    egress policy blocks `download.pytorch.org` (`requirements.txt` pins
    `torch==2.6.0+cpu` from that index), and no cached wheel was available. This is a sandbox
    constraint, not a code defect: `tests/test_engines.py`/`tests/test_v2_api.py` exercise the
    same code paths against mocked models instead (see the main implementation doc's "Known
    limitations" section), and the *legacy* CLIP test suite (`tests/test_app.py`, 135 tests) has
    always used the same mocking approach for exactly this reason.
  - `siglip2_v1`: disabled by default (`SIGLIP2_ENABLED=false`), and even if enabled, this
    sandbox has no outbound access to `huggingface.co` to download the ~3.5GB model weights
    (confirmed: a direct `curl` to the HF API returned a 403 from the egress policy).

  This file is included as evidence the benchmark harness itself runs correctly end-to-end (it
  reaches the model-load call, fails honestly, and reports why) — **it is not a performance
  result**, and must not be read as one.

- A run of `scripts/benchmark_models.py` against a fake in-process engine (not committed, since
  it isn't a real measurement either) confirmed the full success path — timing collection,
  percentile stats, the synthetic-corpus match measurement, and memory/CPU sampling — all
  execute without error and produce sane, monotonic numbers.

## Producing a real report

Real cold-start/warm-latency/memory numbers require the target Azure VM
(`Standard_D8as_v5`, 8 vCPU, 32GiB RAM, no GPU, Ubuntu 24.04 LTS) or an equivalent
network-connected environment with `requirements.txt` installed and (for `siglip2_v1`) a
resolved, pinned `SIGLIP2_MODEL_REVISION` (see the main implementation doc):

```bash
pip install -r requirements.txt
export INFERENCE_DEVICE=cpu
export SIGLIP2_ENABLED=true
export SIGLIP2_DEVICE=cpu
export SIGLIP2_MODEL_REVISION=<real pinned commit SHA>

python scripts/benchmark_models.py --iterations 10 --output docs/benchmarks/report_$(date +%Y%m%d).json
```

Commit the resulting `report_YYYYMMDD.json` here once produced. Do not hand-edit its numbers.
