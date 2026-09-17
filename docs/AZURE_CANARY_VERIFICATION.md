# Azure Canary Verification Procedure

## What this document is

A checklist of the exact, authorized commands an operator with real access to the Azure
canary host must run to establish five facts before any SigLIP2/festival sign-off decision is
made on that canary:

1. Which application commit is actually deployed.
2. The effective SigLIP2 model ID and revision.
3. That the processor and the model received the same revision.
4. That real image and real text inference actually succeed on that deployment.
5. Evidence of the loaded artifact/snapshot revision, where available.

**This document does not perform these checks and does not report a result for any of
them.** Canary host access (SSH/`docker exec`, and network egress to `huggingface.co`) was
not available from this session. Every command below is untested against the real canary; an
operator running it and recording the real output is what constitutes verification evidence —
this document's existence is not that evidence. Do not treat any PASS/FAIL below as filled in
until an authorized operator has actually run the command and pasted its real output.

## Configured vs. resolved/loaded — read this before running anything

Several fields the service already exposes (`/health`, `/v2/models`, every `/v2/embeddings/*`
response) report **configuration**, not proof of what is actually running:

- `modelRevision` in every one of those responses is `EngineProvenance.model_revision`
  (`embedding_engines/base.py`), which for SigLIP2 is literally the `SIGLIP2_MODEL_REVISION`
  environment variable, echoed back unchanged (`embedding_engines/siglip2_engine.py`'s
  `model_revision` property; `embedding_engines/config.py`'s `SIGLIP2_MODEL_REVISION =
  os.environ.get(...)`). **An endpoint echoing `SIGLIP2_MODEL_REVISION` alone proves only that
  the container's environment variable is set to that value — it does not prove the model
  weights actually loaded came from that revision.** If `SIGLIP2_MODEL_REVISION` is left at its
  default `"main"` (a branch, not a commit — see `docs/P13_SIGLIP2_AB_IMPLEMENTATION.md`,
  "Pinning the SigLIP2 revision"), the *same* configured value can silently correspond to a
  *different* actual commit on two different days, or two different container instances that
  pulled the model at different times.
- `embeddingDimension` and `preprocessingVersion`, by contrast, are genuinely observed:
  `embeddingDimension` is `len(vector)` from a real produced embedding (`v2_router.py`), and
  `preprocessingVersion` is a fingerprint computed from the *actually loaded* processor's own
  config dict (`Siglip2Engine._compute_preprocessing_version`). These are real signals that
  *something* loaded and produced output of the expected shape, but neither one is the git
  commit hash of the model that was loaded — a different revision that happens to share the
  same image-preprocessing config and output dimension would look identical on these two
  fields alone.
- There is currently no endpoint that reports the deployed **application's** git commit.
  `serviceVersion` (`SERVICE_VERSION` env var, `app.py`) exists and appears in `/health`, but
  the canary's documented `docker run` invocation (`docs/P14_FESTIVAL_LIVE_GUI.md`, "Staging
  (the current Azure canary)") does not set it, so today it reports the literal default
  string `"dev"` regardless of which commit is actually running. Treat `serviceVersion` as
  real evidence only after an operator starts setting it explicitly (see Check 1 below).

The rule for every check below: a value the service *reports about itself from its own
configuration* is not evidence; a value obtained by inspecting the running container's actual
on-disk state, its actual process output, or a real round-trip through its API is evidence.

## Prerequisites

- SSH or equivalent shell access to the Azure canary host running the `clip-service-canary`
  container (see `docs/P14_FESTIVAL_LIVE_GUI.md` section 8 for how it is started).
- `docker exec` access into that running container.
- A valid internal JWT for the canary's `internal_auth` contract (`match:execute` and
  `health:read` actions), to call the protected `/v2/*` endpoints.
- Ideally, network egress from *some* machine (the canary host, or elsewhere) to
  `huggingface.co`, to independently resolve `google/siglip2-so400m-patch14-384`'s current
  upstream commit for comparison (the sandbox this document was written in confirmed it has
  none — see the P13 doc's "Pinning the SigLIP2 revision" section).
- A real sample image file and a short real text string to use as inference inputs — never a
  synthetic fixture from the evaluation harness (`tests/test_calibration_harness.py` and
  friends) or a `DEMO_FACADE_ENABLED` response, both of which are explicitly out of scope as
  evidence here (see "Standing gates" below).

## Check 1 — Deployed application commit

**Gap today:** the service does not self-report its git commit. `serviceVersion` defaults to
`"dev"` unless explicitly set.

```bash
# On the canary host:
docker inspect clip-service-canary --format '{{.Config.Image}}'
docker inspect clip-service-canary --format '{{.Image}}'          # resolved image digest

# If the CI/CD pipeline that built the image stamps it with an OCI label
# containing the git commit (recommended, not yet confirmed present):
docker inspect <image-ref-from-above> --format \
  '{{index .Config.Labels "org.opencontainers.image.revision"}}'
```

Cross-reference the image tag/digest printed above against the build pipeline's own record
(e.g. the GitHub Actions run that built and pushed that exact tag) to identify the git commit
that produced it. If the image carries no commit-identifying tag or label, the deployed
commit **cannot** be established from the running container alone — that is a real gap, not
something this document can paper over.

**Recommended fix (process, not code — `SERVICE_VERSION` is already a supported env var):**
set it explicitly at deploy time —

```bash
-e SERVICE_VERSION=$(git rev-parse HEAD)
```

— added to the canary's `docker run` invocation. After that change ships, verify with:

```bash
curl -s https://<canary-host>/health | python3 -c "import json,sys; print(json.load(sys.stdin)['serviceVersion'])"
```

and confirm the printed value equals the git commit the operator intended to deploy. Until
`SERVICE_VERSION` is set this way, do not treat `/health`'s `serviceVersion` field as evidence
of the deployed commit.

## Check 2 — Effective model ID and revision: configured vs. resolved

**Configured** (from the running container's environment, echoed by the service):

```bash
curl -s https://<canary-host>/v2/models \
  -H "Authorization: Bearer $INTERNAL_JWT" | python3 -m json.tool
```

Look at the `siglip2_v1` entry's `modelId`/`modelRevision`. This is `SIGLIP2_MODEL_ID`/
`SIGLIP2_MODEL_REVISION` verbatim — configuration, not proof of what loaded.

**Resolved** (what `transformers` actually downloaded and pinned locally): Hugging Face's
local cache names the snapshot directory it materializes a repo into after the *resolved*
commit SHA, even when the configured `revision` was a branch name like `"main"`, and it keeps
a `refs/<branch>` file whose content is that resolved SHA:

```bash
docker exec clip-service-canary \
  find /root/.cache/huggingface/hub -maxdepth 1 -iname '*siglip2*'

docker exec clip-service-canary \
  cat "/root/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/refs/main"

docker exec clip-service-canary \
  ls "/root/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/"
```

(Adjust the cache root if `HF_HOME`/`TRANSFORMERS_CACHE` is overridden in the container's
environment — check `docker exec clip-service-canary env | grep -i -E 'HF_HOME|TRANSFORMERS_CACHE'`
first.)

The snapshot directory name(s) printed by the last command, and the content of `refs/main`,
are the actual resolved commit SHA(s) present on disk.

**Pass criteria:**
- If `SIGLIP2_MODEL_REVISION` is pinned to a specific 40-character commit SHA (the production
  requirement — see the P13 doc), the resolved snapshot directory name **must equal that SHA
  exactly**. Any mismatch means the pin did not take effect (e.g. a stale cache populated
  before the pin was set) and must block sign-off.
- Independently resolve the same model's current upstream commit from a machine with real
  network access, using the same command the P13 doc already documents:
  ```bash
  python3 -c "from huggingface_hub import HfApi; print(HfApi().model_info('google/siglip2-so400m-patch14-384').sha)"
  ```
  and compare it to the pinned `SIGLIP2_MODEL_REVISION` — this confirms the pin is still the
  commit the operator believes it is, independent of what happens to be cached locally.

## Check 3 — Processor and model received the same revision

**Code fact:** `Siglip2Engine.load()` (`embedding_engines/siglip2_engine.py`) passes the same
`self.model_revision` property to both `AutoProcessor.from_pretrained(self.model_id,
revision=self.model_revision)` and `AutoModel.from_pretrained(self.model_id,
revision=self.model_revision)` in the same call. There is exactly one revision value in the
code path; it cannot silently diverge between processor and model unless the cache itself is
inconsistent.

**Authorized verification of the cache, not just the code:** `transformers` caches an entire
repo's snapshot (both the processor's config files and the model's weight files) together
under one snapshot directory per resolved commit. Confirm both are actually present together:

```bash
docker exec clip-service-canary \
  ls "/root/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/<resolved-sha-from-check-2>/"
```

**Pass criteria:** both a processor config file (`preprocessor_config.json` and/or
`tokenizer_config.json`) and a model artifact (`config.json` plus `model.safetensors` or
`pytorch_model.bin`) appear in that one directory. If processor and model files exist under
*two different* snapshot directories despite one configured revision string, escalate
immediately — that is a real, unexpected divergence, not something to route around.

**Corroborating evidence from the running process itself:** `Siglip2Engine.load()` emits one
structured log line per successful combined load, e.g.:

```json
{"event": "model_load", "engine": "siglip2_v1", "modelId": "google/siglip2-so400m-patch14-384",
 "modelRevision": "<configured-revision>", "preprocessingVersion": "<fingerprint>",
 "device": "cpu", "loadSeconds": 2.41}
```

Retrieve it with `docker logs clip-service-canary | grep '"event": "model_load"'` (or the
platform's centralized log store, if wired). This one log line is emitted by the single
`load()` call that loads both, so its existence corroborates a single successful combined
load — it is not, on its own, proof that the *revision* named in it is what actually got
cached (that's Check 2's job).

## Check 4 — Successful real image and text inference

**Never accept as this evidence:** any `DEMO_FACADE_ENABLED=true` response, the synthetic
calibration/evaluation harness's fixtures (Backend PR #58 / CLIP-Service PR #18's harness), or
a unit/integration test's mocked engine. All of those are deliberately isolated from real
model weights and prove nothing about the live canary.

**Authorized checks — real files, real HTTP calls, against the live canary:**

```bash
curl -s -X POST https://<canary-host>/v2/embeddings/image \
  -H "Authorization: Bearer $INTERNAL_JWT" \
  -F "file=@/path/to/a/real/sample.jpg" \
  -F "engine=siglip2_v1" | python3 -m json.tool

curl -s -X POST https://<canary-host>/v2/embeddings/text \
  -H "Authorization: Bearer $INTERNAL_JWT" \
  -F "text=navy leather wallet with a metal clasp" \
  -F "engine=siglip2_v1" | python3 -m json.tool

curl -s -X POST https://<canary-host>/v2/match \
  -H "Authorization: Bearer $INTERNAL_JWT" \
  -F "file=@/path/to/a/real/sample.jpg" \
  -F "text=navy leather wallet" \
  -F "engine=siglip2_v1" -F "topK=5" | python3 -m json.tool
```

**Pass criteria for each call:**
- HTTP 200, not `503` (`engine_disabled`/`engine_unavailable` — see
  `embedding_engines/base.py`'s `EngineUnavailableError`) and not a 4xx auth/validation error.
- `embeddingDimension` present and equal to `SIGLIP2_EXPECTED_DIMENSION` (`1152` by default —
  confirm the configured value via `/v2/models` first, since it's overridable).
- `preprocessingVersion` present and non-null. A `null` here means the processor's config
  dump failed even though loading nominally "succeeded" — worth flagging even if the call
  otherwise returns 200.
- `calibrationStatus: "uncalibrated"` is *expected*, not a defect — do not treat it as a
  failure, and do not treat its presence as license to skip calibration before any threshold-
  gated production use (see "Standing gates" below).
- Latency roughly in line with the canary's own documented CPU-inference expectation (~2–3
  seconds per query on the current `Standard_D8as_v5` host, per
  `docs/P14_FESTIVAL_LIVE_GUI.md` section 12). A suspiciously fast first real call (e.g.
  under 100ms) is a red flag that inference may not have actually executed against real
  weights — investigate before treating the call as a pass.
- Run `/v2/match` (not just the two isolated `/v2/embeddings/*` calls) at least once, to prove
  the full match pipeline executes end to end against the live SigLIP2 engine, not just that
  isolated encode calls work.

## Check 5 — Loaded artifact / snapshot revision evidence

This reuses Check 2's on-disk inspection and Check 3's log line as the retained evidence
artifact for sign-off, since the live container's cache disappears once the container is
removed or redeployed:

- Snapshot directory name(s) and `refs/main` content from Check 2 (the resolved commit SHA
  actually on disk at the moment of inspection).
- The `model_load` structured log line from Check 3 (timestamp, `modelRevision` as
  configured, `preprocessingVersion`, `device`, `loadSeconds`).

Where the platform ships container logs to a retained store (e.g. Azure Monitor / Log
Analytics, if wired for this host), capture and retain that specific `model_load` entry as
the durable audit record for this canary sign-off — the on-disk cache and `docker logs`
buffer are both lost once the container is replaced.

## Standing gates this document does not relax

These carry over unchanged from prior findings and are restated here because a canary PASS on
the checks above answers only "does the canary work as configured" — it answers none of these:

- **Do not enable `SIGLIP2_ENABLED=true` in production** based on this canary verification
  alone. This procedure verifies the canary/staging deployment; it is not a production
  readiness sign-off.
- **Do not treat the synthetic evaluation/calibration harness** (Backend PR #58 / CLIP-Service
  PR #18) **as real accuracy evidence.** It proves the harness's own scoring logic is
  internally consistent; it says nothing about SigLIP2's real-world match accuracy.
- **Do not treat the merged PII inventory/scope proposal as implemented encryption.** It is a
  proposal document, not a shipped control.
- **Do not treat PR #18's green Codacy check as resolution of PR #17's findings.** They are
  different code changes; a green check on one PR does not retroactively resolve findings
  raised against another.
- **`calibrationStatus: "uncalibrated"`** on every SigLIP2 response means exactly what it
  says: its similarity scores are retrieval signals, not ownership probabilities, and must
  never be compared directly against CLIP's calibrated thresholds (`CONF_MIN_SCORE`/
  `CONF_MIN_MARGIN`) — see `docs/P13_SIGLIP2_AB_IMPLEMENTATION.md`.

## Sign-off record (template — fill in only with real, executed command output)

| # | Check | Command run | Real output (paste) | Verdict |
|---|---|---|---|---|
| 1 | Deployed application commit | | | NOT TESTED |
| 2 | Model ID/revision — configured | | | NOT TESTED |
| 2 | Model ID/revision — resolved (cache) | | | NOT TESTED |
| 2 | Resolved matches upstream HF `sha` | | | NOT TESTED |
| 3 | Processor + model share one snapshot dir | | | NOT TESTED |
| 3 | `model_load` log line captured | | | NOT TESTED |
| 4 | Real image inference (`/v2/embeddings/image`) | | | NOT TESTED |
| 4 | Real text inference (`/v2/embeddings/text`) | | | NOT TESTED |
| 4 | Real end-to-end match (`/v2/match`) | | | NOT TESTED |
| 5 | Loaded snapshot/log evidence retained | | | NOT TESTED |

A row stays `NOT TESTED` until an operator with real canary access has actually run its
command and pasted the actual output — never mark a row PASS from this document alone.
