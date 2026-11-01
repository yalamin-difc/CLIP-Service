# Dubai Mall Venue Pilot — Readiness Assessment (pointer)

The full cross-repository assessment pack (documents 01–11) lives in the Backend repository:
`yalamin-difc/Backend` → `docs/dubai-mall-pilot/` (branch `claude/admiring-bell-gwoy00` until merged). The AI detail is in `05_AI_Model_Readiness.md`.

## CLIP-Service highlights (assessed at `1502c3f`)
- unittest: 339 OK, 7 skipped, run locally on 2026-11-01 without torch, so inference was mocked. CLIP CI (Docker CPU smoke) is green on main.
- **LIVE:** pinned `openai/clip-vit-base-patch32@3d74acf9…`, internal-JWT-protected `/items`, `/match` and `/analyze-image`, tenant-filtered vector search, OCR eng+ara, and barcode reading.
- **Unproven:** real-world retrieval accuracy (`docs/eval/retrieval_quality_report.json` shows `clip_v1` *skipped*; every eval set is synthetic), latency on the target host, and Arabic free-text matching (the CLIP text encoder is English-trained).
- **SigLIP2:** EXPERIMENTAL and off by default. No `docs/eval/calibration.json`, so it reports `uncalibrated`.
- **Cleanup:** the demo facade (`demo_router.py`, `demo_config.py`) and the HTML console at `/` and `/ui` must stay disabled or flag-gated in the pilot. The `INTERNAL_JWT_ISSUER` default `dubai-festival-backend` must be renamed together with the Backend. Archive the festival runbooks.
- **Pilot recommendation:** CLIP B/32 on a UAE-region CPU VM; run AI-01 (latency) and AI-02 (real labelled baseline) before quoting any accuracy.
