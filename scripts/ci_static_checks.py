#!/usr/bin/env python3
"""Repo-supported security/static checks for CLIP-Service CI.

These checks encode controls the repository already documents: pinned
container digests, the committed CycloneDX SBOM, and the rule that raw
embeddings never appear in committed test/response fixtures.
"""
from __future__ import annotations

import compileall
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FORBIDDEN_VECTOR_KEYS = {"embedding", "clipEmbedding"}
DOCKERFILES = ("Dockerfile.cpu", "Dockerfile.gpu")
WORKFLOW = ROOT / ".github" / "workflows" / "clip-ci.yml"
SBOM = ROOT / "sbom.cdx.json"
SECRET_PATTERN = re.compile(
    r"secrets\.(INTERNAL_JWT_SECRET|MONGODB_URI|MONGO_URI)\b",
    re.IGNORECASE,
)


def fail(message: str) -> None:
    print(f"STATIC_CHECK_FAIL {message}", file=sys.stderr)
    raise SystemExit(1)


def check_compileall() -> None:
    ok = compileall.compile_dir(str(ROOT), quiet=1, force=False, rx=re.compile(r"/\.git/"))
    if not ok:
        fail("python compileall reported syntax errors")


def check_sbom() -> None:
    if not SBOM.is_file():
        fail(f"missing CycloneDX SBOM at {SBOM.relative_to(ROOT)}")
    payload = json.loads(SBOM.read_text())
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        fail("sbom.cdx.json does not contain a components list")
    names = {item.get("name") for item in components if isinstance(item, dict)}
    for required in ("fastapi", "torch", "transformers", "zxing-cpp", "pytesseract"):
        if required not in names:
            fail(f"sbom.cdx.json is missing pinned component {required}")


def check_dockerfile_digests() -> None:
    digest_re = re.compile(r"^FROM\s+\S+@sha256:[0-9a-f]{64}\s*$")
    for name in DOCKERFILES:
        path = ROOT / name
        if not path.is_file():
            fail(f"missing {name}")
        first_from = next((line.strip() for line in path.read_text().splitlines() if line.startswith("FROM ")), "")
        if not digest_re.match(first_from):
            fail(f"{name} base image is not pinned to an immutable sha256 digest")


def check_workflow_has_no_production_secrets() -> None:
    if not WORKFLOW.is_file():
        fail("missing .github/workflows/clip-ci.yml")
    text = WORKFLOW.read_text()
    match = SECRET_PATTERN.search(text)
    if match:
        fail(f"CI workflow references production secret {match.group(0)}")
    if "token_urlsafe(" not in text:
        fail("CI workflow does not generate an ephemeral INTERNAL_JWT_SECRET")


def _contains_raw_vector(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_VECTOR_KEYS and isinstance(nested, list) and nested and all(
                isinstance(item, (int, float)) for item in nested
            ):
                return True
            if _contains_raw_vector(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_raw_vector(item) for item in value)
    return False


def check_no_raw_embedding_fixtures() -> None:
    scanned = 0
    for path in (ROOT / "tests").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        scanned += 1
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            fail(f"fixture {path.relative_to(ROOT)} is not valid JSON: {exc}")
        if _contains_raw_vector(payload):
            fail(f"raw embedding vector found in {path.relative_to(ROOT)}")
    print(f"STATIC_CHECK embedding fixtures scanned={scanned}")


def main() -> None:
    check_compileall()
    check_sbom()
    check_dockerfile_digests()
    check_workflow_has_no_production_secrets()
    check_no_raw_embedding_fixtures()
    print("STATIC_CHECK_OK compileall sbom dockerfile-digests ephemeral-jwt embedding-fixtures")


if __name__ == "__main__":
    main()
