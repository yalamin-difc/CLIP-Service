#!/usr/bin/env python3
"""P13 section 12: administrative batch re-embedding CLI for one engine.

Connects directly to the configured MongoDB backend (the same store the
service uses) rather than driving the HTTP API one item at a time, since
this is meant for large corpus migrations. It reuses the exact same
embedding_engines package the running service uses for inference, so a
re-embed produced here is byte-for-byte what the service itself would
have produced through POST /v2/items/{id}/embeddings/{engine}.

IMPORTANT: the CLIP service's own database never stores raw image bytes
(only encoded vectors and lightweight image metadata -- see `image_info`
in app.py), by design -- images live wherever the calling backend stores
them. So re-embedding existing corpus items requires the operator to
supply those images from that external source, via --manifest (a JSON
file mapping itemId -> local image path) and/or --image-dir (a directory
of files named "<itemId>.<ext>"). Items with no available image source
are reported under `skippedNoImageSource`, never silently dropped.

Restart/resume: items are scanned in a stable ascending `id` order.
--resume-after <item_id> continues a previously interrupted run without
any server-side bookkeeping -- record the report's `lastItemId` and pass
it back on the next invocation.

Example:
    python scripts/reembed_model.py --engine siglip2_v1 --tenant acme \\
        --site downtown --batch-size 25 --manifest ./images.json

    # Preview only, no writes:
    python scripts/reembed_model.py --engine siglip2_v1 --tenant acme --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embedding_engines.base import EngineUnavailableError  # noqa: E402
from embedding_engines.registry import UnknownEngineError, get_engine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reembed_model")

PROGRESS_LOG_INTERVAL = 50


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def needs_reembedding(
    item: Dict[str, Any],
    engine_id: str,
    model_id: Optional[str],
    model_revision: Optional[str],
    expected_dimension: int,
) -> bool:
    """True when `item` has no current embedding for `engine_id` matching
    the engine's live model id/revision/dimension -- i.e. it's missing
    entirely, or was produced by a stale model. Only ever inspects
    `engine_id`'s own block, so a stale clip_v1 revision never triggers a
    siglip2_v1 re-embed or vice versa (section 12)."""
    block = (item.get("embeddings") or {}).get(engine_id)
    if not isinstance(block, dict) or not block.get("vector"):
        return True
    if model_id is not None and block.get("modelId") != model_id:
        return True
    if model_revision is not None and block.get("modelRevision") != model_revision:
        return True
    if block.get("embeddingDimension") != expected_dimension:
        return True
    return False


def apply_embedding(
    item: Dict[str, Any],
    engine_id: str,
    vector: list,
    model_id: Optional[str],
    model_revision: Optional[str],
    preprocessing_version: Optional[str],
) -> None:
    """Mutates `item` in place with a fresh embeddings.<engine_id> block.
    For clip_v1 only, also mirrors into the legacy embedding/clipEmbedding
    fields (same vector space) -- never for any other engine (section 8)."""
    now = now_iso()
    embeddings = dict(item.get("embeddings") or {})
    existing_block = embeddings.get(engine_id) or {}
    embeddings[engine_id] = {
        "vector": vector,
        "modelId": model_id,
        "modelRevision": model_revision,
        "embeddingDimension": len(vector),
        "preprocessingVersion": preprocessing_version,
        "embeddingModality": "image",
        "createdAt": existing_block.get("createdAt") or now,
        "updatedAt": now,
    }
    item["embeddings"] = embeddings
    if engine_id == "clip_v1":
        item["embedding"] = vector
        item["clipEmbedding"] = vector
        item["embeddingDimension"] = len(vector)
        item["embeddingModality"] = "image"
    item["updatedAt"] = now


def iter_candidate_items(collection, tenant: Optional[str], site: Optional[str], resume_after: Optional[str]):
    query: Dict[str, Any] = {}
    if tenant:
        query["tenantId"] = tenant
    if site:
        query["siteId"] = site
    if resume_after:
        query["id"] = {"$gt": resume_after}
    return collection.find(query).sort("id", 1)


def load_image_manifest(manifest_path: Optional[str], image_dir: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if manifest_path:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            mapping.update(json.load(handle))
    if image_dir:
        for path in sorted(Path(image_dir).glob("*")):
            if path.is_file():
                mapping.setdefault(path.stem, str(path))
    return mapping


def run(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        engine = get_engine(args.engine)
    except UnknownEngineError as exc:
        raise SystemExit(f"Unknown engine '{args.engine}': {exc}") from exc
    if not args.dry_run and not engine.is_enabled():
        raise SystemExit(
            f"Engine '{args.engine}' is disabled by configuration -- set the matching *_ENABLED "
            "environment variable before running a real (non-dry-run) re-embed."
        )

    from pymongo import MongoClient  # noqa: E402
    from PIL import Image  # noqa: E402

    mongodb_uri = args.mongodb_uri or os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
    if not mongodb_uri:
        raise SystemExit("MONGODB_URI is required (pass --mongodb-uri or set the environment variable).")
    database_name = args.mongodb_db or os.environ.get("MONGODB_DB", "clip_service")
    collection_name = args.mongodb_collection or os.environ.get("MONGODB_COLLECTION", "items")
    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=2000)
    collection = client[database_name][collection_name]

    image_map = load_image_manifest(args.manifest, args.image_dir)

    report: Dict[str, Any] = {
        "engine": args.engine,
        "dryRun": args.dry_run,
        "tenant": args.tenant,
        "site": args.site,
        "startedAt": now_iso(),
        "scanned": 0,
        "needingReembedding": 0,
        "reembedded": 0,
        "skippedCurrent": 0,
        "skippedNoImageSource": 0,
        "failed": 0,
        "lastItemId": None,
        "errors": [],
    }

    provenance = None
    if not args.force:
        provenance = engine.provenance()

    for item in iter_candidate_items(collection, args.tenant, args.site, args.resume_after):
        if args.batch_size is not None and report["scanned"] >= args.batch_size:
            break
        report["scanned"] += 1
        report["lastItemId"] = item.get("id")

        stale = True
        if not args.force:
            stale = needs_reembedding(
                item, args.engine, provenance.model_id, provenance.model_revision, engine.expected_dimension
            )
        if not stale:
            report["skippedCurrent"] += 1
            continue
        report["needingReembedding"] += 1

        if args.dry_run:
            continue

        image_path = image_map.get(item.get("id"))
        if not image_path:
            report["skippedNoImageSource"] += 1
            continue
        try:
            with Image.open(image_path) as raw_image:
                pil_image = raw_image.convert("RGB")
                vector = engine.encode_image(pil_image)
            current_provenance = engine.provenance()
            apply_embedding(
                item,
                args.engine,
                vector,
                current_provenance.model_id,
                current_provenance.model_revision,
                current_provenance.preprocessing_version,
            )
            update_fields: Dict[str, Any] = {"embeddings": item["embeddings"], "updatedAt": item["updatedAt"]}
            if args.engine == "clip_v1":
                update_fields.update(
                    {
                        "embedding": item["embedding"],
                        "clipEmbedding": item["clipEmbedding"],
                        "embeddingDimension": item["embeddingDimension"],
                        "embeddingModality": item["embeddingModality"],
                    }
                )
            collection.update_one(
                {"id": item["id"], "tenantId": item["tenantId"], "siteId": item["siteId"]}, {"$set": update_fields}
            )
            report["reembedded"] += 1
        except EngineUnavailableError as exc:
            report["failed"] += 1
            report["errors"].append({"itemId": item.get("id"), "error": type(exc).__name__, "category": exc.error_category})
            logger.warning("Failed to re-embed item %s: %s", item.get("id"), exc)
        except (OSError, ValueError) as exc:
            report["failed"] += 1
            report["errors"].append({"itemId": item.get("id"), "error": type(exc).__name__})
            logger.warning("Failed to load image for item %s: %s", item.get("id"), exc)

        if report["scanned"] % PROGRESS_LOG_INTERVAL == 0:
            logger.info("Progress: %s", json.dumps(report))

    report["finishedAt"] = now_iso()
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P13: batch re-embed corpus items for a given engine.")
    parser.add_argument("--engine", required=True, choices=["clip_v1", "siglip2_v1"])
    parser.add_argument("--tenant", default=None, help="Restrict to this tenantId.")
    parser.add_argument("--site", default=None, help="Restrict to this siteId.")
    parser.add_argument("--batch-size", type=int, default=None, help="Maximum number of items to scan in this run.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing anything.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed every scanned item, even one whose embedding already matches the engine's current model/revision/dimension.",
    )
    parser.add_argument("--resume-after", default=None, help="Resume a prior run: only scan items with id > this value.")
    parser.add_argument("--manifest", default=None, help="JSON file mapping itemId -> local image file path.")
    parser.add_argument(
        "--image-dir", default=None, help="Directory of images named <itemId>.<ext> (used if --manifest is absent or incomplete)."
    )
    parser.add_argument("--mongodb-uri", default=None)
    parser.add_argument("--mongodb-db", default=None)
    parser.add_argument("--mongodb-collection", default=None)
    parser.add_argument("--report-file", default=None, help="Write the structured JSON result report to this path.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run(args)
    output = json.dumps(report, indent=2)
    print(output)
    if args.report_file:
        Path(args.report_file).write_text(output, encoding="utf-8")
    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
