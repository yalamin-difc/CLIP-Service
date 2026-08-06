from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import MongoClient


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _coerce_object_id(value: str) -> ObjectId:
    try:
        return ObjectId(str(value))
    except Exception as e:
        raise KeyError(value) from e


def _oid_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return str(v)
    return str(v)


class MongoStore:
    """
    MongoDB (Atlas) persistence layer for:
      - items
      - audit logs

    MongoDB is the single source of truth (no SQLite fallback).
    """

    def __init__(
        self,
        mongo_uri: str,
        *,
        db_name: str = "clip_service",
        items_collection: str = "items",
        audit_collection: str = "audit_logs",
    ):
        if not (mongo_uri or "").strip():
            raise RuntimeError("Missing MONGODB_URI")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.items = self.db[items_collection]
        self.audit_logs = self.db[audit_collection]

        # Ensure helpful indexes (idempotent)
        self.items.create_index([("status", 1), ("updatedAt", -1)])
        self.items.create_index([("releasedAt", -1)])
        self.audit_logs.create_index([("ts", -1)])
        self.audit_logs.create_index([("itemId", 1), ("ts", -1)])

    # ----------------------------
    # Audit logging
    # ----------------------------
    def add_audit_log(
        self,
        *,
        event_type: str,
        payload: Dict[str, Any],
        item_id: Optional[str] = None,
        request_id: Optional[str] = None,
        ts: Optional[str] = None,
    ) -> str:
        doc = {
            "ts": ts or _utc_now_iso(),
            "eventType": event_type,
            "itemId": item_id,
            "requestId": request_id,
            "payload": payload or {},
        }
        res = self.audit_logs.insert_one(doc)
        return str(res.inserted_id)

    def list_audit_logs(
        self,
        *,
        item_id: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        q: Dict[str, Any] = {}
        if item_id:
            q["itemId"] = item_id
        cur = self.audit_logs.find(q).sort("ts", -1).skip(offset).limit(limit)
        out: List[Dict[str, Any]] = []
        for d in cur:
            out.append(
                {
                    "id": _oid_str(d.get("_id")),
                    "ts": d.get("ts"),
                    "eventType": d.get("eventType"),
                    "itemId": d.get("itemId"),
                    "requestId": d.get("requestId"),
                    "payload": d.get("payload") or {},
                }
            )
        return out

    # ----------------------------
    # Items
    # ----------------------------
    def create_item(
        self,
        *,
        name: str,
        description: Optional[str],
        clip_embedding: List[float],
        ocr_text: Optional[str],
        ocr_words: Optional[List[Dict[str, Any]]],
        barcodes: Optional[List[Dict[str, Any]]],
        status: str = "draft",
    ) -> Dict[str, Any]:
        now = _utc_now_iso()
        doc: Dict[str, Any] = {
            "name": name,
            "description": description or "",
            "status": status,
            "clipEmbedding": clip_embedding,
            "ocr": {"fullText": ocr_text or "", "words": (ocr_words or []), "meta": {}},
            "barcodes": barcodes or [],
            "releasedAt": None,
            "createdAt": now,
            "updatedAt": now,
        }
        res = self.items.insert_one(doc)
        return self.get_item(str(res.inserted_id))

    def update_item_signals(
        self,
        item_id: str,
        *,
        ocr_text: Optional[str] = None,
        ocr_words: Optional[List[Dict[str, Any]]] = None,
        barcodes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        update: Dict[str, Any] = {"updatedAt": _utc_now_iso()}
        if ocr_text is not None:
            update["ocr.fullText"] = ocr_text
        if ocr_words is not None:
            update["ocr.words"] = ocr_words
        if barcodes is not None:
            update["barcodes"] = barcodes
        if len(update) == 1:
            return self.get_item(item_id)
        res = self.items.update_one({"_id": _coerce_object_id(item_id)}, {"$set": update})
        if res.matched_count == 0:
            raise KeyError(item_id)
        return self.get_item(item_id)

    def set_item_status(
        self,
        item_id: str,
        *,
        status: str,
        released: bool = False,
    ) -> Dict[str, Any]:
        now = _utc_now_iso()
        update: Dict[str, Any] = {"status": status, "updatedAt": now}
        if released:
            # Only set releasedAt if not already set.
            update["releasedAt"] = now
        res = self.items.update_one({"_id": _coerce_object_id(item_id)}, {"$set": update})
        if res.matched_count == 0:
            raise KeyError(item_id)
        return self.get_item(item_id)

    def get_item(self, item_id: str) -> Dict[str, Any]:
        d = self.items.find_one({"_id": _coerce_object_id(item_id)})
        if d is None:
            raise KeyError(item_id)
        return self._doc_to_item(d)

    def list_items(self, *, status: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 5000))
        q: Dict[str, Any] = {}
        if status:
            q["status"] = status
        cur = self.items.find(q).sort("updatedAt", -1).limit(limit)
        return [self._doc_to_item(d) for d in cur]

    def list_item_embeddings(self, *, status: Optional[str] = "released", limit: int = 5000) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 5000))
        q: Dict[str, Any] = {}
        if status:
            q["status"] = status
        cur = (
            self.items.find(
                q,
                {
                    "name": 1,
                    "description": 1,
                    "status": 1,
                    "clipEmbedding": 1,
                    "ocr.fullText": 1,
                    "barcodes": 1,
                    "updatedAt": 1,
                },
            )
            .sort("updatedAt", -1)
            .limit(limit)
        )
        out: List[Dict[str, Any]] = []
        for d in cur:
            emb = d.get("clipEmbedding") or []
            ocr = d.get("ocr") or {}
            out.append(
                {
                    "id": _oid_str(d.get("_id")),
                    "name": d.get("name"),
                    "description": d.get("description"),
                    "status": d.get("status"),
                    # Back-compat: internal matching uses `embedding`.
                    "embedding": emb,
                    "clipEmbedding": emb,
                    "ocrText": ocr.get("fullText"),
                    "barcodes": d.get("barcodes") or [],
                }
            )
        return out

    def _doc_to_item(self, d: Dict[str, Any]) -> Dict[str, Any]:
        emb = d.get("clipEmbedding") or []
        ocr = d.get("ocr") or {}
        ocr_text = ocr.get("fullText")
        ocr_words = ocr.get("words") or []
        return {
            "id": _oid_str(d.get("_id")),
            "name": d.get("name"),
            "description": d.get("description"),
            "status": d.get("status"),
            "embedding": emb,
            "clipEmbedding": emb,
            "ocr": ocr,
            "ocrText": ocr_text,
            "ocrWords": ocr_words,
            "barcodes": d.get("barcodes") or [],
            "createdAt": d.get("createdAt"),
            "updatedAt": d.get("updatedAt"),
            "releasedAt": d.get("releasedAt"),
        }

