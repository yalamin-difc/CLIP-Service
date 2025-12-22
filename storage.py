import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _json_loads(s: Optional[str], default: Any) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


class SqliteStore:
    """
    Lightweight persistence layer for:
      - items (image embedding + OCR text + barcode values)
      - audit logs (ocr/barcode/match/release events)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                  id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  description TEXT,
                  status TEXT NOT NULL,
                  clip_embedding_json TEXT NOT NULL,
                  ocr_text TEXT,
                  ocr_words_json TEXT,
                  barcodes_json TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  released_at TEXT
                )
                """.strip()
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                  id TEXT PRIMARY KEY,
                  ts TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  item_id TEXT,
                  request_id TEXT,
                  payload_json TEXT NOT NULL
                )
                """.strip()
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_logs(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_item ON audit_logs(item_id)")

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
        log_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs(id, ts, event_type, item_id, request_id, payload_json)
                VALUES(?, ?, ?, ?, ?, ?)
                """.strip(),
                (
                    log_id,
                    ts or _utc_now_iso(),
                    event_type,
                    item_id,
                    request_id,
                    _json_dumps(payload),
                ),
            )
        return log_id

    def list_audit_logs(
        self,
        *,
        item_id: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        with self._conn() as conn:
            if item_id:
                rows = conn.execute(
                    """
                    SELECT id, ts, event_type, item_id, request_id, payload_json
                    FROM audit_logs
                    WHERE item_id = ?
                    ORDER BY ts DESC
                    LIMIT ? OFFSET ?
                    """.strip(),
                    (item_id, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, ts, event_type, item_id, request_id, payload_json
                    FROM audit_logs
                    ORDER BY ts DESC
                    LIMIT ? OFFSET ?
                    """.strip(),
                    (limit, offset),
                ).fetchall()
        return [
            {
                "id": r["id"],
                "ts": r["ts"],
                "eventType": r["event_type"],
                "itemId": r["item_id"],
                "requestId": r["request_id"],
                "payload": _json_loads(r["payload_json"], default={}),
            }
            for r in rows
        ]

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
        item_id = str(uuid.uuid4())
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO items(
                  id, name, description, status, clip_embedding_json,
                  ocr_text, ocr_words_json, barcodes_json,
                  created_at, updated_at, released_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """.strip(),
                (
                    item_id,
                    name,
                    description,
                    status,
                    _json_dumps(clip_embedding),
                    ocr_text,
                    _json_dumps(ocr_words) if ocr_words is not None else None,
                    _json_dumps(barcodes) if barcodes is not None else None,
                    now,
                    now,
                    None,
                ),
            )
        return self.get_item(item_id)

    def update_item_signals(
        self,
        item_id: str,
        *,
        ocr_text: Optional[str] = None,
        ocr_words: Optional[List[Dict[str, Any]]] = None,
        barcodes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        now = _utc_now_iso()
        fields: List[Tuple[str, Any]] = []
        if ocr_text is not None:
            fields.append(("ocr_text", ocr_text))
        if ocr_words is not None:
            fields.append(("ocr_words_json", _json_dumps(ocr_words)))
        if barcodes is not None:
            fields.append(("barcodes_json", _json_dumps(barcodes)))
        if not fields:
            return self.get_item(item_id)
        set_sql = ", ".join([f"{k}=?" for k, _ in fields] + ["updated_at=?"])
        params = [v for _, v in fields] + [now, item_id]
        with self._conn() as conn:
            cur = conn.execute(f"UPDATE items SET {set_sql} WHERE id=?", params)
            if cur.rowcount == 0:
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
        released_at = now if released else None
        with self._conn() as conn:
            cur = conn.execute(
                """
                UPDATE items
                SET status=?, updated_at=?, released_at=COALESCE(?, released_at)
                WHERE id=?
                """.strip(),
                (status, now, released_at, item_id),
            )
            if cur.rowcount == 0:
                raise KeyError(item_id)
        return self.get_item(item_id)

    def get_item(self, item_id: str) -> Dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        if row is None:
            raise KeyError(item_id)
        return self._row_to_item(row)

    def list_items(self, *, status: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 5000))
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM items WHERE status=? ORDER BY updated_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM items ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_item(r) for r in rows]

    def list_item_embeddings(self, *, status: Optional[str] = "released", limit: int = 5000) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 5000))
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT id, name, description, status, clip_embedding_json, ocr_text, barcodes_json, updated_at
                    FROM items
                    WHERE status=?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """.strip(),
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, name, description, status, clip_embedding_json, ocr_text, barcodes_json, updated_at
                    FROM items
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """.strip(),
                    (limit,),
                ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "description": r["description"],
                    "status": r["status"],
                    "embedding": _json_loads(r["clip_embedding_json"], default=[]),
                    "ocrText": r["ocr_text"],
                    "barcodes": _json_loads(r["barcodes_json"], default=[]),
                }
            )
        return out

    def _row_to_item(self, r: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": r["id"],
            "name": r["name"],
            "description": r["description"],
            "status": r["status"],
            "embedding": _json_loads(r["clip_embedding_json"], default=[]),
            "ocrText": r["ocr_text"],
            "ocrWords": _json_loads(r["ocr_words_json"], default=[]),
            "barcodes": _json_loads(r["barcodes_json"], default=[]),
            "createdAt": r["created_at"],
            "updatedAt": r["updated_at"],
            "releasedAt": r["released_at"],
        }

