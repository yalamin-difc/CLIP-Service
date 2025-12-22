from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def softmax_confidences(scores: List[float], *, temperature: float = 0.07) -> List[float]:
    """
    Convert similarity scores to probabilities over the Top-K.
    Temperature lower => sharper distribution.
    """
    if not scores:
        return []
    t = float(temperature) if temperature and temperature > 0 else 0.07
    scaled = [s / t for s in scores]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def _tokens(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return _TOKEN_RE.findall(s.lower())


def token_overlap(a: Optional[str], b: Optional[str]) -> Dict[str, Any]:
    ta = set(_tokens(a))
    tb = set(_tokens(b))
    if not ta or not tb:
        return {"jaccard": 0.0, "overlap": [], "aCount": len(ta), "bCount": len(tb)}
    inter = sorted(list(ta.intersection(tb)))
    union = ta.union(tb)
    j = float(len(inter)) / float(len(union) or 1)
    # keep small / explainable
    return {"jaccard": j, "overlap": inter[:25], "aCount": len(ta), "bCount": len(tb)}


def barcode_match_signal(
    *,
    query_barcodes: Optional[List[Dict[str, Any]]],
    item_barcodes: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    q = {(b.get("text") or "").strip() for b in (query_barcodes or []) if (b.get("text") or "").strip()}
    i = {(b.get("text") or "").strip() for b in (item_barcodes or []) if (b.get("text") or "").strip()}
    if not q or not i:
        return {"matched": [], "queryCount": len(q), "itemCount": len(i)}
    matched = sorted(list(q.intersection(i)))
    return {"matched": matched[:25], "queryCount": len(q), "itemCount": len(i)}


def build_explanation(
    *,
    similarity: float,
    query_text: Optional[str],
    item_ocr_text: Optional[str],
    query_barcodes: Optional[List[Dict[str, Any]]],
    item_barcodes: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    ocr_overlap = token_overlap(query_text, item_ocr_text)
    barcode_signal = barcode_match_signal(query_barcodes=query_barcodes, item_barcodes=item_barcodes)
    return {
        "signals": {
            "clipCosine": similarity,
            "ocrTokenJaccard": ocr_overlap["jaccard"],
            "barcodeMatches": len(barcode_signal.get("matched") or []),
        },
        "details": {"ocr": ocr_overlap, "barcode": barcode_signal},
    }

