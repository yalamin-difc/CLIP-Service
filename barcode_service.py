from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image

try:
    import zxingcpp
except Exception:  # pragma: no cover
    zxingcpp = None


def scan_barcodes(image: Image.Image) -> Dict[str, Any]:
    """
    ZXing-backed barcode scan (via `zxing-cpp` python bindings).
    Returns:
      - barcodes: list of {text, format, contentType, position}
      - meta
    """
    if zxingcpp is None:
        raise RuntimeError("zxingcpp is not installed")

    img = image.convert("RGB")
    try:
        results = zxingcpp.read_barcodes(img)
    except Exception:
        # Some versions prefer a numpy array; fall back if needed.
        import numpy as np

        results = zxingcpp.read_barcodes(np.array(img))

    out: List[Dict[str, Any]] = []
    for r in results or []:
        # `r` is a Barcode object; attributes differ slightly by version.
        out.append(
            {
                "text": getattr(r, "text", None) or getattr(r, "data", None) or "",
                "format": str(getattr(r, "format", "") or ""),
                "contentType": str(getattr(r, "content_type", "") or getattr(r, "contentType", "") or ""),
                "position": getattr(r, "position", None) or getattr(r, "points", None),
            }
        )

    # Deduplicate on (text,format)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for b in out:
        key = (b.get("text", ""), b.get("format", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(b)

    return {"barcodes": deduped, "meta": {"engine": "zxing-cpp"}}

