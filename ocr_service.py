from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image

try:
    import pytesseract
    from pytesseract import TesseractNotFoundError
except Exception:  # pragma: no cover
    pytesseract = None
    TesseractNotFoundError = Exception


def _clean_text(s: str) -> str:
    return " ".join((s or "").replace("\n", " ").split()).strip()


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def extract_ocr(
    image: Image.Image,
    *,
    lang: str = "eng",
    psm: int = 6,
) -> Dict[str, Any]:
    """
    Returns:
      - fullText: merged extracted text
      - words: list of {text, conf, bbox{x,y,w,h}, lineNum, blockNum, pageNum}
      - meta: {lang, psm}
    """
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed")

    try:
        cfg = f"--psm {int(psm)}"
        data = pytesseract.image_to_data(image, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    except TesseractNotFoundError as e:
        raise RuntimeError("tesseract binary not found on server") from e

    n = len(data.get("text", []) or [])
    words: List[Dict[str, Any]] = []
    full_parts: List[str] = []

    for i in range(n):
        text = _clean_text((data.get("text") or [""])[i])
        if not text:
            continue
        conf = _as_float((data.get("conf") or [0])[i], default=0.0)
        left = _as_int((data.get("left") or [0])[i])
        top = _as_int((data.get("top") or [0])[i])
        width = _as_int((data.get("width") or [0])[i])
        height = _as_int((data.get("height") or [0])[i])
        entry = {
            "text": text,
            "conf": conf,
            "bbox": {"x": left, "y": top, "w": width, "h": height},
            "pageNum": _as_int((data.get("page_num") or [0])[i]),
            "blockNum": _as_int((data.get("block_num") or [0])[i]),
            "parNum": _as_int((data.get("par_num") or [0])[i]),
            "lineNum": _as_int((data.get("line_num") or [0])[i]),
            "wordNum": _as_int((data.get("word_num") or [0])[i]),
        }
        words.append(entry)
        full_parts.append(text)

    return {
        "fullText": _clean_text(" ".join(full_parts)),
        "words": words,
        "meta": {"lang": lang, "psm": int(psm)},
    }


def ocr_dependency_ready(required_languages: str = "eng+ara") -> bool:
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        installed = set(pytesseract.get_languages(config=""))
    except Exception:
        return False
    required = {value for value in required_languages.split("+") if value}
    return required.issubset(installed)

