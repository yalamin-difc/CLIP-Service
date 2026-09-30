#!/usr/bin/env python3
"""D-4: calibrate SigLIP2's min_score/min_margin from a labelled eval set.

Given a labelled set of (topScore, secondScore, label) cases -- where
label=true means "the top-ranked candidate for this query really is the
correct match" and label=false means it is not -- this sweeps a grid of
(min_score, min_margin) pairs, classifies every case through the exact
same decision function the live service uses (app.should_return_no_match,
imported and reused rather than reimplemented, so calibration can never
silently diverge from runtime behaviour), and writes the winning pair to
a calibration.json that embedding_engines/config.py's
get_siglip2_calibration_status()/get_siglip2_min_score()/
get_siglip2_min_margin() read at runtime.

Winner selection:
  * Default: the (min_score, min_margin) pair that maximises F1 over the
    labelled cases.
  * --target-precision P: among pairs whose precision is >= P, the one
    that maximises recall (i.e. the loosest thresholds that still hit the
    required precision). Fails loudly (non-zero exit) if no pair reaches
    P, rather than silently falling back to the F1 winner.

This script NEVER touches CLIP's CONF_MIN_SCORE/CONF_MIN_MARGIN, and
never runs a real model itself -- it only consumes already-computed
scores. Producing those scores from a real, reviewed, labelled set of
recovery cases is D-5's job, not this script's; see docs/eval/README.md
for exactly what "isSynthetic: true" does and does not prove about a
sweep run against the bundled example set.

Usage:
    python scripts/calibrate.py --eval-set docs/eval/synthetic_calibration_set.json
    python scripts/calibrate.py --eval-set path/to/real_labelled_set.json \\
        --output docs/eval/calibration.json --target-precision 0.9
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import should_return_no_match  # noqa: E402

REQUIRED_TOP_KEYS = {"cases", "isSynthetic"}
REQUIRED_CASE_KEYS = {"caseId", "topScore", "label"}


def load_eval_set(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = REQUIRED_TOP_KEYS - data.keys()
    if missing:
        raise ValueError(f"Eval set missing required top-level keys: {sorted(missing)}")
    if not data.get("isSynthetic", False):
        raise ValueError(
            "This script only accepts eval sets explicitly marked isSynthetic=true today. "
            "D-4 depends on D-5's real, reviewed, labelled eval set, which does not exist in "
            "this repository yet -- see docs/eval/README.md, 'Calibration (D-4)'. Do not point "
            "this at unreviewed real data without first deliberately loosening this check, the "
            "same way docs/eval/README.md documents for scripts/evaluate_models.py's manifest."
        )
    if not data["cases"]:
        raise ValueError("Eval set contains no cases.")
    for case in data["cases"]:
        missing_case = REQUIRED_CASE_KEYS - case.keys()
        if missing_case:
            raise ValueError(f"Case {case.get('caseId', '?')!r} missing required keys: {sorted(missing_case)}")
        if not isinstance(case["label"], bool):
            raise ValueError(
                f"Case {case['caseId']!r}: 'label' must be a real boolean "
                "(true = the top-ranked candidate is a genuine match), not a string or number."
            )
    return data


def _predicted_accept(case: Dict[str, Any], min_score: float, min_margin: float) -> bool:
    top_score = case["topScore"]
    if top_score is None:
        return False
    second_score = case.get("secondScore")
    no_match, _meta = should_return_no_match(top_score, second_score, min_score, min_margin)
    return not no_match


def confusion_matrix(cases: List[Dict[str, Any]], min_score: float, min_margin: float) -> Dict[str, int]:
    tp = fp = fn = tn = 0
    for case in cases:
        accept = _predicted_accept(case, min_score, min_margin)
        label = case["label"]
        if accept and label:
            tp += 1
        elif accept and not label:
            fp += 1
        elif not accept and label:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_metrics(confusion: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = confusion["tp"], confusion["fp"], confusion["fn"], confusion["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def build_grid(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    values = []
    value = start
    # round() avoids float-accumulation drift (e.g. 0.30000000000000004)
    # from repeatedly adding a fractional step.
    while value <= stop + 1e-9:
        values.append(round(value, 6))
        value += step
    return values


def sweep(
    cases: List[Dict[str, Any]],
    score_grid: List[float],
    margin_grid: List[float],
    target_precision: Optional[float],
) -> Dict[str, Any]:
    """Evaluates every (min_score, min_margin) pair in the grid. Ties are
    broken deterministically (higher min_score, then higher min_margin) so
    a re-run of the same eval set and grid always reproduces the same
    winner -- never an arbitrary pick that happens to depend on iteration
    or dict order."""
    best_key: Optional[Tuple[float, float, float]] = None
    best_row: Optional[Dict[str, Any]] = None
    all_results = []
    for min_score in score_grid:
        for min_margin in margin_grid:
            confusion = confusion_matrix(cases, min_score, min_margin)
            metrics = compute_metrics(confusion)
            row = {"minScore": min_score, "minMargin": min_margin, "confusion": confusion, "metrics": metrics}
            all_results.append(row)
            if target_precision is not None:
                if metrics["precision"] < target_precision:
                    continue
                key = (metrics["recall"], min_score, min_margin)
            else:
                key = (metrics["f1"], min_score, min_margin)
            if best_key is None or key > best_key:
                best_key = key
                best_row = row
    if best_row is None:
        raise ValueError(
            f"No (minScore, minMargin) pair in the sweep grid reached the target precision "
            f"{target_precision} over {len(cases)} case(s). Widen --score-min/--score-max/"
            "--margin-min/--margin-max, gather more labelled cases, or lower --target-precision."
        )
    return {"winner": best_row, "allResults": all_results}


def calibrate(
    eval_set_path: str,
    output_path: str,
    score_range: Tuple[float, float, float],
    margin_range: Tuple[float, float, float],
    target_precision: Optional[float],
) -> Dict[str, Any]:
    data = load_eval_set(eval_set_path)
    cases = data["cases"]
    score_grid = build_grid(*score_range)
    margin_grid = build_grid(*margin_range)
    result = sweep(cases, score_grid, margin_grid, target_precision)
    winner = result["winner"]
    calibration = {
        "minScore": winner["minScore"],
        "minMargin": winner["minMargin"],
        "evalDate": time.strftime("%Y-%m-%d", time.gmtime()),
        "evalSetPath": str(eval_set_path),
        "datasetVersion": data.get("datasetVersion"),
        "isSynthetic": data.get("isSynthetic"),
        "caseCount": len(cases),
        "optimizedFor": f"targetPrecision>={target_precision}" if target_precision is not None else "maxF1",
        "metrics": winner["metrics"],
        "confusion": winner["confusion"],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"calibration": calibration, "sweep": result["allResults"]}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-4: calibrate SigLIP2's min_score/min_margin from a labelled eval set.")
    parser.add_argument("--eval-set", required=True, help="Path to a labelled eval set JSON (see docs/eval/README.md).")
    parser.add_argument(
        "--output",
        default="docs/eval/calibration.json",
        help="Where to write the winning calibration (default matches SIGLIP2_CALIBRATION_FILE's default).",
    )
    parser.add_argument("--score-min", type=float, default=0.0)
    parser.add_argument("--score-max", type=float, default=1.0)
    parser.add_argument("--score-step", type=float, default=0.01)
    parser.add_argument("--margin-min", type=float, default=0.0)
    parser.add_argument("--margin-max", type=float, default=0.3)
    parser.add_argument("--margin-step", type=float, default=0.01)
    parser.add_argument(
        "--target-precision",
        type=float,
        default=None,
        help="If set, maximise recall among (minScore, minMargin) pairs at/above this precision instead of maximising F1.",
    )
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = calibrate(
        args.eval_set,
        args.output,
        (args.score_min, args.score_max, args.score_step),
        (args.margin_min, args.margin_max, args.margin_step),
        args.target_precision,
    )
    print(json.dumps(result["calibration"], indent=2, ensure_ascii=False))
    print(f"Wrote calibration to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
