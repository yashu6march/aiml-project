"""Evaluate Exposure Lab's current exposure rules on the MSEC test split.

The official MSEC input filenames encode a known exposure adjustment
(-1.5, -1, 0, +1, +1.5 EV). This script evaluates only exposure direction
(under / well / over), because the benchmark does not provide the app's seven
severity labels. It reads images directly from testing.zip; no extraction is
needed.

Usage:
    python scripts/evaluate_exposure_benchmark.py /path/to/testing.zip
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import analyze, load_image

LABELS = ("Underexposed", "Well Exposed", "Overexposed")
EV_SUFFIXES = {
    "N1.5": "Underexposed",
    "N1": "Underexposed",
    "0": "Well Exposed",
    "P1": "Overexposed",
    "P1.5": "Overexposed",
}
INPUT_PATH = re.compile(r"^INPUT_IMAGES/.+_(N1\.5|N1|0|P1\.5|P1)\.(?:jpe?g|png)$", re.I)


def predicted_direction(category: str) -> str:
    if "Underexposed" in category:
        return "Underexposed"
    if "Overexposed" in category:
        return "Overexposed"
    return "Well Exposed"


def scores(matrix: list[list[int]]) -> dict[str, float]:
    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(len(LABELS)))
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        actual = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        recall = tp / actual if actual else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    return {
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": sum(x["recall"] for x in per_class.values()) / len(LABELS),
        "macro_f1": sum(x["f1"] for x in per_class.values()) / len(LABELS),
        "per_class": per_class,
    }


def evaluate(zip_path: Path) -> dict:
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    seen = Counter()
    errors: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if INPUT_PATH.match(name)]
        if not names:
            raise ValueError("No MSEC INPUT_IMAGES were found in this ZIP.")
        for index, name in enumerate(names, start=1):
            match = INPUT_PATH.match(name)
            ev = match.group(1).upper()
            truth = EV_SUFFIXES[ev]
            try:
                rgb = load_image(archive.read(name))
                result = analyze(rgb)
                guess = predicted_direction(result["category"])
                matrix[LABELS.index(truth)][LABELS.index(guess)] += 1
                seen[ev] += 1
            except Exception as exc:  # continue and disclose unreadable files
                errors.append({"file": name, "error": str(exc)})
            if index % 250 == 0 or index == len(names):
                print(f"Processed {index:,}/{len(names):,} images", flush=True)

    evaluated = sum(map(sum, matrix))
    if not evaluated:
        raise ValueError("No readable test images were evaluated.")
    return {
        "dataset": "MSEC / Learning Multi-Scale Photo Exposure Correction",
        "split": zip_path.name,
        "task": "3-way exposure direction (EV suffix labels grouped as under / well / over)",
        "method": "Existing explainable image-statistics baseline in app.py; no model training",
        "evaluated_images": evaluated,
        "skipped_images": len(errors),
        "ev_level_counts": dict(sorted(seen.items())),
        "label_order": list(LABELS),
        "confusion_matrix_actual_rows_predicted_columns": matrix,
        **scores(matrix),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_zip", type=Path, help="MSEC testing.zip")
    parser.add_argument("--output", type=Path, help="Optional path to save JSON results")
    args = parser.parse_args()
    result = evaluate(args.test_zip)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
