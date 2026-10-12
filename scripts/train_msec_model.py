"""Train and evaluate Exposure Lab's 3-way exposure-direction model on MSEC.

The official MSEC filenames encode known exposure shifts. N1/N1.5 are grouped
as underexposed, 0 as well exposed, and P1/P1.5 as overexposed. The official
scene-level train/validation/test archives are kept separate to prevent scene
leakage. Images are read directly from ZIP files.

Usage:
    python scripts/train_msec_model.py training.zip validation.zip testing.zip
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import ML_DIRECTIONS, exposure_feature_vector, load_image  # noqa: E402

LABELS = tuple(ML_DIRECTIONS)
EV_TO_CLASS = {'N1.5': 0, 'N1': 0, '0': 1, 'P1': 2, 'P1.5': 2}
INPUT_PATH = re.compile(r'^INPUT_IMAGES/.+_(N1\.5|N1|0|P1\.5|P1)\.(?:jpe?g|png)$', re.I)
CLASS_NAMES = {i: label for i, label in enumerate(LABELS)}


def read_features(archive_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    features: list[np.ndarray] = []
    targets: list[int] = []
    counts: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if INPUT_PATH.match(name)]
        if not names:
            raise ValueError(f'No MSEC INPUT_IMAGES were found in {archive_path}.')
        for index, name in enumerate(names, 1):
            match = INPUT_PATH.match(name)
            ev = match.group(1).upper()
            try:
                rgb = load_image(archive.read(name))
                features.append(exposure_feature_vector(rgb).reshape(-1))
                targets.append(EV_TO_CLASS[ev])
                counts[ev] += 1
            except Exception as exc:
                raise ValueError(f'Could not process {name}: {exc}') from exc
            if index % 500 == 0 or index == len(names):
                print(f'{archive_path.name}: processed {index:,}/{len(names):,}', flush=True)
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.int32), dict(sorted(counts.items()))


def metrics(expected: np.ndarray, predicted: np.ndarray) -> dict:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for truth, guess in zip(expected.tolist(), predicted.tolist()):
        matrix[int(truth), int(guess)] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = int(matrix[i, i])
        actual = int(matrix[i].sum())
        pred = int(matrix[:, i].sum())
        recall = tp / actual if actual else 0.0
        precision = tp / pred if pred else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': actual}
    total = int(matrix.sum())
    return {
        'accuracy': float(np.trace(matrix) / total) if total else 0.0,
        'balanced_accuracy': float(np.mean([v['recall'] for v in per_class.values()])),
        'macro_f1': float(np.mean([v['f1'] for v in per_class.values()])),
        'confusion_matrix_actual_rows_predicted_columns': matrix.tolist(),
        'per_class': per_class,
    }


def fit_and_evaluate(train_zip: Path, validation_zip: Path, test_zip: Path, model_path: Path) -> dict:
    x_train, y_train, train_counts = read_features(train_zip)
    x_val, y_val, val_counts = read_features(validation_zip)

    model = cv2.ml.RTrees_create()
    model.setMaxDepth(16)
    model.setMinSampleCount(5)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0))
    model.setCalculateVarImportance(True)
    model.setActiveVarCount(0)
    train_data = cv2.ml.TrainData_create(x_train, cv2.ml.ROW_SAMPLE, y_train)
    if not model.train(train_data):
        raise RuntimeError('OpenCV RTrees training failed.')

    _, val_response = model.predict(x_val)
    val_pred = np.rint(val_response.reshape(-1)).astype(np.int32)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    x_test, y_test, test_counts = read_features(test_zip)
    _, test_response = model.predict(x_test)
    test_pred = np.rint(test_response.reshape(-1)).astype(np.int32)
    return {
        'dataset': 'MSEC — Learning Multi-Scale Photo Exposure Correction',
        'label_definition': {
            'Underexposed': ['-1 EV', '-1.5 EV'],
            'Well Exposed': ['0 EV'],
            'Overexposed': ['+1 EV', '+1.5 EV'],
        },
        'model': 'OpenCV Random Forest (RTrees), 11 image-statistics features',
        'training_images': int(len(y_train)),
        'validation_images': int(len(y_val)),
        'test_images': int(len(y_test)),
        'exposure_level_counts': {'training': train_counts, 'validation': val_counts, 'testing': test_counts},
        'validation': metrics(y_val, val_pred),
        'test': metrics(y_test, test_pred),
        'test_is_held_out_scene_split': True,
        'model_file': str(model_path),
        'interpretation': 'These scores measure 3-way exposure direction on MSEC EV-shifted images. They are not accuracy for the app’s seven severity categories or a guarantee for arbitrary real-world photos.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training_zip', type=Path)
    parser.add_argument('validation_zip', type=Path)
    parser.add_argument('testing_zip', type=Path)
    parser.add_argument('--model', type=Path, default=Path('models/msec_exposure_direction.xml'))
    parser.add_argument('--output', type=Path, default=Path('outputs/msec_model_results.json'))
    args = parser.parse_args()
    result = fit_and_evaluate(args.training_zip, args.validation_zip, args.testing_zip, args.model)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
