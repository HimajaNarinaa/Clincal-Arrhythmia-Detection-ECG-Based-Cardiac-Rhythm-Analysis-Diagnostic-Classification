"""Classification metrics and confusion-matrix visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ecg_arrhythmia.data import CLASS_NAMES


class PerClassMetrics(TypedDict):
    precision: float
    sensitivity: float
    f1: float
    support: int


class ClassificationMetrics(TypedDict):
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    abnormal_sensitivity: float
    per_class: Dict[str, PerClassMetrics]
    confusion_matrix: List[List[int]]


def classification_metrics(
    true_labels: Sequence[int], predicted_labels: Sequence[int]
) -> ClassificationMetrics:
    """Calculate fixed-label multiclass metrics and abnormal-beat sensitivity."""
    if len(true_labels) == 0 or len(true_labels) != len(predicted_labels):
        raise ValueError("True and predicted labels must have equal, non-zero lengths.")
    labels = [0, 1, 2]
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=labels, zero_division=0
    )
    abnormal_support = int(matrix[1:, :].sum())
    abnormal_sensitivity = (
        float(matrix[1:, 1:].sum() / abnormal_support) if abnormal_support else 0.0
    )
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "abnormal_sensitivity": abnormal_sensitivity,
        "per_class": {
            CLASS_NAMES[index]: {
                "precision": float(precision[index]),
                "sensitivity": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def save_confusion_matrices(
    named_metrics: Dict[str, ClassificationMetrics], output_path: Path
) -> None:
    """Save row-normalized confusion matrices for named evaluation cohorts."""
    if not named_metrics:
        return
    figure, axes = plt.subplots(
        1, len(named_metrics), figsize=(5 * len(named_metrics), 4), squeeze=False
    )
    for axis, (name, metrics) in zip(axes[0], named_metrics.items()):
        matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0
        )
        image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set_title(name)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(range(len(CLASS_NAMES)), labels=("N", "S", "V"))
        axis.set_yticks(range(len(CLASS_NAMES)), labels=("N", "S", "V"))
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column, row, f"{int(matrix[row, column])}\n{normalized[row, column]:.1%}",
                    ha="center", va="center",
                    color="white" if normalized[row, column] > 0.55 else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
