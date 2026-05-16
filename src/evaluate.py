"""
evaluate.py

Evaluates a trained model on the held-out test set and persists per-run metrics.
"""

import os
import csv
import math

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)

from src.visualize import plot_confusion_matrix


def evaluate_model(
    model,
    test_generator,
    run_name: str,
    output_dir: str,
) -> dict:
    """
    Evaluate a trained model and persist all metrics and plots.

    Args:
        model:          Trained tf.keras.Model.
        test_generator: Keras data generator for the test set.
                        Must NOT shuffle (shuffle=False) so labels stay aligned.
        run_name:       Unique identifier for this run, e.g.
                        'ResNet50_Adam_LR1e-4'.  Used as the filename stem for
                        all saved artefacts.
        output_dir:     Root results/ directory.  Sub-directories
                        (metrics/, confusion_matrices/) are created if absent.

    Returns:
        dict with keys:
            run_name, architecture, optimizer, learning_rate,
            accuracy, precision, recall, specificity, f1, auc
    """
    # -- Predict ---------------------------------------------------------------
    # Reset generator so prediction starts at the first batch.
    test_generator.reset()
    y_prob = model.predict(test_generator, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_generator.classes

    # -- Scalar metrics --------------------------------------------------------
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    auc       = roc_auc_score(y_true, y_prob)

    # -- Specificity: TN / (TN + FP) ------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # -- Parse run_name into components ---------------------------------------
    # Expected format: {Architecture}_{Optimizer}_LR{lr}  e.g. ResNet50_Adam_LR1e-4
    parts = run_name.split("_")
    architecture   = parts[0] if len(parts) > 0 else run_name
    optimizer_name = parts[1] if len(parts) > 1 else ""
    learning_rate  = parts[2].replace("LR", "") if len(parts) > 2 else ""

    # -- Confusion matrix plot -------------------------------------------------
    cm_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    plot_confusion_matrix(
        cm=cm,
        run_name=run_name,
        class_names=list(test_generator.class_indices.keys()),
        output_dir=cm_dir,
    )

    # -- Save metrics CSV (one row per run) ------------------------------------
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, f"{run_name}.csv")

    metrics = {
        "run_name":     run_name,
        "architecture": architecture,
        "optimizer":    optimizer_name,
        "learning_rate": learning_rate,
        "accuracy":     round(accuracy,    4),
        "precision":    round(precision,   4),
        "recall":       round(recall,      4),
        "specificity":  round(specificity, 4),
        "f1":           round(f1,          4),
        "auc":          round(auc,         4),
    }

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(
        f"[evaluate] {run_name} | "
        f"acc={accuracy:.4f}  prec={precision:.4f}  rec={recall:.4f}  "
        f"spec={specificity:.4f}  f1={f1:.4f}  auc={auc:.4f}"
    )

    return metrics
