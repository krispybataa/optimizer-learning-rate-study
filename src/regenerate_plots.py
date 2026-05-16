"""
regenerate_plots.py

Companion utility to the main training pipeline. Reproduces accuracy/loss and
AUC-ROC plots for any completed run using only its saved artifacts
(training_log.csv and best_model.keras) — no retraining required.

Useful when plot styling or layout changes and figures need to be refreshed
without re-running the full 84-run experiment matrix.
"""

import itertools
import pathlib

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.visualize import plot_accuracy_loss, plot_auc_roc


RANDOM_SEED    = 42
ARCHITECTURES  = ['VGG19', 'EfficientNetB3', 'ResNet50', 'DenseNet121']
OPTIMIZERS     = ['Adam', 'Adagrad', 'Adamax', 'AdaDelta', 'SGD', 'RMSProp']
LEARNING_RATES = [1e-4, 1e-5, 1e-6]


def regenerate_all_plots() -> None:
    project_root = pathlib.Path(__file__).resolve().parent.parent
    results_dir  = project_root / "results"

    test_datagen   = ImageDataGenerator(samplewise_std_normalization=True)
    test_generator = test_datagen.flow_from_directory(
        str(project_root / "dataset" / "processed" / "test"),
        target_size=(256, 256),
        batch_size=1,
        shuffle=False,
        class_mode="binary",
    )

    run_names = [
        f"{arch}_{opt}_LR{lr:.0e}"
        for arch, opt, lr in itertools.product(ARCHITECTURES, OPTIMIZERS, LEARNING_RATES)
    ]

    total_processed = 0
    total_plots     = 0
    skipped: list[str] = []

    for run_name in run_names:
        run_dir = results_dir / run_name

        if not run_dir.exists():
            print(f"[regenerate] SKIP {run_name}: directory not found")
            skipped.append(f"{run_name} (no directory)")
            continue

        model_path = run_dir / "best_model.keras"
        if not model_path.exists():
            print(f"[regenerate] SKIP {run_name}: best_model.keras not found")
            skipped.append(f"{run_name} (no model)")
            continue

        csv_path = run_dir / "training_log.csv"
        if not csv_path.exists():
            print(f"[regenerate] SKIP {run_name}: training_log.csv not found")
            skipped.append(f"{run_name} (no CSV)")
            continue

        # Accuracy / loss curve from CSV
        df = pd.read_csv(csv_path)
        history_dict = {
            "accuracy":     df["accuracy"].tolist(),
            "val_accuracy": df["val_accuracy"].tolist(),
            "loss":         df["loss"].tolist(),
            "val_loss":     df["val_loss"].tolist(),
        }
        plot_accuracy_loss(history_dict, run_name, str(run_dir))
        acc_loss_path = run_dir / f"{run_name}_acc_loss.png"
        print(f"[regenerate] acc_loss saved → {acc_loss_path}")
        total_plots += 1

        # AUC-ROC curve from saved model
        model = tf.keras.models.load_model(str(model_path))
        test_generator.reset()
        plot_auc_roc(model, test_generator, run_name, str(run_dir))
        auc_roc_path = run_dir / f"{run_name}_auc_roc.png"
        print(f"[regenerate] auc_roc saved → {auc_roc_path}")
        total_plots += 1

        tf.keras.backend.clear_session()
        total_processed += 1

    print()
    print("=" * 60)
    print(f"Runs processed : {total_processed}")
    print(f"Plots saved    : {total_plots}")
    if skipped:
        print(f"Runs skipped   : {len(skipped)}")
        for reason in skipped:
            print(f"  - {reason}")
    print("=" * 60)


if __name__ == "__main__":
    regenerate_all_plots()
