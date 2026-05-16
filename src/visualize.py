"""
visualize.py

Visualization utilities: accuracy/loss curves, AUC-ROC curves, confusion matrices.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend - safe for headless runs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc as sklearn_auc

# Shared style Settings

_STYLE        = "seaborn-v0_8-whitegrid"
_FIG_DPI      = 150
_CANCER_COLOR = "#C0392B"   # red - cancer class
_NORMAL_COLOR = "#2980B9"   # blue - no_cancer class


# Accuracy / Loss curves 

def plot_accuracy_loss(history, run_name: str, output_dir: str) -> None:
    """
    Plot training and validation accuracy + loss curves and save as PNG.

    Args:
        history:    tf.keras.callbacks.History object (or its .history dict).
        run_name:   Stem used for the output filename.
        output_dir: Directory to save the figure (results/curves/accuracy_loss/).
    """
    os.makedirs(output_dir, exist_ok=True)

    h = history.history if hasattr(history, "history") else history
    epochs = range(1, len(h["accuracy"]) + 1)

    with plt.style.context(_STYLE):
        fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(run_name, fontsize=11, fontweight="bold")

        # Accuracy subplot
        ax_acc.plot(epochs, h["accuracy"],     label="Train Accuracy", linewidth=1.8)
        ax_acc.plot(epochs, h["val_accuracy"], label="Val Accuracy",   linewidth=1.8, linestyle="--")
        ax_acc.set_title("Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.set_ylim(0, 1)

        # Loss subplot
        ax_loss.plot(epochs, h["loss"],     label="Train Loss", linewidth=1.8)
        ax_loss.plot(epochs, h["val_loss"], label="Val Loss",   linewidth=1.8, linestyle="--")
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Binary Cross-Entropy")
        ax_loss.legend()

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"{run_name}.png")
        fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"[visualize] Accuracy/Loss curve saved -> {out_path}")


# AUC-ROC curve 

def plot_auc_roc(model, test_generator, run_name: str, output_dir: str) -> None:
    """
    Compute the ROC curve and save the AUC-ROC plot as PNG.

    Args:
        model:          Trained tf.keras.Model.
        test_generator: Test generator with shuffle=False.
        run_name:       Stem used for the output filename.
        output_dir:     Directory to save the figure (results/curves/auc_roc/).
    """
    os.makedirs(output_dir, exist_ok=True)

    test_generator.reset()
    y_prob = model.predict(test_generator, verbose=0).ravel()
    y_true = test_generator.classes

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = sklearn_auc(fpr, tpr)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color=_CANCER_COLOR, linewidth=2,
                label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1,
                label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {run_name}", fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"{run_name}.png")
        fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"[visualize] AUC-ROC curve saved -> {out_path}")


# Confusion matrix 

def plot_confusion_matrix(
    cm: np.ndarray,
    run_name: str,
    class_names: list,
    output_dir: str,
) -> None:
    """
    Plot a seaborn heatmap of the confusion matrix and save as PNG.

    Args:
        cm:           2x2 numpy confusion matrix from sklearn.
        run_name:     Stem used for the output filename.
        class_names:  List of class label strings, e.g. ['cancer', 'no_cancer'].
        output_dir:   Directory to save the figure (results/confusion_matrices/).
    """
    os.makedirs(output_dir, exist_ok=True)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))

        # Annotate each cell with count and row percentage
        total_per_row = cm.sum(axis=1, keepdims=True)
        pct = cm / total_per_row.clip(min=1) * 100
        annot = np.array(
            [[f"{cm[i, j]}\n({pct[i, j]:.1f}%)" for j in range(cm.shape[1])]
             for i in range(cm.shape[0])]
        )

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.75},
        )
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_title(f"Confusion Matrix - {run_name}", fontweight="bold", fontsize=11)

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"{run_name}.png")
        fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"[visualize] Confusion matrix saved -> {out_path}")
