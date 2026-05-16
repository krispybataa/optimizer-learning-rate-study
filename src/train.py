"""
train.py

Runs the training loop for a single model configuration (one of the 84 runs).
Called once per simulation from run_all.py or 02_experiments.ipynb.
"""

import pathlib

import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.evaluate import evaluate_model
from src.visualize import plot_accuracy_loss, plot_auc_roc


def train_model(
    model: tf.keras.Model,
    train_generator,
    val_generator,
    test_generator,
    architecture: str,
    optimizer_name: str,
    learning_rate: float,
    results_base_dir,
    epochs: int = 50,
) -> dict:
    """
    Train one model configuration, evaluate on the test set, and save all artifacts.

    Args:
        model:             Compiled Keras model from model_builder.build_model().
        train_generator:   Training data generator (shared, not re-instantiated).
        val_generator:     Validation data generator.
        test_generator:    Test data generator.
        architecture:      Base architecture name (e.g. 'ResNet50').
        optimizer_name:    Optimizer name (e.g. 'Adam').
        learning_rate:     Learning rate float (e.g. 1e-4).
        results_base_dir:  Path-like; run output written to results_base_dir/run_name/.
        epochs:            Maximum number of training epochs.

    Returns:
        metrics dict as returned by evaluate_model (one row of the master CSV).
    """
    results_base_dir = pathlib.Path(results_base_dir)

    # Step 1 - Setup 
    lr_str   = f"{learning_rate:.0e}"                    # e.g. '1e-04'
    run_name = f"{architecture}_{optimizer_name}_LR{lr_str}"
    run_dir  = results_base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"TRAINING: {run_name}")
    print(f"Architecture : {architecture}")
    print(f"Optimizer    : {optimizer_name}")
    print(f"Learning Rate: {learning_rate}")
    print("=" * 60)

    # Step 2 - Callbacks 
    callbacks = [
        ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
        EarlyStopping(
            monitor="val_auc",
            patience=10,
            mode="max",
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1,
        ),
        CSVLogger(
            filename=str(run_dir / "training_log.csv"),
            append=False,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # Step 3 - Training
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    actual_epochs  = len(history.history["loss"])
    best_val_auc   = max(history.history["val_auc"])
    print(f"\nTraining complete - epochs run: {actual_epochs} / {epochs}  |  best val_auc: {best_val_auc:.4f}")

    # Step 4 - Evaluation and Visualization 
    metrics = evaluate_model(
        model=model,
        test_generator=test_generator,
        run_name=run_name,
        output_dir=str(run_dir),
    )

    plot_accuracy_loss(
        history=history,
        run_name=run_name,
        output_dir=str(run_dir),
    )

    plot_auc_roc(
        model=model,
        test_generator=test_generator,
        run_name=run_name,
        output_dir=str(run_dir),
    )

    metrics["epochs_run"] = actual_epochs

    # Step 5 - Cleanup 
    tf.keras.backend.clear_session()

    print(f"\nCOMPLETED: {run_name}")
    print("=" * 60)

    return metrics
