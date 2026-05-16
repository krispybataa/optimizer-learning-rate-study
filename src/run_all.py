"""
run_all.py

Master orchestration script for all 72 simulation runs.
Builds the shared data pipeline once, then loops over every combination of:
  - 4 architectures : VGG19, EfficientNetB3, ResNet50, DenseNet121
  - 6 optimizers    : Adam, Adagrad, Adamax, AdaDelta, SGD, RMSProp
  - 3 learning rates: 1e-4, 1e-5, 1e-6

Run directly:  python src/run_all.py
"""

import itertools
import pathlib
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.model_builder import build_model
from src.train import train_model


# -- Configuration -------------------------------------------------------------

RANDOM_SEED   = 42
EPOCHS        = 50
BATCH_SIZE    = 32
INPUT_SIZE    = (256, 256)

ARCHITECTURES  = ['VGG19', 'EfficientNetB3', 'ResNet50', 'DenseNet121']
OPTIMIZERS     = ['Adam', 'Adagrad', 'Adamax', 'AdaDelta', 'SGD', 'RMSProp']
LEARNING_RATES = [1e-4, 1e-5, 1e-6]


# -- Helpers -------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -- Main ----------------------------------------------------------------------

def run_all() -> None:
    _seed_everything(RANDOM_SEED)

    project_root  = pathlib.Path(__file__).resolve().parent.parent
    processed_dir = project_root / "dataset" / "processed"
    results_dir   = project_root / "results"
    summary_dir   = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Section 2 - Shared Data Pipeline (constructed ONCE, reused across all 72)
    # ==========================================================================

    image_generatorFullAugment = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.3,
        zoom_range=0.30,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    image_generatorNoAugment = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
    )

    train_generator = image_generatorFullAugment.flow_from_directory(
        str(processed_dir / "train"),
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_generator = image_generatorFullAugment.flow_from_directory(
        str(processed_dir / "val"),
        target_size=INPUT_SIZE,
        batch_size=1,
        class_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED,
    )

    test_generator = image_generatorNoAugment.flow_from_directory(
        str(processed_dir / "test"),
        target_size=INPUT_SIZE,
        batch_size=1,
        class_mode="binary",
        shuffle=False,
    )

    print(f"Train : {train_generator.samples} images")
    print(f"Val   : {val_generator.samples} images")
    print(f"Test  : {test_generator.samples} images")
    print(f"Classes: {train_generator.class_indices}")
    print()
    print(
        "[WARNING]  EXPERIMENTAL INTEGRITY NOTE: generators are constructed ONCE here "
        "and shared across all 72 runs. Do not re-initialize them inside the loop - "
        "doing so would reshuffle the training data differently per run and confound "
        "optimizer/LR comparisons."
    )

    # ==========================================================================
    # Section 3 - 72-Run Training Loop
    # ==========================================================================

    experiment_matrix = list(itertools.product(ARCHITECTURES, OPTIMIZERS, LEARNING_RATES))
    total_runs        = len(experiment_matrix)

    print(f"\nTOTAL RUNS: {total_runs}")
    print("Starting simulation...\n")

    all_metrics: list[dict] = []

    for counter, (arch, opt, lr) in enumerate(experiment_matrix, start=1):
        lr_str = f"{lr:.0e}"
        print(f"\n{'=' * 60}")
        print(f"  Run {counter:02d}/{total_runs}  |  {arch}  |  {opt}  |  LR={lr_str}")
        print(f"{'=' * 60}")

        model = build_model(
            architecture_name=arch,
            optimizer_name=opt,
            learning_rate=lr,
        )

        metrics = train_model(
            model=model,
            train_generator=train_generator,
            val_generator=val_generator,
            test_generator=test_generator,
            architecture=arch,
            optimizer_name=opt,
            learning_rate=lr,
            results_base_dir=results_dir,
            epochs=EPOCHS,
        )

        all_metrics.append(metrics)

        train_generator.reset()
        test_generator.reset()

    # ==========================================================================
    # Section 4 - Aggregate Results
    # ==========================================================================

    master_df   = pd.DataFrame(all_metrics)
    summary_csv = summary_dir / "master_results.csv"
    master_df.to_csv(summary_csv, index=False)

    print(f"\nMaster results saved -> {summary_csv}")
    print(master_df.to_string())
    print(f"\nALL {total_runs} SIMULATIONS COMPLETE")


# Entry point

if __name__ == "__main__":
    run_all()
