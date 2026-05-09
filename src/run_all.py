"""
run_all.py

Master orchestration script for all 84 simulation runs.
Iterates over the full experiment matrix:
  - 4 architectures : VGG19, EfficientNetB3, ResNet50, DenseNet121
  - 3 learning rates: 0.0001, 0.00001, 0.000001
  - 7 optimizers    : Adam, Adagrad, Adamax, AdaDelta, SGD, RMSProp, <TBD>

For each combination (arch × lr × optimizer), this script:
  1. Builds the model via model_builder.build_model()
  2. Trains it via train.train_model()
  3. Evaluates it via evaluate.evaluate_model()
  4. Saves all artifacts with a deterministic run ID string:
       <arch>__<optimizer>__lr<learning_rate>
  5. After all runs, collates metrics into a single summary CSV and triggers
     aggregate visualization via visualize.

Total runs: 4 × 3 × 7 = 84
"""

# TODO: Implement the nested loop over ARCHITECTURES, LEARNING_RATES, and OPTIMIZERS.
# TODO: Define ARCHITECTURES, LEARNING_RATES, and OPTIMIZERS config lists.
# TODO: Add logging and progress tracking per run.
