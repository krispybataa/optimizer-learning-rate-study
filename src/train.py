"""
train.py

Manages the training loop for a single model configuration.
Responsibilities:
  - Accepting a compiled Keras model and data generators
  - Running model.fit() for 50 epochs with batch size 32
  - Collecting per-epoch accuracy, loss, val_accuracy, val_loss
  - Saving the trained model weights to results/
  - Returning the Keras History object for downstream use by evaluate.py

Designed to be called once per simulation by run_all.py.
"""

# TODO: Implement train_model(model, train_gen, val_gen, run_id) function.
