"""
model_builder.py

Builds compiled Keras transfer-learning models for the 72-simulation experiment matrix.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19, EfficientNetB3, ResNet50, DenseNet121
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import (
    Adam, Adagrad, Adamax, Adadelta, SGD, RMSprop,
)

# Architecture registry 

_ARCHITECTURE_REGISTRY = {
    "VGG19":           VGG19,
    "EfficientNetB3":  EfficientNetB3,
    "ResNet50":        ResNet50,
    "DenseNet121":     DenseNet121,
}

# Optimizer registry 
def _make_optimizer(name: str, learning_rate: float):
    """Instantiate an optimizer by name with the given learning rate."""
    registry = {
        "Adam":     lambda lr: Adam(learning_rate=lr),
        "Adagrad":  lambda lr: Adagrad(learning_rate=lr),
        "Adamax":   lambda lr: Adamax(learning_rate=lr),
        "AdaDelta": lambda lr: Adadelta(learning_rate=lr),
        "SGD":      lambda lr: SGD(learning_rate=lr, momentum=0.9),
        "RMSProp":  lambda lr: RMSprop(learning_rate=lr),
    }
    if name not in registry:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Choose from: {sorted(registry.keys())}"
        )
    return registry[name](learning_rate)


# Model builder 
def build_model(
    architecture_name: str,
    optimizer_name: str,
    learning_rate: float,
    input_shape: tuple = (256, 256, 3),
) -> Model:
    """
    Build and compile a transfer-learning model for binary ovarian cancer classification.

    Args:
        architecture_name: One of 'VGG19', 'EfficientNetB3', 'ResNet50', 'DenseNet121'.
        optimizer_name:    One of 'Adam', 'Adagrad', 'Adamax', 'AdaDelta',
                           'SGD', 'RMSProp'.
        learning_rate:     Floating-point learning rate (e.g. 1e-4).
        input_shape:       HWC tuple; must match the generators (default 256x256x3).

    Returns:
        A compiled tf.keras.Model ready for model.fit().
    """
    if architecture_name not in _ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture_name}'. "
            f"Choose from: {sorted(_ARCHITECTURE_REGISTRY.keys())}"
        )

    # Base model (frozen ImageNet weights) 
    base_cls = _ARCHITECTURE_REGISTRY[architecture_name]
    base_model = base_cls(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False

    # Custom classification head 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Compile 
    optimizer = _make_optimizer(optimizer_name, learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )

    return model
