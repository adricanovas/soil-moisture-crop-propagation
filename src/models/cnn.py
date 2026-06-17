"""
Canonical 1-D CNN for per-pixel soil-moisture regression.

Single definition used by both the main experiment (train.py) and the
raster-training script (train_raster.py).  Replaces the two slightly
divergent build_cnn / build_cnn_model implementations that previously
existed in VerticalSoilMoistureLag.py and train_cnn_dategroup.py.

Architecture rationale
----------------------
- Four Conv1D blocks with increasing filter depth (64→128→256→512).
- MaxPool(2) after the first three blocks, not the last (prevents the
  feature-vector from collapsing to length 0 with small n_features).
- Dropout(0.25) after each pooling for conv-stage regularisation.
- Three Dense layers (256→64→1) with moderate dropout (0.4, 0.2).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Conv1D, Dense, Dropout, Flatten, MaxPooling1D,
)
from tensorflow.keras.models import Sequential


def build_cnn(n_features: int) -> Sequential:
    """
    Return a compiled 1-D CNN for soil-moisture regression.

    Input shape: (n_features, 1)  — one channel per spectral/meteo feature.
    Output: single scalar (soil moisture %).
    """
    model = Sequential([
        Conv1D(64,  3, activation="relu", padding="same",
               input_shape=(n_features, 1)),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.25),

        Conv1D(128, 3, activation="relu", padding="same"),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.25),

        Conv1D(256, 3, activation="relu", padding="same"),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.25),

        Conv1D(512, 3, activation="relu", padding="same"),
        BatchNormalization(), Flatten(),       # no MaxPool on last block

        Dense(256, activation="relu"), Dropout(0.4),
        Dense(64,  activation="relu"), Dropout(0.2),
        Dense(1),
    ], name="cnn_soilmoisture")

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model
