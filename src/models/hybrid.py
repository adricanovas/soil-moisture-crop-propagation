"""
CNN-LSTM hybrid model for sliding-window soil-moisture regression.

Combines a 1-D CNN feature extractor with a stacked LSTM temporal encoder.
Input is a sliding window of (WINDOW_SIZE, n_features) where the model
learns both spatial/spectral patterns (CNN) and temporal dependencies (LSTM).
One-hot parcel encoding is appended to the feature vector before windowing
so the shared model can be conditioned on parcel identity.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow.keras.layers import (
    BatchNormalization, Conv1D, Dense, Dropout, Input, LSTM,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.config import HYBRID_LR


def build_hybrid(n_features: int, window_size: int) -> Model:
    """
    Return a compiled CNN-LSTM hybrid model.

    Parameters
    ----------
    n_features  : total number of input channels (spectral + meteo + parcel one-hot).
    window_size : number of time steps in the sliding window.

    Input shape : (window_size, n_features)
    Output      : single scalar (10-cm soil moisture %).
    """
    inp = Input(shape=(window_size, n_features), name="window_input")

    x = Conv1D(64,  3, padding="same", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1, name="sm_pred")(x)

    model = Model(inp, out, name="cnn_lstm_hybrid")
    model.compile(optimizer=Adam(learning_rate=HYBRID_LR), loss="mse")
    return model
