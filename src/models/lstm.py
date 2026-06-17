"""
Deep Bidirectional LSTM model for date-aggregated soil-moisture regression.

Architecture: stacked BiLSTM → GRU → LSTM → GRU → LSTM tower with
BatchNormalization and Dropout at each layer for regularisation.
Input is a single time-step (1, n_features) — the LSTM sees one day of
spatially-aggregated spectral + optional meteo features.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow.keras.layers import (
    BatchNormalization, Bidirectional, Dense, Dropout, GRU, LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.config import LSTM_LR


def build_lstm(n_features: int) -> Sequential:
    """
    Return a compiled deep BiLSTM model.

    Input shape : (1, n_features)  — single time-step per sample.
    Output      : single scalar (10-cm soil moisture %).
    """
    model = Sequential([
        Bidirectional(LSTM(512, return_sequences=True,
                           input_shape=(1, n_features))),
        Dropout(0.4), BatchNormalization(),

        GRU(256,  return_sequences=True), Dropout(0.4), BatchNormalization(),
        LSTM(128, return_sequences=True), Dropout(0.4), BatchNormalization(),
        GRU(64,   return_sequences=True), Dropout(0.4), BatchNormalization(),
        LSTM(32),                         Dropout(0.4), BatchNormalization(),

        Dense(1),
    ], name="bilstm_soilmoisture")

    model.compile(optimizer=Adam(learning_rate=LSTM_LR), loss="mse")
    return model
