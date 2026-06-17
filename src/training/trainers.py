"""
Training loops for the four model types.

Each function takes a prepared DataFrame, feature column list, config/patch
labels and returns a metrics dict (or None if skipped). All four use the
canonical hyperparameters from src.config and the model builders from
src.models.*.

Improvements vs the original VerticalSoilMoistureLag.py:
  - _impute is shared across all trainers.
  - EarlyStopping patience and val_split are constants, not magic numbers.
  - make_windows is used only by train_hybrid; cleaner interface.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping

from src.config import (
    PATCH_IDS, PATCH_LABELS,
    LSTM_EPOCHS, LSTM_BATCH,
    CNN_EPOCHS,  CNN_BATCH,
    HYBRID_EPOCHS, HYBRID_BATCH,
    WINDOW_SIZE, STRIDE,
    TEST_SIZE, RANDOM_STATE,
)
from src.models.cnn    import build_cnn
from src.models.lstm   import build_lstm
from src.models.hybrid import build_hybrid
from src.training.metrics import compute_metrics

# ── Shared constants ───────────────────────────────────────────────────────────
_LSTM_VAL   = 0.20
_CNN_VAL    = 0.15
_HYBRID_VAL = 0.20
_PATIENCE_LSTM   = 20
_PATIENCE_CNN    = 15
_PATIENCE_HYBRID = 20


# ── Utilities ──────────────────────────────────────────────────────────────────

def _impute(X: np.ndarray) -> np.ndarray:
    """Replace inf/nan with column means (or 0 when the whole column is NaN)."""
    X = np.where(np.isinf(X), np.nan, X)
    means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = means[j] if not np.isnan(means[j]) else 0.0
    return X


def _early_stop(monitor: str = "val_loss", patience: int = 15) -> EarlyStopping:
    return EarlyStopping(monitor=monitor, patience=patience,
                         restore_best_weights=True, verbose=0)


# ── LSTM trainer ───────────────────────────────────────────────────────────────

def train_lstm(df: pd.DataFrame, feature_cols: list,
               config_name: str, patch_label: str) -> dict | None:
    data = df.dropna(subset=feature_cols + ["soil_moisture"]).copy()
    if len(data) < 10:
        print(f"  {patch_label} [LSTM/{config_name}]: SKIPPED ({len(data)} rows)")
        return None

    X  = _impute(data[feature_cols].values.astype(float))
    y  = data["soil_moisture"].values.reshape(-1, 1)
    sX, sY = MinMaxScaler(), MinMaxScaler()
    Xs, ys = sX.fit_transform(X), sY.fit_transform(y)

    Xtr, Xte, ytr, yte = train_test_split(
        Xs, ys, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    Xtr = Xtr.reshape(-1, 1, len(feature_cols))
    Xte = Xte.reshape(-1, 1, len(feature_cols))

    model = build_lstm(len(feature_cols))
    model.fit(Xtr, ytr,
              validation_split=_LSTM_VAL,
              epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
              callbacks=[_early_stop(patience=_PATIENCE_LSTM)],
              verbose=0)

    preds  = sY.inverse_transform(model.predict(Xte, verbose=0).reshape(-1, 1)).flatten()
    y_true = sY.inverse_transform(yte).flatten()
    return compute_metrics("LSTM", config_name, patch_label,
                           y_true, preds, len(ytr), len(feature_cols))


# ── CNN trainer (random split) ─────────────────────────────────────────────────

def train_cnn(df: pd.DataFrame, feature_cols: list,
              config_name: str, patch_label: str) -> dict | None:
    data = df.dropna(subset=feature_cols + ["soil_moisture"]).copy()
    if len(data) < 20:
        print(f"  {patch_label} [CNN/{config_name}]: SKIPPED ({len(data)} rows)")
        return None

    X  = data[feature_cols].values.astype(float)
    y  = data["soil_moisture"].values
    Xs = RobustScaler().fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    Xtr = Xtr.reshape(-1, len(feature_cols), 1)
    Xte = Xte.reshape(-1, len(feature_cols), 1)

    model = build_cnn(len(feature_cols))
    model.fit(Xtr, ytr,
              validation_split=_CNN_VAL,
              epochs=CNN_EPOCHS, batch_size=CNN_BATCH,
              callbacks=[_early_stop(patience=_PATIENCE_CNN)],
              verbose=0)

    preds = model.predict(Xte, verbose=0).flatten()
    return compute_metrics("CNN", config_name, patch_label,
                           yte, preds, len(ytr), len(feature_cols))


# ── CNN-DateGroup trainer (leakage-free) ───────────────────────────────────────

def train_cnn_dategroup(df: pd.DataFrame, feature_cols: list,
                        config_name: str, patch_label: str) -> dict | None:
    """
    Same CNN architecture as train_cnn but with GroupShuffleSplit by acquisition
    date: no pixels from the same date appear in both train and test sets.

    The R² gap between CNN and CNN-DateGroup quantifies the magnitude of
    temporal-leakage inflating the random-split CNN score.
    """
    data = df.dropna(subset=feature_cols + ["soil_moisture"]).copy()
    if len(data) < 20 or data["date"].nunique() < 5:
        print(f"  {patch_label} [CNN-DateGroup/{config_name}]: SKIPPED "
              f"({len(data)} rows, {data['date'].nunique()} dates)")
        return None

    X      = data[feature_cols].values.astype(float)
    y      = data["soil_moisture"].values
    groups = data["date"].values
    Xs     = RobustScaler().fit_transform(X)

    gss     = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te  = next(gss.split(Xs, y, groups))
    Xtr     = Xs[tr].reshape(-1, len(feature_cols), 1)
    Xte     = Xs[te].reshape(-1, len(feature_cols), 1)

    model = build_cnn(len(feature_cols))
    model.fit(Xtr, y[tr],
              validation_split=_CNN_VAL,
              epochs=CNN_EPOCHS, batch_size=CNN_BATCH,
              callbacks=[_early_stop(patience=_PATIENCE_CNN)],
              verbose=0)

    preds = model.predict(Xte, verbose=0).flatten()
    return compute_metrics("CNN-DateGroup", config_name, patch_label,
                           y[te], preds, len(tr), len(feature_cols))


# ── CNN-LSTM Hybrid trainer (pooled) ─────────────────────────────────────────

def _make_windows(df_dict: dict, feature_cols: list,
                  window_size: int, stride: int) -> tuple:
    """
    Build sliding-window arrays from multiple parcels with one-hot parcel encoding.

    Returns (X: float32 [N, window, n_feat+n_parcels],
             y: float32 [N],
             patch_ids: list[str])
    """
    n_p = len(PATCH_IDS)
    Xs, ys, pids = [], [], []
    for p_idx, pid in enumerate(PATCH_IDS):
        df     = df_dict[pid].sort_values("date")
        df     = df.dropna(subset=feature_cols + ["soil_moisture"]).reset_index(drop=True)
        feats  = _impute(df[feature_cols].values.astype(float))
        target = df["soil_moisture"].values.astype(float)
        oh     = np.zeros((len(df), n_p)); oh[:, p_idx] = 1.0
        data   = np.concatenate([feats, oh], axis=1)
        for s in range(0, len(data) - window_size + 1, stride):
            Xs.append(data[s:s + window_size])
            ys.append(target[s + window_size - 1])
            pids.append(PATCH_LABELS[pid])
    return np.array(Xs, np.float32), np.array(ys, np.float32), pids


def train_hybrid(ds_dict: dict, feature_cols: list, config_name: str) -> tuple:
    """
    Train the CNN-LSTM hybrid on all parcels pooled with one-hot encoding.

    Returns (pooled_result_dict, [per_patch_result_dict, ...]).
    Both are None / empty list on skip.
    """
    X, y, pids = _make_windows(ds_dict, feature_cols, WINDOW_SIZE, STRIDE)
    if len(X) < 20:
        print(f"  [CNN-LSTM/{config_name}]: SKIPPED ({len(X)} windows)")
        return None, []

    n_data  = len(feature_cols)
    n_total = n_data + len(PATCH_IDS)

    sX, sY = MinMaxScaler(), MinMaxScaler()
    sX.fit(X[..., :n_data].reshape(-1, n_data))
    Xsc = X.copy()
    for t in range(WINDOW_SIZE):
        Xsc[:, t, :n_data] = sX.transform(X[:, t, :n_data])
    ysc = sY.fit_transform(y.reshape(-1, 1)).flatten()

    Xtr, Xte, ytr, yte_sc, id_tr, id_te = train_test_split(
        Xsc, ysc, pids, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    yte_sc = np.array(yte_sc)

    model = build_hybrid(n_total, WINDOW_SIZE)
    model.fit(Xtr, ytr,
              validation_split=_HYBRID_VAL,
              epochs=HYBRID_EPOCHS, batch_size=HYBRID_BATCH,
              callbacks=[_early_stop(patience=_PATIENCE_HYBRID)],
              verbose=0)

    preds  = sY.inverse_transform(model.predict(Xte, verbose=0).reshape(-1, 1)).flatten()
    y_true = sY.inverse_transform(yte_sc.reshape(-1, 1)).flatten()

    pooled = compute_metrics("CNN-LSTM", config_name, "Pooled",
                             y_true, preds, len(ytr), n_data)

    per_patch = []
    for pid in PATCH_IDS:
        lbl = PATCH_LABELS[pid]
        idx = [i for i, p in enumerate(id_te) if p == lbl]
        if not idx:
            continue
        per_patch.append(compute_metrics(
            "CNN-LSTM", config_name, lbl,
            y_true[idx], preds[idx], None, n_data))

    return pooled, per_patch
