#!/usr/bin/env python3
"""
Train a per-pixel CNN (sat+meteo_optlag, date-grouped split) for ONE parcel
and save the model + scaler + metadata for raster inference.

The date-grouped split (GroupShuffleSplit on acquisition date) ensures no
pixels from the same Sentinel-2 scene appear in both train and test sets,
avoiding temporal leakage that inflates per-pixel CNN scores.

Usage
-----
python train_raster.py --patch P07-A1
python train_raster.py --patch 5f182afa1cb289095a59ed80 --epochs 300

Outputs  (under --output-dir/<patch>/)
-------
  cnn_dategroup_<patch>.keras
  scaler_dategroup_<patch>.joblib
  metadata_<patch>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.config import (
    PATCH_STATION, METEO_KEYS,
    CNN_EPOCHS, CNN_BATCH,
    TEST_SIZE, RANDOM_STATE,
)
from src.data.io       import load_station_meteo, BAND_ORDER
from src.data.features import INDEX_ORDER
from src.models.cnn    import build_cnn
from src.training.ccf  import compute_patch_meteo_ccf
from src.training.fusion import merge_meteo_by_lag

DEFAULT_DATA_DIR   = os.path.join(os.path.dirname(__file__), "data", "datasets")
DEFAULT_METEO_DIR  = os.path.join(os.path.dirname(__file__), "data", "meteo")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")

VAL_SPLIT = 0.15
PATIENCE  = 15


def seed_everything(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch",      required=True,
                    help="Parcel id (hex or P0x-A1).")
    ap.add_argument("--data-dir",   default=DEFAULT_DATA_DIR)
    ap.add_argument("--meteo-dir",  default=DEFAULT_METEO_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--epochs",     type=int, default=CNN_EPOCHS)
    ap.add_argument("--batch",      type=int, default=CNN_BATCH)
    args = ap.parse_args()
    seed_everything()

    if args.patch not in PATCH_STATION:
        sys.exit(f"ERROR: unknown patch {args.patch!r}; known: {list(PATCH_STATION)}")
    station = PATCH_STATION[args.patch]

    # ── Load dataset ───────────────────────────────────────────────────────────
    path = Path(args.data_dir) / f"CNN_{args.patch}_dataset.csv"
    if not path.exists():
        sys.exit(f"ERROR: dataset not found: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    print(f"[1/6] {args.patch}: {len(df):,} px-rows, {df['date'].nunique()} dates")

    # ── Load meteo ─────────────────────────────────────────────────────────────
    meteo_file = Path(args.meteo_dir) / f"{station}.csv"
    if not meteo_file.exists():
        sys.exit(f"ERROR: meteo CSV not found: {meteo_file}")
    meteo = load_station_meteo(str(meteo_file))
    print(f"[2/6] meteo {station}: {len(meteo)} records "
          f"({meteo.index.min().date()} → {meteo.index.max().date()})")

    # ── CCF lags ───────────────────────────────────────────────────────────────
    sm_series = df.groupby("date")["soil_moisture"].mean()
    lag_table = compute_patch_meteo_ccf(sm_series, meteo)
    print("[3/6] CCF lags:")
    for var, info in lag_table.items():
        print(f"      {var:13s} lag={info['lag']:>2}d  corr={info['corr']:+.4f}")

    # ── Feature fusion ─────────────────────────────────────────────────────────
    merged = merge_meteo_by_lag(df, lag_table, meteo)
    feature_cols = ([c for c in BAND_ORDER    if c in merged.columns] +
                    [c for c in INDEX_ORDER   if c in merged.columns] +
                    METEO_KEYS)
    missing = set(BAND_ORDER + INDEX_ORDER + METEO_KEYS) - set(merged.columns)
    if missing:
        sys.exit(f"ERROR: dataset missing columns: {sorted(missing)}")

    # ── Train / test split ─────────────────────────────────────────────────────
    data   = merged.dropna(subset=feature_cols + ["soil_moisture"]).copy()
    X      = data[feature_cols].values.astype(np.float32)
    y      = data["soil_moisture"].values.astype(np.float32)
    groups = pd.to_datetime(data["date"]).dt.normalize().values
    n_dates = int(pd.Series(groups).nunique())
    print(f"[4/6] usable: {len(data):,} px, {n_dates} dates, {len(feature_cols)} features")
    if n_dates < 5:
        sys.exit(f"ERROR: only {n_dates} dates; cannot GroupShuffleSplit.")

    gss    = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=groups))

    scaler = RobustScaler()
    X_tr   = scaler.fit_transform(X[tr]).reshape(-1, len(feature_cols), 1)
    X_te   = scaler.transform(X[te]).reshape(-1, len(feature_cols), 1)

    print(f"[5/6] train {len(tr):,}px / test {len(te):,}px — training CNN …")
    model = build_cnn(len(feature_cols))
    model.fit(X_tr, y[tr],
              validation_split=VAL_SPLIT,
              epochs=args.epochs, batch_size=args.batch,
              callbacks=[EarlyStopping("val_loss", patience=PATIENCE,
                                       restore_best_weights=True, verbose=1)],
              verbose=2)

    preds   = model.predict(X_te, verbose=0).flatten()
    metrics = {
        "r2":             float(r2_score(y[te], preds)),
        "mae":            float(mean_absolute_error(y[te], preds)),
        "rmse":           float(np.sqrt(mean_squared_error(y[te], preds))),
        "n_train_pixels": int(len(tr)),
        "n_test_pixels":  int(len(te)),
        "n_train_dates":  int(pd.Series(groups[tr]).nunique()),
        "n_test_dates":   int(pd.Series(groups[te]).nunique()),
    }
    print(f"\n=== Test (held-out dates) ===  "
          f"R²={metrics['r2']:+.4f}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")

    # ── Save artefacts ─────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / args.patch
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(out_dir / f"cnn_dategroup_{args.patch}.keras")
    joblib.dump(scaler, out_dir / f"scaler_dategroup_{args.patch}.joblib")

    metadata = {
        "patch": args.patch, "station": station,
        "config": "sat+meteo_optlag",
        "feature_order": feature_cols,
        "lag_table": lag_table,
        "test_metrics": metrics,
        "split": {
            "strategy": "GroupShuffleSplit", "group_col": "date",
            "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
        },
        "scaler": "RobustScaler",
    }
    (out_dir / f"metadata_{args.patch}.json").write_text(
        json.dumps(metadata, indent=2, default=str))
    print(f"[6/6] saved → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
