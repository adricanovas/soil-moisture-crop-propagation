"""
Unified data I/O for the soil-moisture experiments.

Consolidates meteo_io.py + the loading helpers previously scattered across
VerticalSoilMoistureLag.py, train_cnn_dategroup.py, and predict_raster_dategroup.py
into a single module with no duplicates.
"""
import os
import numpy as np
import pandas as pd

from src.config import PATCH_IDS, PATCH_LABELS, PATCH_STATION, METEO_KEYS, DEPTH_COLS

# ── Sentinel-2 band constants ──────────────────────────────────────────────────
BAND_ORDER  = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
_IDX_12     = list(range(1, 13))
_IDX_13     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]  # drops B10 (cirrus, band #11)

# ── Meteo loading ──────────────────────────────────────────────────────────────

def load_station_meteo(path: str) -> pd.DataFrame:
    """Load a SIAR station CSV (comma-sep, Date=%Y-%m-%d) → timezone-naive daily index."""
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="coerce")
    for col in METEO_KEYS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    daily = raw[["date"] + METEO_KEYS].dropna().set_index("date").sort_index()
    daily.index = daily.index.tz_localize(None).normalize()
    return daily


def load_meteo_by_patch(meteo_dir: str) -> dict:
    """
    Load each parcel's assigned station meteo (deduplicated per station).

    Returns {patch_id: DataFrame} where each df has METEO_KEYS columns
    indexed by timezone-naive dates.
    """
    cache: dict = {}
    by_pid: dict = {}
    for pid in PATCH_IDS:
        st = PATCH_STATION[pid]
        if st not in cache:
            cache[st] = load_station_meteo(os.path.join(meteo_dir, f"{st}.csv"))
            d = cache[st]
            print(f"  Station {st}: {len(d)} records "
                  f"({d.index.min().date()} → {d.index.max().date()})")
        by_pid[pid] = cache[st]
        print(f"  {PATCH_LABELS[pid]} → {st}")
    return by_pid


def meteo_for_patch(meteo_dir: str, patch: str) -> tuple:
    """Return (meteo_df, station_name) for a single patch id."""
    station = PATCH_STATION[patch]
    return load_station_meteo(os.path.join(meteo_dir, f"{station}.csv")), station


# ── Raster band reading ────────────────────────────────────────────────────────

def read_bands_12(src) -> dict:
    """
    Return {band_name: 2-D array} from a 12- or 13-band Sentinel-2 rasterio source.

    13-band rasters have B10 (cirrus) as band #11; it is skipped to recover the
    canonical 12-band order used throughout this project.
    """
    if src.count == 13:
        idx = _IDX_13
    elif src.count == 12:
        idx = _IDX_12
    elif src.count > 13:
        idx = _IDX_13
    else:
        raise ValueError(f"raster has {src.count} bands; expected 12 or 13")
    return {name: src.read(k) for name, k in zip(BAND_ORDER, idx)}


# ── Training dataset loading ───────────────────────────────────────────────────

def load_lstm_datasets(dataset_dir: str) -> tuple:
    """
    Load LSTM_<id>_dataset.csv for all parcels.

    Returns ({patch_id: df}, feature_cols) where feature_cols excludes
    'date' and 'soil_moisture'.
    """
    out: dict = {}
    for pid in PATCH_IDS:
        df = pd.read_csv(os.path.join(dataset_dir, f"LSTM_{pid}_dataset.csv"))
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        out[pid] = df
        print(f"  {PATCH_LABELS[pid]} (LSTM): {len(df):,} rows")
    meta = {"date", "soil_moisture"}
    cols = [c for c in out[PATCH_IDS[0]].columns if c not in meta]
    print(f"  LSTM satellite features: {len(cols)}")
    return out, cols


def load_cnn_datasets(dataset_dir: str) -> tuple:
    """
    Load CNN_<id>_dataset.csv for all parcels.

    Returns ({patch_id: df}, feature_cols) where feature_cols excludes
    positional/metadata and target columns.
    """
    out: dict = {}
    for pid in PATCH_IDS:
        df = pd.read_csv(os.path.join(dataset_dir, f"CNN_{pid}_dataset.csv"))
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        out[pid] = df
        print(f"  {PATCH_LABELS[pid]} (CNN):  {len(df):>7,} rows")
    meta = {"row", "colum", "date", "coord_x", "coord_y", "soil_moisture"}
    cols = [c for c in out[PATCH_IDS[0]].columns if c not in meta]
    print(f"  CNN satellite features: {len(cols)}")
    return out, cols


def load_depth_data(sm_dir: str) -> dict:
    """
    Load daily_mean_SoilMoisture_<id>.csv for all parcels.

    Returns {patch_id: df} indexed by date with DEPTH_COLS columns.
    """
    out: dict = {}
    for pid in PATCH_IDS:
        df = pd.read_csv(os.path.join(sm_dir, f"daily_mean_SoilMoisture_{pid}.csv"))
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for c in DEPTH_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        out[pid] = df
        print(f"  {PATCH_LABELS[pid]}: {len(df)} records "
              f"({df.index.min().date()} → {df.index.max().date()})")
    return out
