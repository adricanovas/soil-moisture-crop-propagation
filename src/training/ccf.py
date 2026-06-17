"""
Cross-Correlation Function (CCF) analysis for lag determination.

Unifies the three separate CCF implementations that existed previously:
  - compute_meteo_ccf  in VerticalSoilMoistureLag.py  (all-parcel loop)
  - compute_depth_ccf  in VerticalSoilMoistureLag.py  (inter-depth loop)
  - compute_ccf_lags   in train_cnn_dategroup.py       (single-parcel helper)

All three now delegate to _compute_ccf(), the single core function.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.config import (
    PATCH_IDS, PATCH_LABELS, PATCH_STATION,
    METEO_KEYS, DEEP_COLS,
    MAX_LAG_DAYS_METEO, MAX_LAG_DAYS_DEPTH,
)


# ── Core computation ───────────────────────────────────────────────────────────

def _compute_ccf(sm_series: pd.Series, driver_series: pd.Series,
                 max_lag: int, min_obs: int = 5) -> tuple:
    """
    Compute Pearson r at lags 0..max_lag between sm_series and a shifted driver.

    Returns (best_lag: int, best_corr: float, corr_array: np.ndarray).
    """
    cors = []
    for d in range(max_lag + 1):
        shifted = driver_series.shift(d)
        common  = sm_series.index.intersection(shifted.index)
        s, m    = sm_series.reindex(common), shifted.reindex(common)
        valid   = s.notna() & m.notna()
        if valid.sum() >= min_obs:
            cors.append(pearsonr(s[valid], m[valid])[0])
        else:
            cors.append(np.nan)
    cors = np.array(cors)
    best = int(np.nanargmax(np.abs(cors)))
    return best, float(cors[best]), cors


# ── Single-parcel helpers (used by train_raster.py) ───────────────────────────

def compute_patch_meteo_ccf(sm_series: pd.Series, meteo_df: pd.DataFrame,
                             max_lag: int = MAX_LAG_DAYS_METEO) -> dict:
    """
    Compute optimal meteo lags for one parcel.

    Parameters
    ----------
    sm_series : daily soil-moisture series (index = tz-naive dates).
    meteo_df  : daily meteo DataFrame with METEO_KEYS columns.

    Returns
    -------
    {variable: {"lag": int, "corr": float}}
    """
    sm = sm_series.copy()
    sm.index = pd.to_datetime(sm.index).tz_localize(None).normalize()
    table = {}
    for var in METEO_KEYS:
        lag, corr, _ = _compute_ccf(sm, meteo_df[var], max_lag)
        table[var] = {"lag": lag, "corr": corr}
    return table


# ── All-parcel loops (used by train.py) ───────────────────────────────────────

def compute_meteo_ccf(lstm_datasets: dict, meteo_by_pid: dict,
                      max_lag: int = MAX_LAG_DAYS_METEO) -> dict:
    """
    Compute optimal meteo lags for every parcel.

    Returns {patch_id: {variable: {"lag": int, "corr": float}}}.
    Prints a summary table.
    """
    tables = {}
    for pid in PATCH_IDS:
        sm = lstm_datasets[pid].groupby("date")["soil_moisture"].mean()
        tables[pid] = compute_patch_meteo_ccf(sm, meteo_by_pid[pid], max_lag)

    rows = [
        {"Patch": PATCH_LABELS[pid], "Station": PATCH_STATION[pid],
         "Variable": v, "Optimal_Lag_days": tables[pid][v]["lag"],
         "r": round(tables[pid][v]["corr"], 4)}
        for pid in PATCH_IDS for v in METEO_KEYS
    ]
    print("\n=== Meteorological CCF Summary ===")
    print(pd.DataFrame(rows).to_string(index=False))
    return tables


def compute_depth_ccf(depth_daily: dict,
                      max_lag: int = MAX_LAG_DAYS_DEPTH) -> tuple:
    """
    Compute inter-depth CCF (SM_10cm vs deeper layers) for every parcel.

    Returns (tables, curves) where:
      tables = {pid: {depth_col: {"lag": int, "corr": float}}}
      curves = {pid: {depth_col: corr_array}}
    Prints a summary table.
    """
    tables, curves = {}, {}
    for pid in PATCH_IDS:
        sm_10 = depth_daily[pid]["10cm"]
        tables[pid], curves[pid] = {}, {}
        for dcol in DEEP_COLS:
            lag, corr, cors = _compute_ccf(sm_10, depth_daily[pid][dcol],
                                           max_lag, min_obs=10)
            curves[pid][dcol] = cors
            tables[pid][dcol] = {"lag": lag, "corr": corr}

    rows = [
        {"Patch": PATCH_LABELS[pid], "Depth": dcol,
         "Optimal_Lag_days": tables[pid][dcol]["lag"],
         "r": round(tables[pid][dcol]["corr"], 4)}
        for pid in PATCH_IDS for dcol in DEEP_COLS
    ]
    print("\n=== Inter-Depth CCF Summary (SM_10cm(t) vs SM_z(t-d)) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    return tables, curves
