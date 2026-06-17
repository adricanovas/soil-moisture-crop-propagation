"""
Feature fusion: merge meteorological and depth features into satellite datasets.

merge_meteo_by_lag and merge_depth_by_lag were previously duplicated in
VerticalSoilMoistureLag.py and train_cnn_dategroup.py. Both scripts now
import from here.
"""
import pandas as pd

from src.config import PATCH_IDS, METEO_KEYS, DEEP_COLS, CONFIGS


def merge_meteo_by_lag(df: pd.DataFrame, lag_dict: dict,
                       meteo_src: pd.DataFrame,
                       meteo_keys: list = METEO_KEYS) -> pd.DataFrame:
    """
    Left-join lagged meteo columns into a pixel/date DataFrame.

    Parameters
    ----------
    df        : dataset with a 'date' column.
    lag_dict  : {variable: {"lag": int, ...}}  (from CCF analysis).
    meteo_src : daily meteo DataFrame indexed by tz-naive dates.
    meteo_keys: meteo variables to merge (default: all METEO_KEYS).

    Returns a copy of df with meteo columns appended.
    """
    result = df.copy()
    result["_md"] = pd.to_datetime(result["date"]).dt.tz_localize(None).dt.normalize()
    for var in meteo_keys:
        lag_d   = lag_dict.get(var, {}).get("lag", 0)
        shifted = meteo_src[[var]].shift(lag_d).copy()
        shifted["_md"] = shifted.index
        result = result.merge(shifted[["_md", var]], on="_md", how="left")
    return result.drop(columns=["_md"])


def merge_depth_by_lag(df: pd.DataFrame, depth_src: pd.DataFrame,
                       depth_cols: list = DEEP_COLS,
                       lag_dict: dict | None = None) -> pd.DataFrame:
    """
    Left-join lagged deeper-depth SM columns into a pixel/date DataFrame.

    Column names in output: SM_20cm, SM_30cm, SM_40cm, SM_50cm.

    Parameters
    ----------
    lag_dict : {depth_col: {"lag": int, ...}}. None → lag 0 for all depths.
    """
    result = df.copy()
    result["_md"] = pd.to_datetime(result["date"]).dt.tz_localize(None).dt.normalize()
    for dcol in depth_cols:
        lag_d = 0 if lag_dict is None else lag_dict.get(dcol, {}).get("lag", 0)
        feat  = f"SM_{dcol}"
        shifted = depth_src[[dcol]].shift(lag_d).rename(columns={dcol: feat}).copy()
        shifted["_md"] = shifted.index
        result = result.merge(shifted[["_md", feat]], on="_md", how="left")
    return result.drop(columns=["_md"])


def build_merged_datasets(lstm_datasets: dict, cnn_datasets: dict,
                          lstm_sat_cols: list, cnn_sat_cols: list,
                          meteo_by_pid: dict, meteo_lag_tables: dict,
                          depth_daily: dict, depth_lag_tables: dict) -> tuple:
    """
    Assemble all five feature configs for LSTM and CNN datasets.

    Returns (lstm_merged, lstm_feat, cnn_merged, cnn_feat) where:
      *_merged : {pid: {config_key: df}}
      *_feat   : {config_key: [feature_col, ...]}
    """
    depth_feat_cols = [f"SM_{d}" for d in DEEP_COLS]
    zero_lag        = {v: {"lag": 0} for v in METEO_KEYS}

    def _build(datasets: dict) -> dict:
        merged = {}
        for pid in PATCH_IDS:
            meteo = meteo_by_pid[pid]
            base  = datasets[pid].copy()
            l0    = merge_meteo_by_lag(base,   zero_lag,             meteo)
            opt   = merge_meteo_by_lag(base,   meteo_lag_tables[pid], meteo)
            d0    = merge_depth_by_lag(opt,    depth_daily[pid])
            dopt  = merge_depth_by_lag(opt,    depth_daily[pid], lag_dict=depth_lag_tables[pid])
            merged[pid] = {
                "sat_only":     base,
                "meteo_lag0":   l0,
                "meteo_optlag": opt,
                "depth_lag0":   d0,
                "depth_optlag": dopt,
            }
        return merged

    lstm_merged = _build(lstm_datasets)
    cnn_merged  = _build(cnn_datasets)

    def _feat(sat: list) -> dict:
        return {
            "sat_only":     sat,
            "meteo_lag0":   sat + METEO_KEYS,
            "meteo_optlag": sat + METEO_KEYS,
            "depth_lag0":   sat + METEO_KEYS + depth_feat_cols,
            "depth_optlag": sat + METEO_KEYS + depth_feat_cols,
        }

    return lstm_merged, _feat(lstm_sat_cols), cnn_merged, _feat(cnn_sat_cols)
