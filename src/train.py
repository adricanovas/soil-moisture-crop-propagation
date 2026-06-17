#!/usr/bin/env python3
"""
Main experiment: 4 models × 7 parcels × 5 feature configs.

Models  : LSTM, CNN, CNN-DateGroup (leakage-free), CNN-LSTM Hybrid
Configs : sat_only, sat+meteo_lag0, sat+meteo_optlag,
          sat+meteo+depth_lag0, sat+meteo+depth_optlag

Results are cached by (Model, Patch, Config) so the script can be
interrupted and resumed without re-running completed combinations.

Usage
-----
python train.py                                    # all 4 models, all 5 configs
python train.py --models CNN CNN-DateGroup         # subset of models
python train.py --configs sat_only sat+meteo_optlag
python train.py --data-dir /path/to/data --output-dir /path/to/outputs
"""
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import tensorflow as tf

from src.config import (
    PATCH_IDS, PATCH_LABELS,
    CONFIGS, CONFIG_KEYS, CONFIG_LABELS,
)
from src.data.io import (
    load_lstm_datasets, load_cnn_datasets,
    load_meteo_by_patch, load_depth_data,
)
from src.training.ccf     import compute_meteo_ccf, compute_depth_ccf
from src.training.fusion  import build_merged_datasets
from src.training.metrics import METRIC_COLS
from src.training.cache   import save_result, load_cached_result
from src.training.trainers import (
    train_lstm, train_cnn, train_cnn_dategroup, train_hybrid,
)
from src.training.plots import (
    plot_depth_profiles, plot_depth_ccf, plot_vertical_lag_profile,
    plot_bar_charts, plot_cross_model, plot_leakage_gap,
    plot_scatter_pred_vs_actual, MODEL_LIST,
)

DEFAULT_DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",   default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=None,
                   choices=MODEL_LIST,
                   help="Run only these models (default: all four).")
    p.add_argument("--configs", nargs="+", default=None,
                   choices=CONFIG_LABELS,
                   help="Run only these feature configs (default: all five).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir   = args.data_dir
    output_dir = args.output_dir

    dataset_dir = os.path.join(data_dir, "datasets")
    sm_dir      = os.path.join(data_dir, "soil_moisture")
    meteo_dir   = os.path.join(data_dir, "meteo")

    results_csv      = os.path.join(output_dir, "results", "results.csv")
    predictions_dir  = os.path.join(output_dir, "predictions")
    fig_dir          = os.path.join(output_dir, "figures")
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    os.makedirs(predictions_dir,              exist_ok=True)
    os.makedirs(fig_dir,                      exist_ok=True)

    print(f"TensorFlow {tf.__version__}  | GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"Data dir   : {os.path.abspath(data_dir)}")
    print(f"Output dir : {os.path.abspath(output_dir)}")
    if os.path.exists(results_csv):
        n_cached = len(pd.read_csv(results_csv))
        print(f"Found {n_cached} cached results — those will be skipped.")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n--- Loading LSTM datasets ---")
    lstm_datasets, lstm_sat_cols = load_lstm_datasets(dataset_dir)
    print("\n--- Loading CNN datasets ---")
    cnn_datasets,  cnn_sat_cols  = load_cnn_datasets(dataset_dir)
    print("\n--- Loading per-parcel meteo ---")
    meteo_by_pid = load_meteo_by_patch(meteo_dir)
    print("\n--- Loading multi-depth soil moisture ---")
    depth_daily  = load_depth_data(sm_dir)

    # ── Diagnostics ────────────────────────────────────────────────────────────
    print("\n--- Plotting depth profiles ---")
    plot_depth_profiles(depth_daily, fig_dir)

    print("\n--- Meteorological CCF ---")
    meteo_lag_tables = compute_meteo_ccf(lstm_datasets, meteo_by_pid)

    print("\n--- Inter-depth CCF ---")
    depth_lag_tables, depth_ccf_curves = compute_depth_ccf(depth_daily)

    print("\n--- Plotting CCF curves ---")
    plot_depth_ccf(depth_ccf_curves, depth_lag_tables, fig_dir)
    plot_vertical_lag_profile(depth_lag_tables, fig_dir)

    # ── Feature fusion ─────────────────────────────────────────────────────────
    print("\n--- Building merged datasets ---")
    lstm_merged, lstm_feat, cnn_merged, cnn_feat = build_merged_datasets(
        lstm_datasets, cnn_datasets, lstm_sat_cols, cnn_sat_cols,
        meteo_by_pid, meteo_lag_tables, depth_daily, depth_lag_tables)

    # ── Filter requested configs ───────────────────────────────────────────────
    if args.configs is not None:
        req_labels = set(args.configs)
        config_keys = [ck for ck in CONFIG_KEYS if CONFIGS[ck] in req_labels]
    else:
        config_keys = CONFIG_KEYS

    models = args.models if args.models is not None else MODEL_LIST
    print(f"\n*** Models: {models} | "
          f"Configs: {[CONFIGS[ck] for ck in config_keys]} ***")

    all_results = []

    # ── Per-patch trainers (LSTM, CNN, CNN-DateGroup) ──────────────────────────
    def _run_per_patch(model_name: str, trainer, merged: dict, feat: dict) -> None:
        print("\n" + "=" * 65 + f"\n  {model_name} TRAINING\n" + "=" * 65)
        for pid in PATCH_IDS:
            label = PATCH_LABELS[pid]
            print(f"\n--- {label} ---")
            for ck in config_keys:
                cfg = CONFIGS[ck]
                cached = load_cached_result(results_csv, predictions_dir,
                                            model_name, label, cfg)
                if cached is not None:
                    print(f"  {label} [{model_name}/{cfg}]: CACHED "
                          f"(RMSE={cached['RMSE']:.4f}  R²={cached['R2']:.4f})")
                    all_results.append(cached)
                    continue
                res = trainer(merged[pid][ck], feat[ck], cfg, label)
                if res:
                    save_result(results_csv, predictions_dir, res)
                    all_results.append(res)

    if "LSTM" in models:
        _run_per_patch("LSTM", train_lstm, lstm_merged, lstm_feat)
    if "CNN" in models:
        _run_per_patch("CNN", train_cnn, cnn_merged, cnn_feat)
    if "CNN-DateGroup" in models:
        _run_per_patch("CNN-DateGroup", train_cnn_dategroup, cnn_merged, cnn_feat)

    # ── CNN-LSTM Hybrid (pooled across parcels) ────────────────────────────────
    if "CNN-LSTM" in models:
        print("\n" + "=" * 65 + "\n  CNN-LSTM HYBRID TRAINING\n" + "=" * 65)
        for ck in config_keys:
            cfg = CONFIGS[ck]
            print(f"\n--- Config: {cfg} ---")
            cached_pooled = load_cached_result(results_csv, predictions_dir,
                                               "CNN-LSTM", "Pooled", cfg)
            if cached_pooled is not None:
                all_results.append(cached_pooled)
                for pid in PATCH_IDS:
                    c = load_cached_result(results_csv, predictions_dir,
                                           "CNN-LSTM", PATCH_LABELS[pid], cfg)
                    if c is not None:
                        all_results.append(c)
                continue
            ds_map = {pid: lstm_merged[pid][ck] for pid in PATCH_IDS}
            overall, per_patch = train_hybrid(ds_map, lstm_feat[ck], cfg)
            if overall:
                save_result(results_csv, predictions_dir, overall)
                all_results.append(overall)
                for pr in per_patch:
                    save_result(results_csv, predictions_dir, pr)
                all_results.extend(per_patch)

    # ── Print summary tables ───────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    for model_name in MODEL_LIST:
        sub = results_df[results_df["Model"] == model_name]
        if sub.empty:
            continue
        print(f"\n=== {model_name} ===")
        print(sub[METRIC_COLS].round(4).to_string(index=False))

    # ── Generate figures ───────────────────────────────────────────────────────
    print("\n--- Generating result plots ---")
    plot_bar_charts(results_df, fig_dir)
    plot_cross_model(results_df, fig_dir)
    plot_leakage_gap(results_df, fig_dir)
    plot_scatter_pred_vs_actual(results_df, fig_dir)

    print(f"\nResults : {os.path.abspath(results_csv)}")
    print(f"Figures : {os.path.abspath(fig_dir)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
