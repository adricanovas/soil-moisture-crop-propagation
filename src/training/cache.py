"""
Disk-based results cache for the experiment runner.

Completed (Model, Patch, Config) combinations are persisted to a CSV and
companion .npz files so long experiments can be interrupted and resumed
without re-running finished jobs.
"""
import os
import numpy as np
import pandas as pd

from src.training.metrics import METRIC_COLS


def _pred_path(predictions_dir: str, model: str, patch: str, config: str) -> str:
    safe = f"{model}__{patch}__{config}".replace(" ", "_").replace("/", "_").replace("+", "_")
    return os.path.join(predictions_dir, f"{safe}.npz")


def save_result(results_csv: str, predictions_dir: str, result: dict) -> None:
    """Append one result row to the CSV and save predictions to .npz."""
    pred_file = _pred_path(predictions_dir, result["Model"], result["Patch"], result["Config"])
    np.savez_compressed(pred_file, y_true=result["y_true"], preds=result["preds"])
    row_df = pd.DataFrame([{k: result[k] for k in METRIC_COLS}])
    row_df.to_csv(results_csv,
                  mode="a", header=not os.path.exists(results_csv), index=False)


def load_cached_result(results_csv: str, predictions_dir: str,
                       model: str, patch: str, config: str) -> dict | None:
    """
    Return cached result dict (including y_true, preds arrays) or None.

    Returns None if the (model, patch, config) combination is not yet in
    the results CSV.
    """
    if not os.path.exists(results_csv):
        return None
    cached = pd.read_csv(results_csv)
    match  = cached[(cached["Model"] == model) & (cached["Patch"] == patch) &
                    (cached["Config"] == config)]
    if match.empty:
        return None
    row = match.iloc[0].to_dict()
    pred_file = _pred_path(predictions_dir, model, patch, config)
    if os.path.exists(pred_file):
        data = np.load(pred_file)
        row["y_true"] = data["y_true"]
        row["preds"]  = data["preds"]
    else:
        row["y_true"] = np.array([])
        row["preds"]  = np.array([])
    return row
