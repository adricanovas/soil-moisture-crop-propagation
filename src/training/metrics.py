"""Regression metrics for soil-moisture predictions."""
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(model_name: str, config_name: str, patch_label: str,
                    y_true: np.ndarray, preds: np.ndarray,
                    n_train: int | None, n_features: int) -> dict:
    """
    Compute RMSE, MAE, CVRMSE, R², and Pearson CC.

    Returns a dict suitable for appending to the results CSV and the
    in-memory all_results list.
    """
    rmse   = float(np.sqrt(mean_squared_error(y_true, preds)))
    mae    = float(mean_absolute_error(y_true, preds))
    y_mean = float(np.mean(y_true))
    cvrmse = (rmse / y_mean * 100.0) if y_mean != 0 else float("nan")
    r2     = float(r2_score(y_true, preds))
    cc     = float(pearsonr(y_true, preds)[0]) if len(y_true) > 1 else float("nan")

    print(f"  {patch_label} [{model_name}/{config_name}]: "
          f"RMSE={rmse:.4f}  MAE={mae:.4f}  CVRMSE={cvrmse:.2f}%  "
          f"R²={r2:.4f}  CC={cc:.4f}  (N={len(y_true)})")

    return {
        "Model": model_name, "Patch": patch_label, "Config": config_name,
        "N_train": n_train,  "N_test": len(y_true), "N_features": n_features,
        "RMSE": rmse, "MAE": mae, "CVRMSE": cvrmse, "R2": r2, "CC": cc,
        "y_true": y_true, "preds": preds,
    }


METRIC_COLS = ["Model", "Patch", "Config", "N_train", "N_test",
               "N_features", "RMSE", "MAE", "CVRMSE", "R2", "CC"]
