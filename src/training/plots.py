"""
Visualization for the soil-moisture experiments.

All plotting functions save to a figures directory and close the figure to
avoid memory leaks in long experiment runs.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    PATCH_IDS, PATCH_LABELS, PATCH_STATION,
    DEPTH_COLS, DEEP_COLS, DEPTH_VALUES,
)

MODEL_LIST = ["LSTM", "CNN", "CNN-DateGroup", "CNN-LSTM"]
CFG_ORDER  = [
    "sat_only", "sat+meteo_lag0", "sat+meteo_optlag",
    "sat+meteo+depth_lag0", "sat+meteo+depth_optlag",
]
CFG_COLORS = dict(zip(CFG_ORDER, ["#4c72b0","#ff7f0e","#dd8452","#55a868","#c44e52"]))


def _grid(n: int, ncols: int = 4) -> tuple:
    return int(np.ceil(n / ncols)), ncols


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Data diagnostic plots ──────────────────────────────────────────────────────

def plot_depth_profiles(depth_daily: dict, fig_dir: str) -> None:
    """Time-series of daily mean SM at all 5 depths, one subplot per parcel."""
    n = len(PATCH_IDS)
    nrows, ncols = _grid(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             constrained_layout=True, squeeze=False)
    fig.suptitle(f"Daily Mean Soil Moisture at {len(DEPTH_COLS)} Depths", fontsize=14)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DEPTH_COLS)))
    for idx, pid in enumerate(PATCH_IDS):
        ax = axes[idx // ncols, idx % ncols]
        df = depth_daily[pid]
        for j, col in enumerate(DEPTH_COLS):
            ax.plot(df.index, df[col], label=col, color=colors[j], lw=0.9, alpha=0.85)
        ax.set_title(f"{PATCH_LABELS[pid]} ({PATCH_STATION[pid]})", fontsize=11)
        ax.set_xlabel("Date"); ax.set_ylabel("Soil Moisture (%)")
        ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    _save(fig, os.path.join(fig_dir, "depth_profiles.png"))


def plot_depth_ccf(curves: dict, tables: dict, fig_dir: str) -> None:
    """Bar charts of CCF(SM_10cm, SM_z) at all lags, one panel per (parcel, depth)."""
    n, m = len(PATCH_IDS), len(DEEP_COLS)
    fig, axes = plt.subplots(n, m, figsize=(4 * m, 2.6 * n),
                             constrained_layout=True, sharey="row", squeeze=False)
    for i, pid in enumerate(PATCH_IDS):
        for j, dcol in enumerate(DEEP_COLS):
            ax   = axes[i, j]
            cors = curves[pid][dcol]
            ax.bar(np.arange(len(cors)), cors, color="mediumseagreen", alpha=0.7)
            best = tables[pid][dcol]["lag"]
            ax.axvline(best, color="darkred", ls="--", lw=1.2, label=f"opt={best}d")
            ax.set_title(f"{PATCH_LABELS[pid]} — 10 cm vs {dcol}", fontsize=9)
            ax.set_xlabel("Lag (days)", fontsize=8)
            if j == 0:
                ax.set_ylabel("Pearson r", fontsize=8)
            ax.legend(fontsize=7)
    fig.suptitle("Inter-Depth CCF: SM at 10 cm vs Deeper Layers", fontsize=13, y=1.005)
    _save(fig, os.path.join(fig_dir, "depth_ccf_curves.png"))


def plot_vertical_lag_profile(tables: dict, fig_dir: str) -> None:
    """Optimal lag and |r| as a function of depth, one line per parcel."""
    markers = ["o","s","^","D","v","P","*","X","<",">"]
    colors  = plt.cm.tab10(np.linspace(0, 1, len(PATCH_IDS)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for idx, pid in enumerate(PATCH_IDS):
        lags  = [tables[pid][f"{d}cm"]["lag"]          for d in DEPTH_VALUES]
        corrs = [abs(tables[pid][f"{d}cm"]["corr"])    for d in DEPTH_VALUES]
        mk, cl = markers[idx % len(markers)], colors[idx]
        ax1.plot(DEPTH_VALUES, lags,  marker=mk, lw=1.5, label=PATCH_LABELS[pid], ms=7, color=cl)
        ax2.plot(DEPTH_VALUES, corrs, marker=mk, lw=1.5, label=PATCH_LABELS[pid], ms=7, color=cl)
    for ax, ylabel, title in [
        (ax1, "Optimal Lag (days)",  "Optimal Lag vs Depth"),
        (ax2, "|Pearson r|",         "Correlation Strength at Optimal Lag"),
    ]:
        ax.set_xlabel("Depth (cm)"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_xticks(DEPTH_VALUES)
    ax2.set_ylim(0, 1.05)
    _save(fig, os.path.join(fig_dir, "vertical_lag_profile.png"))


# ── Model performance plots ────────────────────────────────────────────────────

def plot_bar_charts(results_df: pd.DataFrame, fig_dir: str) -> None:
    """Per-model grouped bar charts: metric × patch × config."""
    metrics = ["RMSE", "CVRMSE", "R2", "CC"]
    for model_name in MODEL_LIST:
        sub = results_df[(results_df["Model"] == model_name) &
                         (results_df["Patch"] != "Pooled")]
        if sub.empty:
            continue
        patches = sorted(sub["Patch"].unique())
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5.5),
                                 constrained_layout=True)
        for ax, met in zip(axes, metrics):
            x = np.arange(len(patches)); n_c = len(CFG_ORDER); w = 0.8 / n_c
            for k, cfg in enumerate(CFG_ORDER):
                sub_cfg = sub[sub["Config"] == cfg]
                vals    = [sub_cfg[sub_cfg["Patch"] == p][met].values[0]
                           if len(sub_cfg[sub_cfg["Patch"] == p]) else np.nan
                           for p in patches]
                ax.bar(x + k * w, vals, w, label=cfg,
                       color=CFG_COLORS[cfg], alpha=0.85)
            ax.set_xticks(x + w * (n_c - 1) / 2)
            ax.set_xticklabels(patches, fontsize=9)
            ax.set_ylabel(met); ax.set_title(met)
            ax.legend(fontsize=5.5, loc="best")
            if met == "R2":
                ax.axhline(0, color="k", lw=0.8, ls="--")
        fig.suptitle(f"{model_name} — Performance by Config (per Patch)", fontsize=13)
        _save(fig, os.path.join(fig_dir, f"bar_{model_name}.png"))


def plot_cross_model(results_df: pd.DataFrame, fig_dir: str,
                     best_cfg: str = "sat+meteo+depth_optlag") -> None:
    """Cross-model comparison on the best config, one bar per (model, patch)."""
    metrics = ["RMSE", "CVRMSE", "R2", "CC"]
    colors  = dict(zip(MODEL_LIST, plt.cm.Set2(np.linspace(0, 1, len(MODEL_LIST)))))
    df_cmp  = results_df[(results_df["Config"] == best_cfg) &
                         (results_df["Patch"] != "Pooled")].copy()
    patches = sorted(df_cmp["Patch"].unique())
    if not patches:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5.5),
                             constrained_layout=True)
    for ax, met in zip(axes, metrics):
        x = np.arange(len(patches)); n_m = len(MODEL_LIST); w = 0.8 / n_m
        for k, mdl in enumerate(MODEL_LIST):
            sub  = df_cmp[df_cmp["Model"] == mdl]
            vals = [sub[sub["Patch"] == p][met].values[0]
                    if len(sub[sub["Patch"] == p]) else np.nan
                    for p in patches]
            ax.bar(x + k * w, vals, w, label=mdl, color=colors[mdl], alpha=0.85)
        ax.set_xticks(x + w * (n_m - 1) / 2)
        ax.set_xticklabels(patches, fontsize=9)
        ax.set_ylabel(met); ax.set_title(met); ax.legend(fontsize=7)
        if met == "R2":
            ax.axhline(0, color="k", lw=0.8, ls="--")
    fig.suptitle(f"Cross-Model Comparison — {best_cfg}", fontsize=14)
    _save(fig, os.path.join(fig_dir, "cross_model_comparison.png"))


def plot_leakage_gap(results_df: pd.DataFrame, fig_dir: str,
                     cfg: str = "sat+meteo_optlag") -> None:
    """CNN vs CNN-DateGroup R² per patch: visualises the leakage magnitude."""
    a = results_df[(results_df["Model"] == "CNN") & (results_df["Config"] == cfg)]
    b = results_df[(results_df["Model"] == "CNN-DateGroup") & (results_df["Config"] == cfg)]
    patches = sorted(set(a["Patch"]) & set(b["Patch"]) - {"Pooled"})
    if not patches:
        return
    fig, ax = plt.subplots(figsize=(1.4 * len(patches) + 3, 5), constrained_layout=True)
    x = np.arange(len(patches)); w = 0.38
    va = [a[a["Patch"] == p]["R2"].values[0] for p in patches]
    vb = [b[b["Patch"] == p]["R2"].values[0] for p in patches]
    ax.bar(x - w / 2, va, w, label="CNN (random split)",        color="#c44e52", alpha=0.85)
    ax.bar(x + w / 2, vb, w, label="CNN-DateGroup (date split)", color="#4c72b0", alpha=0.85)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(patches)
    ax.set_ylabel("R²")
    ax.set_title(f"Date-group leakage: CNN vs CNN-DateGroup ({cfg})")
    ax.legend()
    _save(fig, os.path.join(fig_dir, "leakage_gap.png"))


def plot_scatter_pred_vs_actual(results_df: pd.DataFrame, fig_dir: str,
                                plot_cfg: str = "sat+meteo+depth_optlag") -> None:
    """Predicted vs actual scatter plots for each model × parcel."""
    for model_name in MODEL_LIST:
        sub = results_df[(results_df["Model"] == model_name) &
                         (results_df["Config"] == plot_cfg) &
                         (results_df["Patch"] != "Pooled")].reset_index(drop=True)
        sub = sub[sub["y_true"].apply(lambda a: hasattr(a, "__len__") and len(a) > 0)]
        if sub.empty:
            continue
        ncols = 3
        nrows = int(np.ceil(len(sub) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                                 constrained_layout=True, squeeze=False)
        fig.suptitle(f"{model_name} — Predicted vs Actual ({plot_cfg})", fontsize=13)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax     = axes[i // ncols, i % ncols]
            yt, yp = row["y_true"], row["preds"]
            ax.scatter(yt, yp, alpha=0.5, s=18, color="steelblue")
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlabel("Actual SM (%)"); ax.set_ylabel("Predicted SM (%)")
            ax.set_title(f"{row['Patch']}  R²={row['R2']:.3f}  RMSE={row['RMSE']:.2f}")
        for idx in range(len(sub), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)
        _save(fig, os.path.join(fig_dir, f"scatter_{model_name}.png"))
