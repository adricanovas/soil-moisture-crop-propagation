#!/usr/bin/env python3
"""
Generate a per-pixel soil-moisture raster using a trained date-grouped CNN.

Run train_raster.py first to produce the model artefacts, then use this
script to apply the model to any Sentinel-2 scene for the same parcel.

Acquisition date is inferred from the raster file automatically:
  - new layout   : filename prefix "<YYYY-MM-DD>_..._c.tif"
  - original layout : sibling "request.json"
  - override     : --date YYYY-MM-DD

Usage
-----
python predict_raster.py --patch P07-A1 \\
    --tiff data/images/P07-A1/2024-08-22_..._c.tif \\
    --output outputs/rasters/sm_P07-A1_2024-08-22.tiff
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from tensorflow.keras.models import load_model

from src.config import METEO_KEYS, PATCH_STATION
from src.data.io       import load_station_meteo, BAND_ORDER, read_bands_12
from src.data.features import compute_spectral_indices, INDEX_ORDER

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")
DEFAULT_METEO_DIR  = os.path.join(os.path.dirname(__file__), "data", "meteo")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _infer_date(tiff_path: Path, override: str | None) -> pd.Timestamp:
    if override:
        return pd.to_datetime(override).tz_localize(None).normalize()
    req = tiff_path.parent / "request.json"
    if req.exists():
        try:
            d = json.loads(req.read_text())
            s = (d["request"]["payload"]["input"]["data"][0]
                   ["dataFilter"]["timeRange"]["from"])
            return pd.to_datetime(s).tz_localize(None).normalize()
        except Exception:
            pass
    stem = tiff_path.name.split("_")[0]
    d    = pd.to_datetime(stem, format="%Y-%m-%d", errors="coerce")
    if pd.isna(d):
        sys.exit("ERROR: could not infer acquisition date; pass --date YYYY-MM-DD.")
    return d.normalize()


def _meteo_values(meteo_daily: pd.DataFrame, date: pd.Timestamp,
                  lag_table: dict, station: str) -> dict:
    out = {}
    for var in METEO_KEYS:
        src_date = date - pd.Timedelta(days=lag_table[var]["lag"])
        if src_date not in meteo_daily.index:
            sys.exit(
                f"ERROR: station {station} has no record for {src_date.date()} "
                f"(needed for {var}). Available: "
                f"{meteo_daily.index.min().date()} → {meteo_daily.index.max().date()}"
            )
        out[var] = float(meteo_daily.at[src_date, var])
    return out


def _save_preview(preds: np.ndarray, valid: np.ndarray, nodata: float,
                  out_png: Path, cmap: str, vmin, vmax, title: str) -> None:
    masked = np.ma.masked_where(~valid | (preds == nodata), preds)
    vp     = preds[valid]
    if vp.size and vmin is None:
        vmin = float(np.nanpercentile(vp, 2))
    if vp.size and vmax is None:
        vmax = float(np.nanpercentile(vp, 98))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.set_facecolor("#dddddd")
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("col"); ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Soil moisture (%)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch",      required=True,
                    help="Parcel id (hex or P0x-A1).")
    ap.add_argument("--tiff",       required=True,
                    help="Path to the Sentinel-2 scene TIFF.")
    ap.add_argument("--output",     required=True,
                    help="Output raster path (.tiff).")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--meteo-dir",  default=DEFAULT_METEO_DIR)
    ap.add_argument("--date",       default=None,
                    help="Acquisition date YYYY-MM-DD (auto-inferred if omitted).")
    ap.add_argument("--nodata",     type=float, default=-9999.0)
    ap.add_argument("--preview",    default=None,
                    help="Path for a PNG preview. Default: same stem as output.")
    ap.add_argument("--cmap",       default="YlGnBu")
    ap.add_argument("--vmin",       type=float, default=None)
    ap.add_argument("--vmax",       type=float, default=None)
    args = ap.parse_args()

    if args.patch not in PATCH_STATION:
        sys.exit(f"ERROR: unknown patch {args.patch!r}")

    # ── Load model artefacts ───────────────────────────────────────────────────
    pdir       = Path(args.models_dir) / args.patch
    model_path  = pdir / f"cnn_dategroup_{args.patch}.keras"
    scaler_path = pdir / f"scaler_dategroup_{args.patch}.joblib"
    meta_path   = pdir / f"metadata_{args.patch}.json"
    tiff_path   = Path(args.tiff)

    for p in (model_path, scaler_path, meta_path, tiff_path):
        if not p.exists():
            sys.exit(f"ERROR: file not found: {p}")

    metadata      = json.loads(meta_path.read_text())
    feature_order = metadata["feature_order"]
    lag_table     = metadata["lag_table"]
    station       = metadata.get("station", PATCH_STATION[args.patch])
    n_features    = len(feature_order)
    print(f"Loaded model: {n_features} features, station={station}, "
          f"test R²={metadata['test_metrics']['r2']:+.4f}")

    model  = load_model(str(model_path), compile=False)
    scaler = joblib.load(str(scaler_path))

    # ── Acquisition date + meteo ───────────────────────────────────────────────
    acq_date = _infer_date(tiff_path, args.date)
    print(f"Acquisition date: {acq_date.date()}")
    meteo       = load_station_meteo(str(Path(args.meteo_dir) / f"{station}.csv"))
    meteo_vals  = _meteo_values(meteo, acq_date, lag_table, station)

    # ── Read raster + compute indices ──────────────────────────────────────────
    with rasterio.open(args.tiff) as src:
        H, W        = src.height, src.width
        bands       = read_bands_12(src)
        src_profile = src.profile.copy()

    stacked = np.stack([bands[b] for b in BAND_ORDER], axis=0)
    valid   = ~(stacked == 0).all(axis=0)

    idx    = compute_spectral_indices(bands)
    layers = {**{b: bands[b].astype(np.float32) for b in BAND_ORDER}, **idx}
    for var in METEO_KEYS:
        layers[var] = np.full((H, W), meteo_vals[var], dtype=np.float32)

    missing = [c for c in feature_order if c not in layers]
    if missing:
        sys.exit(f"ERROR: cannot build features: {missing}")

    # ── Inference ──────────────────────────────────────────────────────────────
    feats  = np.stack([layers[c] for c in feature_order], axis=-1).reshape(H * W, n_features)
    feats  = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    X      = scaler.transform(feats).astype(np.float32).reshape(-1, n_features, 1)
    preds  = model.predict(X, verbose=0).reshape(H, W).astype(np.float32)
    preds[~valid] = args.nodata

    # ── Write GeoTIFF ──────────────────────────────────────────────────────────
    out_profile = src_profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=args.nodata, compress="lzw")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(preds, 1)
        dst.set_band_description(1, "soil_moisture_pct")
    print(f"Wrote raster : {out_path}  (valid pixels {int(valid.sum()):,}/{H*W:,})")

    # ── Preview PNG ────────────────────────────────────────────────────────────
    if args.preview is None:
        preview_path = out_path.with_suffix(".png")
    elif args.preview.lower() == "none":
        preview_path = None
    else:
        preview_path = Path(args.preview)

    if preview_path is not None:
        _save_preview(preds, valid, args.nodata, preview_path,
                      args.cmap, args.vmin, args.vmax,
                      f"Soil moisture\n{args.patch} — {acq_date.date()}")
        print(f"Wrote preview: {preview_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
