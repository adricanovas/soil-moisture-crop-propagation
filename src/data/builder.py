"""
Build CNN (per-pixel) and LSTM (date-aggregated) training datasets from raw
Sentinel-2 scenes and in-situ soil-moisture CSVs.

This is a refactored version of build_all_datasets.py. Key changes vs the
original:
  - Hardcoded absolute paths replaced with CLI arguments.
  - Spectral index computation delegates to src.data.features.
  - IMAGE_DIRS dict is now built at runtime from --images-dir arguments
    instead of being embedded in the source file.

Two image layouts are supported:
  - crop_response : <parcel_dir>/<hash>/crop_response.tiff + request.json  (P01-P03)
  - flat13        : <parcel_dir>/<YYYY-MM-DD>_<id>_<cloud>_c.tif  (P05-P08, 13 bands)

Two soil-moisture layouts are supported:
  - wide6 : date, 10cm, …, 60cm  (original parcels, half-hourly)
  - long5 : device_id, sensor_id, timestamp, v1..v5  (new parcels)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.data.features import compute_spectral_indices, INDEX_ORDER
from src.data.io import BAND_ORDER

# ── Constants ──────────────────────────────────────────────────────────────────
DEPTHS       = ["10cm", "20cm", "30cm", "40cm", "50cm"]
TARGET_DEPTH = "10cm"
MIN_VALID_SM = 1.0   # drop days where 10cm SM < 1% vol (probe off / pre-installation)
B10_IDX_13   = 10    # 0-based index of B10 (cirrus) in 13-band rasters

# Parcel layout metadata: label → (dataset_id, image_layout, sm_layout)
# Image directories are passed at runtime via CLI (--images-dir-original / --images-dir-new).
PARCEL_META = {
    "P01-A1": ("5f182afa1cb289095a59ed80", "crop_response", "wide6"),
    "P02-A1": ("64535effe1cf614c1bb1ee1f", "crop_response", "wide6"),
    "P03-A1": ("64536060e1cf614c1bb1ee21", "crop_response", "wide6"),
    "P05-A1": ("P05-A1", "flat13", "long5"),
    "P06-A1": ("P06-A1", "flat13", "long5"),
    "P07-A1": ("P07-A1", "flat13", "long5"),
    "P08-A1": ("P08-A1", "flat13", "long5"),
}


# ── Soil moisture loading ──────────────────────────────────────────────────────

def _load_sm_daily(label: str, layout: str, sm_dir: Path) -> pd.DataFrame:
    """Return daily-mean SM (5 depths) for one parcel; index = tz-naive dates."""
    path = sm_dir / f"{label}.csv"
    df = pd.read_csv(path)
    if layout == "wide6":
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date").sort_index()
        keep = [c for c in DEPTHS if c in df.columns]
        daily = df[keep].resample("D").mean()
    elif layout == "long5":
        df["date"] = pd.to_datetime(df["timestamp"], utc=True)
        ren = {f"v{i+1}": DEPTHS[i] for i in range(5)}
        df = df.rename(columns=ren).set_index("date").sort_index()
        daily = df[DEPTHS].resample("D").mean()
    else:
        raise ValueError(f"Unknown SM layout: {layout!r}")
    daily = daily.dropna(how="all")
    daily = daily[daily[TARGET_DEPTH] >= MIN_VALID_SM]
    daily.index = daily.index.tz_localize(None).normalize()
    return daily.reset_index().rename(columns={"index": "date"})


# ── Image iteration ────────────────────────────────────────────────────────────

def _iter_images_crop_response(img_dir: Path):
    """Yield (date, bands_12xHxW, transform) for original parcels."""
    for d in sorted(img_dir.iterdir()):
        rj  = d / "request.json"
        tif = d / "crop_response.tiff"
        if not (rj.exists() and tif.exists()):
            continue
        meta = json.loads(rj.read_text())
        raw  = (meta["request"]["payload"]["input"]["data"][0]
                    ["dataFilter"]["timeRange"]["from"])
        date = pd.to_datetime(raw, utc=True).tz_localize(None).normalize()
        with rasterio.open(tif) as src:
            arr = src.read().astype(np.float64)
            if arr.shape[0] != 12:
                print(f"  WARN {tif}: {arr.shape[0]} bands, skipped")
                continue
            yield date, arr, src.transform


def _iter_images_flat13(img_dir: Path):
    """Yield (date, bands_12xHxW, transform) for new parcels (best scene per date)."""
    best: dict = {}
    for tif in sorted(img_dir.glob("*.tif")):
        parts = tif.stem.replace("_c", "").split("_")
        date  = pd.to_datetime(parts[0], format="%Y-%m-%d", errors="coerce")
        if pd.isna(date):
            print(f"  WARN {tif.name}: unparseable date, skipped")
            continue
        try:
            cloud = float(parts[2]) if len(parts) >= 3 else 999.0
        except ValueError:
            cloud = 999.0
        date = date.normalize()
        if date not in best or cloud < best[date][0]:
            best[date] = (cloud, tif)

    for date in sorted(best):
        tif = best[date][1]
        with rasterio.open(tif) as src:
            arr = src.read().astype(np.float64)
            if arr.shape[0] == 13:
                arr = np.delete(arr, B10_IDX_13, axis=0)
            elif arr.shape[0] != 12:
                print(f"  WARN {tif.name}: {arr.shape[0]} bands, skipped")
                continue
            yield date, arr, src.transform


def _iter_images(label: str, layout: str, img_dir: Path):
    if layout == "crop_response":
        yield from _iter_images_crop_response(img_dir / PARCEL_META[label][0])
    elif layout == "flat13":
        yield from _iter_images_flat13(img_dir / label)
    else:
        raise ValueError(f"Unknown image layout: {layout!r}")


# ── Per-parcel dataset builder ─────────────────────────────────────────────────

def build_parcel(label: str, img_dir: Path, sm_dir: Path) -> tuple:
    """
    Build CNN and LSTM datasets for one parcel.

    Returns (cnn_df, lstm_df, n_images, sm_daily_df).
    """
    ds_id, img_layout, sm_layout = PARCEL_META[label]
    sm = _load_sm_daily(label, sm_layout, sm_dir)
    sm_target = sm[["date", TARGET_DEPTH]].rename(columns={TARGET_DEPTH: "soil_moisture"})

    pix_rows, agg_rows, n_imgs = [], [], 0
    for date, arr, transform in _iter_images(label, img_layout, img_dir):
        n_imgs += 1
        _, H, W = arr.shape
        band = {BAND_ORDER[i]: arr[i] for i in range(12)}

        # per-pixel (CNN)
        flat = arr.reshape(12, -1).T
        rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        rr, cc = rr.ravel(), cc.ravel()
        nonzero = np.any(flat != 0, axis=1)
        if nonzero.any():
            sel = flat[nonzero]
            xs, ys = rasterio.transform.xy(transform, rr[nonzero], cc[nonzero])
            dfp = pd.DataFrame(sel, columns=BAND_ORDER)
            dfp.insert(0, "coord_y", np.asarray(ys))
            dfp.insert(0, "coord_x", np.asarray(xs))
            dfp.insert(0, "date", date)
            dfp.insert(0, "colum", cc[nonzero])
            dfp.insert(0, "row", rr[nonzero])
            idx = compute_spectral_indices({b: dfp[b] for b in BAND_ORDER})
            for k in INDEX_ORDER:
                dfp[k] = idx[k]
            pix_rows.append(dfp)

        # aggregate (LSTM)
        agg = {"date": date}
        masked = {b: np.ma.masked_equal(band[b], 0) for b in BAND_ORDER}
        for stat, fn in (("min", np.ma.min), ("mean", np.ma.mean), ("max", np.ma.max)):
            for b in BAND_ORDER:
                agg[f"{b}_{stat}"] = float(fn(masked[b]))
        agg_rows.append(agg)

    # assemble CNN
    cnn = pd.concat(pix_rows, ignore_index=True)
    cnn = cnn.replace([np.inf, -np.inf], np.nan).dropna(subset=BAND_ORDER + INDEX_ORDER)
    cnn = cnn.merge(sm_target, on="date", how="inner")
    cnn = cnn.sort_values(["date", "row", "colum"]).reset_index(drop=True)

    # assemble LSTM
    lstm = pd.DataFrame(agg_rows)
    for stat in ("min", "mean", "max"):
        cols = {b: lstm[f"{b}_{stat}"] for b in BAND_ORDER}
        idx = compute_spectral_indices(cols)
        for k in INDEX_ORDER:
            lstm[f"{k}_{stat}"] = idx[k]
    band_cols = [f"{b}_{s}" for s in ("min","mean","max") for b in BAND_ORDER]
    idx_cols  = [f"{k}_{s}" for k in INDEX_ORDER for s in ("min","mean","max")]
    lstm = lstm[["date"] + band_cols + idx_cols]
    lstm = lstm.replace([np.inf, -np.inf], np.nan).dropna()
    lstm = lstm.merge(sm_target, on="date", how="inner")
    lstm = lstm.sort_values("date").reset_index(drop=True)

    return cnn, lstm, n_imgs, sm


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build CNN / LSTM datasets from raw Sentinel-2 + soil-moisture CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--images-dir-original", required=True,
                    help="Root image dir for P01-P03 (contains <hex_id>/ sub-dirs).")
    ap.add_argument("--images-dir-new", required=True,
                    help="Root image dir for P05-P08 (contains P0x-A1/ sub-dirs).")
    ap.add_argument("--sm-dir", default="data/soil_moisture",
                    help="Directory with <label>.csv soil-moisture files.")
    ap.add_argument("--output-dir", default="data/datasets",
                    help="Where to write CNN_*.csv, LSTM_*.csv, and daily_mean_*.csv.")
    ap.add_argument("--parcels", nargs="*", default=list(PARCEL_META.keys()),
                    help="Subset of parcels to build (default: all seven).")
    args = ap.parse_args()

    sm_dir  = Path(args.sm_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def img_dir_for(label: str) -> Path:
        _, img_layout, _ = PARCEL_META[label]
        return Path(args.images_dir_original if img_layout == "crop_response"
                    else args.images_dir_new)

    summary = []
    for label in args.parcels:
        if label not in PARCEL_META:
            print(f"WARNING: unknown parcel {label!r}, skipping.")
            continue
        ds_id = PARCEL_META[label][0]
        print(f"\n=== {label}  (id={ds_id}) ===")
        cnn, lstm, n_imgs, sm = build_parcel(label, img_dir_for(label), sm_dir)

        cnn.to_csv(out_dir / f"CNN_{ds_id}_dataset.csv", index=False)
        lstm.to_csv(out_dir / f"LSTM_{ds_id}_dataset.csv", index=False)

        sm_daily = sm.copy()
        sm_daily["date"] = pd.to_datetime(sm_daily["date"]).dt.strftime("%Y-%m-%d")
        sm_daily.to_csv(out_dir / f"daily_mean_SoilMoisture_{ds_id}.csv",
                        index=False, float_format="%.3f")

        d0, d1 = cnn["date"].min(), cnn["date"].max()
        print(f"  images: {n_imgs}")
        print(f"  CNN  : {len(cnn):,} px-rows × {cnn.shape[1]} cols  "
              f"({d0.date()} → {d1.date()}, {cnn['date'].nunique()} dates)")
        print(f"  LSTM : {len(lstm):,} date-rows × {lstm.shape[1]} cols  "
              f"({lstm['date'].nunique()} dates)")
        summary.append(dict(parcel=label, ds_id=ds_id, images=n_imgs,
                            cnn_rows=len(cnn), lstm_rows=len(lstm),
                            date_min=str(d0.date()), date_max=str(d1.date())))

    pd.DataFrame(summary).to_csv(out_dir / "_build_summary.csv", index=False)
    print(f"\nWrote {len(summary)} parcel(s) → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
