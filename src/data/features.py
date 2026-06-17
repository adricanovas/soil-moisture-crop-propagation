"""
Sentinel-2 spectral index computation.

Unifies the two implementations that previously existed in:
  - build_all_datasets.py  (_indices_from_bands, plain division, Series/arrays)
  - predict_raster_dategroup.py  (compute_indices + safe_div, 2-D raster arrays)

A single compute_spectral_indices() works with any array shape (scalars, 1-D
pixel vectors, 2-D spatial rasters) by relying on numpy broadcasting.
"""
import numpy as np

INDEX_ORDER = ["NDVI","ARVI","GCI","GNDVI","NBR2","NDBI","NDMI","NDRE","NDSI","NDWI","REIP","SAVI"]


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Element-wise division; returns NaN where denominator is zero."""
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    out = np.full_like(num, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=(den != 0))
    return out


def compute_spectral_indices(bands: dict) -> dict:
    """
    Compute the 12 spectral indices used in this study.

    Parameters
    ----------
    bands : dict
        Mapping of band name → array-like (any shape).
        Required keys: B02, B03, B04, B05, B08, B11. Others are ignored.

    Returns
    -------
    dict
        {index_name: array} with the same shape as the input arrays.
        Keys follow INDEX_ORDER.
    """
    B02 = np.asarray(bands["B02"], dtype=np.float32)
    B03 = np.asarray(bands["B03"], dtype=np.float32)
    B04 = np.asarray(bands["B04"], dtype=np.float32)
    B05 = np.asarray(bands["B05"], dtype=np.float32)
    B08 = np.asarray(bands["B08"], dtype=np.float32)
    B11 = np.asarray(bands["B11"], dtype=np.float32)
    L = np.float32(0.5)

    return {
        "NDVI":  _safe_div(B08 - B04,           B08 + B04),
        "ARVI":  _safe_div(B08 - (B04 - B02),   B08 + (B04 - B02)),
        "GCI":   _safe_div(B08,                  B03) - 1.0,
        "GNDVI": _safe_div(B08 - B03,            B08 + B03),
        "NBR2":  _safe_div(B08 - B11,            B08 + B11),
        "NDBI":  _safe_div(B11 - B08,            B11 + B08),
        "NDMI":  _safe_div(B08 - B11,            B08 + B11),
        "NDRE":  _safe_div(B08 - B05,            B08 + B05),
        "NDSI":  _safe_div(B03 - B11,            B03 + B11),
        "NDWI":  _safe_div(B03 - B08,            B03 + B08),
        "REIP":  _safe_div(B08,                  B05),
        "SAVI":  _safe_div(B08 - B04, B08 + B04 + L) * (1.0 + L),
    }
