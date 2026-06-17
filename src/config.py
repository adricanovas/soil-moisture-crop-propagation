"""
Single source of truth: parcel registry, depth config, meteo keys, and all
hyperparameters used across training scripts.

Changing a value here propagates automatically to every script that imports it.
"""

# ── Parcel registry ────────────────────────────────────────────────────────────
# (dataset_id, short_label, nearest_SIAR_station)
# P04 is excluded: missing reliable soil-moisture data.
PARCELS = [
    ("5f182afa1cb289095a59ed80", "P01", "TP73"),
    ("64535effe1cf614c1bb1ee1f", "P02", "ML21"),   # ML12 nearest but lacks Radiation before 2023-10
    ("64536060e1cf614c1bb1ee21", "P03", "ML21"),
    ("P05-A1",                   "P05", "MO41"),
    ("P06-A1",                   "P06", "MO41"),
    ("P07-A1",                   "P07", "MU31"),
    ("P08-A1",                   "P08", "MU31"),
]

PATCH_IDS     = [p[0] for p in PARCELS]
PATCH_LABELS  = {p[0]: p[1] for p in PARCELS}
PATCH_STATION = {p[0]: p[2] for p in PARCELS}

# ── Depth configuration ────────────────────────────────────────────────────────
DEPTH_COLS   = ["10cm", "20cm", "30cm", "40cm", "50cm"]
DEEP_COLS    = ["20cm", "30cm", "40cm", "50cm"]   # depths below the 10cm target
DEPTH_VALUES = [20, 30, 40, 50]                   # numeric counterparts of DEEP_COLS

# ── Meteorological variables ───────────────────────────────────────────────────
METEO_KEYS = ["TempMean", "TempMax", "TempMin", "Precipitation", "Radiation"]

# ── Model hyperparameters ──────────────────────────────────────────────────────
LSTM_EPOCHS   = 900
LSTM_BATCH    = 64
LSTM_LR       = 0.001

CNN_EPOCHS    = 200
CNN_BATCH     = 64

HYBRID_EPOCHS = 300
HYBRID_BATCH  = 32
HYBRID_LR     = 0.001

# ── Experiment settings ────────────────────────────────────────────────────────
MAX_LAG_DAYS_METEO = 30
MAX_LAG_DAYS_DEPTH = 15
WINDOW_SIZE        = 10   # sliding-window length for CNN-LSTM hybrid
STRIDE             = 1

TEST_SIZE    = 0.2
RANDOM_STATE = 143

# ── Experiment configs (key → display label) ───────────────────────────────────
CONFIGS = {
    "sat_only":      "sat_only",
    "meteo_lag0":    "sat+meteo_lag0",
    "meteo_optlag":  "sat+meteo_optlag",
    "depth_lag0":    "sat+meteo+depth_lag0",
    "depth_optlag":  "sat+meteo+depth_optlag",
}
CONFIG_KEYS   = list(CONFIGS.keys())
CONFIG_LABELS = list(CONFIGS.values())
