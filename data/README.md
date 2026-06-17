# Data directory

Place your data files here before running the experiments.

## Directory structure

```
data/
├── aoi/              ← GeoJSON parcel boundaries (included in repo)
├── station/          ← SIAR station coordinates (included in repo)
├── soil_moisture/    ← daily mean SM CSVs (you provide)
├── meteo/            ← SIAR station meteo CSVs (you provide)
└── datasets/         ← pre-built training datasets (you provide or build)
```

---

## Files you need to provide

### `soil_moisture/`

One CSV per parcel — daily mean soil moisture at 5 depths (10–50 cm).

| Filename | Parcel |
|---|---|
| `daily_mean_SoilMoisture_5f182afa1cb289095a59ed80.csv` | P01 |
| `daily_mean_SoilMoisture_64535effe1cf614c1bb1ee1f.csv` | P02 |
| `daily_mean_SoilMoisture_64536060e1cf614c1bb1ee21.csv` | P03 |
| `daily_mean_SoilMoisture_P05-A1.csv` | P05 |
| `daily_mean_SoilMoisture_P06-A1.csv` | P06 |
| `daily_mean_SoilMoisture_P07-A1.csv` | P07 |
| `daily_mean_SoilMoisture_P08-A1.csv` | P08 |

Expected columns: `date, 10cm, 20cm, 30cm, 40cm, 50cm`

---

### `meteo/`

One CSV per SIAR station — daily meteorological variables.

| Filename | Used by |
|---|---|
| `TP73.csv` | P01 |
| `ML21.csv` | P02, P03 |
| `MO41.csv` | P05, P06 |
| `MU31.csv` | P07, P08 |

Expected format: comma-separated, header row with `Date` column (`%Y-%m-%d`),
columns: `Date, TempMean, TempMax, TempMin, Precipitation, Radiation`

---

### `datasets/`

Pre-built per-parcel training datasets.  
Generate them with `build_datasets.py` from raw Sentinel-2 imagery,
or copy them from an existing build.

| Filename | Type | Content |
|---|---|---|
| `CNN_<id>_dataset.csv` | per-pixel | one row per (date × pixel), 12 bands + 12 indices |
| `LSTM_<id>_dataset.csv` | per-date | one row per date, band stats (min/mean/max) |

The `<id>` matches the parcel's dataset id in `config/parcels.py`.

---

## Files included in the repo

### `aoi/`

GeoJSON polygon boundaries for each parcel (P01-A1 through P08-A1).

### `station/`

- `stations.csv` — SIAR station coordinates
- `sensor_coord.csv` — soil-moisture sensor (parcel) coordinates
