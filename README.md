# Soil Moisture Estimation from Sentinel‑2

This repository contains the code and experiment suite accompanying the paper  
**“Vertical Soil‑Moisture Estimation from Sentinel‑2 Spectral Indices and Meteorological Forcing.”**

We evaluate deep‑learning models for predicting soil moisture at 5 depths (10–50 cm) across 7 agricultural parcels, using Sentinel‑2 spectral indices and SIAR meteorological data.  
A key contribution of this work is the **leak‑free temporal evaluation protocol**, which prevents the inflation of R² observed under pixel‑level random splits.

---

## 1. Models evaluated

| Model | Summary |
|------|---------|
| **LSTM** | Bidirectional LSTM on date‑aggregated spectral statistics |
| **CNN** | 1‑D CNN on per‑pixel spectral features (baseline) |
| **CNN‑DateGroup** | CNN with date‑grouped split (leak‑free evaluation) |
| **CNN–LSTM Hybrid** | CNN feature extractor + LSTM temporal encoder |

---

## 2. Feature configurations

| Config | Description |
|--------|-------------|
| `sat_only` | Sentinel‑2 spectral indices |
| `sat+meteo_lag0` | + meteorological variables (lag 0) |
| `sat+meteo_optlag` | + meteorological variables at optimal CCF lag |
| `sat+depth_lag0` | + deeper soil‑moisture depths (lag 0) |
| `sat+depth_optlag` | + deeper depths at optimal CCF lag |

A total of **140 experiments** were run: 4 models × 7 parcels × 5 configurations.

---

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
