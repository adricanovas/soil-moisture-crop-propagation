#!/usr/bin/env python3
"""
Build CNN (per-pixel) and LSTM (date-aggregated) training datasets.

This is the CLI entry point that delegates to src/data/builder.py.

Usage
-----
python build_datasets.py \\
    --images-dir-original /path/to/original_images \\
    --images-dir-new      /path/to/new_images \\
    --sm-dir              data/soil_moisture \\
    --output-dir          data/datasets

The script also writes daily_mean_SoilMoisture_<id>.csv files into
--output-dir for use by the vertical-lag depth experiment in train.py.

Image directory layout
----------------------
  original (P01-P03): <images-dir-original>/<hex_id>/<hash>/crop_response.tiff
                       + <hash>/request.json
  new      (P05-P08): <images-dir-new>/P0x-A1/<YYYY-MM-DD>_<id>_<cloud>_c.tif
"""
from src.data.builder import main

if __name__ == "__main__":
    raise SystemExit(main())
