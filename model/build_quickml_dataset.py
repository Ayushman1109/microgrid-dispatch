"""
Build the CSV you upload to Zoho Catalyst QuickML.

    python model/build_quickml_dataset.py

Reads  model/train_telemetry.csv  (from the starter kit)
Writes model/quickml_train.csv    (engineered features + target_demand_kw)
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "appsail"))
from core.features import FEATURE_COLUMNS, build_feature_row, parse_timestamp  # noqa: E402

src = os.path.join(HERE, "train_telemetry.csv")
dst = os.path.join(HERE, "quickml_train.csv")

with open(src, newline="") as f_in, open(dst, "w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=FEATURE_COLUMNS + ["target_demand_kw"])
    writer.writeheader()
    n = 0
    for r in reader:
        ts = parse_timestamp(r["timestamp"])
        row = build_feature_row(ts.hour, r["ambient_temp_c"], r["humidity_pct"],
                                r["solar_irradiance_wm2"])
        row["target_demand_kw"] = float(r["target_demand_kw"])
        writer.writerow(row)
        n += 1
print(f"Wrote {n} rows x {len(FEATURE_COLUMNS)} features to {dst}")
