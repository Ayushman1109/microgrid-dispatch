"""
Feature engineering shared by training (QuickML dataset) and inference.

Pure Python (no pandas) so it runs inside Catalyst AppSail without heavy
native dependencies. predict.py carries an identical copy of these rules;
tests/test_consistency.py checks the two never drift apart.

Model insight (see model/validate_model.py):
    demand ~= schedule[hour] + COOLING_SLOPE * max(0, temp - 24 C) + noise
The daily schedule is captured by one-hot hour columns, the cooling load by
`cooling_deg_c`. No lag/rolling features are used, so every row can be
predicted independently and row order never matters.
"""
from datetime import datetime

COOLING_BASE_C = 24.0

# Hour 0 is the reference category (dropped) so a linear model with an
# intercept has no perfectly collinear columns.
HOUR_COLUMNS = [f"hour_{h:02d}" for h in range(1, 24)]
WEATHER_COLUMNS = ["ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]
FEATURE_COLUMNS = HOUR_COLUMNS + ["cooling_deg_c"] + WEATHER_COLUMNS

TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%Y/%m/%d %H:%M:%S",
)


def parse_timestamp(value):
    """Return a datetime or None. Accepts the common formats people paste."""
    if value is None:
        return None
    text = str(value).strip().replace("Z", "")
    if not text or text.lower() == "nan":
        return None
    if "." in text:                                   # drop fractional seconds
        text = text.split(".")[0]
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def build_feature_row(hour, temp_c, humidity_pct, solar_wm2):
    """One model-ready feature dict. Inputs must already be cleaned numbers."""
    row = {col: 0 for col in HOUR_COLUMNS}
    if hour is not None and 1 <= int(hour) <= 23:
        row[f"hour_{int(hour):02d}"] = 1
    row["cooling_deg_c"] = round(max(0.0, float(temp_c) - COOLING_BASE_C), 4)
    row["ambient_temp_c"] = float(temp_c)
    row["humidity_pct"] = float(humidity_pct)
    row["solar_irradiance_wm2"] = float(solar_wm2)
    return row
