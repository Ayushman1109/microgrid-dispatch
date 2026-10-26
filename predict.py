"""
HackSphere 3.0 - demand forecast inference script.

    python predict.py --input sample_test_input.csv --output predictions.csv

Contract: predict(input_df) returns one float per input row, same order,
never NaN/inf. Input columns: timestamp, ambient_temp_c, humidity_pct,
solar_irradiance_wm2.

How predictions are obtained (first that works wins):
  1. PROXY  - POST to our deployed Catalyst AppSail app (/api/predict-batch),
              which calls the published QuickML endpoint server-side. Needs no
              credentials on the evaluator's machine.
  2. DIRECT - call the QuickML endpoint over REST with a Zoho OAuth refresh
              token (set the env vars listed in README.md).
  3. LOCAL  - offline copy of the same linear model, used ONLY if both calls
              fail, so the script never crashes. Prints a loud warning.

Feature engineering here is identical to appsail/core/features.py
(tests/test_consistency.py enforces this).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# Configure after deployment (or set the matching environment variables).
# --------------------------------------------------------------------------
PREDICT_PROXY_URL = os.environ.get(
    "PREDICT_PROXY_URL", "https://microgrid-dispatch-50047177630.development.catalystappsail.in")
QUICKML_ENDPOINT_URL = os.environ.get(
    "QUICKML_ENDPOINT_URL", "YOUR_PUBLISHED_QUICKML_ENDPOINT_URL_HERE")
ALLOW_LOCAL_FALLBACK = os.environ.get("ALLOW_LOCAL_FALLBACK", "true").lower() in ("1", "true", "yes")
PROXY_CHUNK = 200

# --------------------------------------------------------------------------
# Feature engineering (must match training: model/build_quickml_dataset.py)
# --------------------------------------------------------------------------
COOLING_BASE_C = 24.0
HOUR_COLUMNS = [f"hour_{h:02d}" for h in range(1, 24)]
WEATHER_COLUMNS = ["ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]
FEATURE_COLUMNS = HOUR_COLUMNS + ["cooling_deg_c"] + WEATHER_COLUMNS
LIMITS = {"ambient_temp_c": (-40.0, 60.0), "humidity_pct": (0.0, 100.0),
          "solar_irradiance_wm2": (0.0, 1500.0)}
DEFAULTS = {"ambient_temp_c": 18.0, "humidity_pct": 70.0, "solar_irradiance_wm2": 0.0}


def clean_inputs(input_df: pd.DataFrame) -> pd.DataFrame:
    """Order-preserving cleanup: parse, validate, fill gaps. Never drops rows."""
    df = pd.DataFrame(index=range(len(input_df)))
    src = input_df.reset_index(drop=True)
    df["timestamp"] = src["timestamp"] if "timestamp" in src else pd.Series([None] * len(src))
    ts = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    hour = ts.dt.hour.astype("float")
    # unreadable timestamps: infer the hour from neighbours (hourly data)
    if hour.isna().any():
        idx = pd.Series(np.arange(len(df)), dtype=float)
        base = (hour - idx).ffill().bfill()
        hour = hour.fillna((base + idx) % 24).fillna(12)
    df["hour"] = hour.astype(int)
    for col in WEATHER_COLUMNS:
        v = pd.to_numeric(src[col], errors="coerce") if col in src else pd.Series(np.nan, index=df.index)
        v = v.replace([np.inf, -np.inf], np.nan)
        lo, hi = LIMITS[col]
        if col == "ambient_temp_c":
            v = v.where(v.between(lo, hi))          # implausible -> missing
        else:
            v = v.clip(lo, hi)
        v = v.interpolate(limit_direction="both")
        df[col] = v.fillna(DEFAULTS[col]).astype(float)
    return df


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(0, index=clean.index, columns=HOUR_COLUMNS, dtype=int)
    for h in range(1, 24):
        feats[f"hour_{h:02d}"] = (clean["hour"] == h).astype(int)
    feats["cooling_deg_c"] = (clean["ambient_temp_c"] - COOLING_BASE_C).clip(lower=0).round(4)
    for col in WEATHER_COLUMNS:
        feats[col] = clean[col].astype(float)
    return feats[FEATURE_COLUMNS]


# --------------------------------------------------------------------------
# Response parsing (QuickML response shapes vary by pipeline/version)
# --------------------------------------------------------------------------
PRIORITY_KEYS = ("predictions", "prediction", "predicted_value", "predicted_values",
                 "target_demand_kw", "result", "results", "output", "outputs", "data", "value")


def extract_predictions(obj):
    if isinstance(obj, bool):
        return None
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        try:
            return extract_predictions(json.loads(obj))
        except (ValueError, TypeError):
            try:
                return [float(obj)]
            except ValueError:
                return None
    if isinstance(obj, list):
        out = []
        for item in obj:
            vals = extract_predictions(item)
            if not vals:
                return None
            out.append(vals[0])
        return out
    if isinstance(obj, dict):
        for key in PRIORITY_KEYS:
            for k in obj:
                if k.lower() == key:
                    vals = extract_predictions(obj[k])
                    if vals:
                        return vals
        for v in obj.values():
            vals = extract_predictions(v)
            if vals:
                return vals
    return None


def _valid(preds, n):
    arr = np.asarray(preds, dtype=float)
    return arr if arr.shape == (n,) and np.isfinite(arr).all() else None


# --------------------------------------------------------------------------
# 1. Proxy through the deployed app
# --------------------------------------------------------------------------
def via_proxy(clean: pd.DataFrame):
    if not PREDICT_PROXY_URL.startswith("http"):
        raise RuntimeError("PREDICT_PROXY_URL not configured")
    url = PREDICT_PROXY_URL.rstrip("/") + "/api/predict-batch"
    # Rows are already cleaned here; a normalised timestamp carries the
    # (possibly inferred) hour so the server derives exactly the same features.
    records = [{"timestamp": f"2000-01-01 {int(h):02d}:00:00", "ambient_temp_c": t,
                "humidity_pct": hu, "solar_irradiance_wm2": s}
               for h, t, hu, s in zip(clean["hour"], clean["ambient_temp_c"],
                                      clean["humidity_pct"], clean["solar_irradiance_wm2"])]
    out, sources = [], set()
    session = requests.Session()
    for i in range(0, len(records), PROXY_CHUNK):
        chunk = records[i:i + PROXY_CHUNK]
        for attempt in range(3):
            try:
                r = session.post(url, json={"rows": chunk}, timeout=120)
                r.raise_for_status()
                body = r.json()
                break
            except Exception:                              # noqa: BLE001
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        if body.get("source") == "local-reference":
            raise RuntimeError("proxy answered from its local fallback, not QuickML")
        vals = _valid(body.get("predictions", []), len(chunk))
        if vals is None:
            raise RuntimeError(f"proxy returned bad chunk: {str(body)[:200]}")
        sources.add(body.get("source"))
        out.extend(vals.tolist())
    return np.asarray(out, dtype=float), "proxy:" + ",".join(sorted(map(str, sources)))


# --------------------------------------------------------------------------
# 2. Direct QuickML REST call with OAuth refresh
# --------------------------------------------------------------------------
def _access_token():
    r = requests.post(
        os.environ.get("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.in").rstrip("/") + "/oauth/v2/token",
        params={"refresh_token": os.environ["ZOHO_REFRESH_TOKEN"],
                "client_id": os.environ["ZOHO_CLIENT_ID"],
                "client_secret": os.environ["ZOHO_CLIENT_SECRET"],
                "grant_type": "refresh_token"}, timeout=15)
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError(f"token refresh failed: {r.text[:200]}")
    return token


def via_quickml(feats: pd.DataFrame):
    if not QUICKML_ENDPOINT_URL.startswith("http"):
        raise RuntimeError("QUICKML_ENDPOINT_URL not configured")
    headers = {"Authorization": f"Zoho-oauthtoken {_access_token()}",
               "X-QUICKML-ENDPOINT-KEY": os.environ["QUICKML_ENDPOINT_KEY"],
               "Content-Type": "application/json",
               "Environment": os.environ.get("QUICKML_ENVIRONMENT", "Production")}
    if os.environ.get("CATALYST_ORG"):
        headers["CATALYST-ORG"] = os.environ["CATALYST_ORG"]
    records = feats.to_dict(orient="records")
    session = requests.Session()

    def post(body):
        r = session.post(QUICKML_ENDPOINT_URL, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        return extract_predictions(r.json())

    style = os.environ.get("QUICKML_PAYLOAD_STYLE", "auto").lower()
    if style in ("auto", "data_list"):
        try:
            vals = _valid(post({"data": records}), len(records))
            if vals is not None:
                return vals, "quickml-rest(batch)"
        except Exception:                                  # noqa: BLE001
            if style == "data_list":
                raise
    wrap = {"data_obj": lambda r: {"data": r}, "flat": lambda r: r}
    order = [style] if style in wrap else ["data_obj", "flat"]
    for name in order:
        try:
            first = post(wrap[name](records[0]))
            if not first:
                continue
        except Exception:                                  # noqa: BLE001
            continue
        out = [first[0]]
        for rec in records[1:]:
            out.append(post(wrap[name](rec))[0])
        vals = _valid(out, len(records))
        if vals is not None:
            return vals, f"quickml-rest({name})"
    raise RuntimeError("QuickML rejected every payload style")


# --------------------------------------------------------------------------
# 3. Offline reference (same Linear Regression, fitted on quickml_train.csv)
# --------------------------------------------------------------------------
REFERENCE_MODEL = {
    "intercept": 91.051338,
    "coefficients": {
        "hour_01": 1.035887,
        "hour_02": 1.758999,
        "hour_03": 0.919193,
        "hour_04": 0.542985,
        "hour_05": 0.811595,
        "hour_06": 43.683788,
        "hour_07": 83.474369,
        "hour_08": 84.552736,
        "hour_09": 85.482884,
        "hour_10": 55.367006,
        "hour_11": 54.508349,
        "hour_12": 53.705691,
        "hour_13": 54.61633,
        "hour_14": 54.802405,
        "hour_15": 54.14898,
        "hour_16": 55.386417,
        "hour_17": 54.644935,
        "hour_18": 108.545231,
        "hour_19": 109.673906,
        "hour_20": 109.691495,
        "hour_21": 108.966187,
        "hour_22": 108.288333,
        "hour_23": 20.90909,
        "cooling_deg_c": 8.590779,
        "ambient_temp_c": -0.031534,
        "humidity_pct": -0.02033,
        "solar_irradiance_wm2": 0.00097
    }
}  # from model/reference_model.json


def via_local(feats: pd.DataFrame):
    w = np.array([REFERENCE_MODEL["coefficients"][c] for c in FEATURE_COLUMNS])
    return feats.to_numpy(dtype=float) @ w + REFERENCE_MODEL["intercept"], "local-reference"


# --------------------------------------------------------------------------
def predict(input_df: pd.DataFrame):
    """Return a numpy array with one demand prediction (kW) per input row."""
    if len(input_df) == 0:
        return np.array([], dtype=float)
    clean = clean_inputs(input_df)
    feats = build_features(clean)
    errors = []
    for fn, arg in ((via_proxy, clean), (via_quickml, feats)):
        try:
            preds, source = fn(arg)
            print(f"[predict] {len(preds)} predictions via {source}", file=sys.stderr)
            return preds
        except Exception as e:                             # noqa: BLE001
            errors.append(f"{fn.__name__}: {e}")
    if not ALLOW_LOCAL_FALLBACK:
        raise RuntimeError("All prediction sources failed: " + " | ".join(errors))
    print("[predict] WARNING: QuickML unreachable, using offline reference model. "
          + " | ".join(errors), file=sys.stderr)
    preds, _ = via_local(feats)
    return preds


def main():
    ap = argparse.ArgumentParser(description="Forecast hourly demand (kW).")
    ap.add_argument("--input", default="sample_test_input.csv")
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.input, dtype={"timestamp": str})
    preds = predict(df)
    out = pd.DataFrame({"timestamp": df["timestamp"], "target_demand_kw": np.round(preds, 2)})
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output}")


if __name__ == "__main__":
    main()
