"""
Telemetry cleaning and validation.

Two modes:
  * forecast mode (preserve_order=False): sort by time, drop duplicate hours,
    infer missing timestamps -- used by the dashboard.
  * contract mode (preserve_order=True): never drop or reorder rows, because
    predict.py must return exactly one prediction per input row in order.

Missing or invalid weather values are filled by linear interpolation between
the nearest valid neighbours; if a whole column is empty we fall back to a
typical value for that hour of day.
"""
from datetime import timedelta

from .features import parse_timestamp

NUMERIC = ("ambient_temp_c", "humidity_pct", "solar_irradiance_wm2")

# Accept a few header spellings people commonly use.
ALIASES = {
    "timestamp": ("timestamp", "time", "datetime", "date_time", "ts"),
    "ambient_temp_c": ("ambient_temp_c", "temp", "temperature", "temp_c", "ambient_temp"),
    "humidity_pct": ("humidity_pct", "humidity", "rh", "humidity_percent"),
    "solar_irradiance_wm2": ("solar_irradiance_wm2", "solar", "irradiance", "solar_irradiance", "ghi"),
}

# Physical plausibility limits. Values outside are treated as sensor errors.
LIMITS = {
    "ambient_temp_c": (-40.0, 60.0),
    "humidity_pct": (0.0, 100.0),
    "solar_irradiance_wm2": (0.0, 1500.0),
}

# Hour-of-day fallbacks (rounded means from the training data), used only
# when an entire column is missing.
HOURLY_DEFAULTS = {
    "ambient_temp_c": [15.0, 14.1, 13.6, 13.4, 13.8, 14.3, 15.2, 16.0, 17.1, 18.6, 19.9, 21.1,
                       21.8, 22.8, 23.4, 23.2, 23.4, 22.9, 22.0, 20.9, 19.9, 18.4, 17.1, 16.0],
    "humidity_pct": [74.5, 75.9, 77.0, 76.9, 76.7, 75.7, 74.5, 73.6, 72.2, 71.3, 69.1, 68.0,
                     66.8, 65.6, 64.8, 65.1, 65.4, 65.3, 67.1, 67.7, 68.8, 71.0, 72.5, 73.8],
    "solar_irradiance_wm2": [0.0, 0.0, 0.0, 0.6, 9.4, 36.9, 79.4, 135.1, 193.5, 241.3, 272.3, 281.0,
                             266.9, 230.3, 177.1, 116.0, 61.3, 22.1, 4.4, 0.2, 0.0, 0.0, 0.0, 0.0],
}


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        f = float(value)
    else:
        text = str(value).strip().replace(",", "")
        if text.lower() in ("", "na", "nan", "null", "none", "-", "?"):
            return None
        try:
            f = float(text)
        except ValueError:
            return None
    if f != f or f in (float("inf"), float("-inf")):   # NaN / inf
        return None
    return f


def normalise_headers(row):
    lowered = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    out = {}
    for canon, names in ALIASES.items():
        for name in names:
            if name in lowered:
                out[canon] = lowered[name]
                break
    return out


def clean_rows(raw_rows, preserve_order=False):
    """
    Returns (rows, report). Each row: {"timestamp": datetime|None, "hour": int,
    "ambient_temp_c": float, "humidity_pct": float, "solar_irradiance_wm2": float}
    report: {"input_rows", "output_rows", "issues": [str], "counts": {...}}
    """
    counts = {"bad_timestamp": 0, "inferred_timestamp": 0, "missing_values": 0,
              "out_of_range": 0, "clipped": 0, "duplicates_dropped": 0}
    issues = []
    rows = []
    for idx, raw in enumerate(raw_rows):
        r = normalise_headers(raw)
        ts = parse_timestamp(r.get("timestamp"))
        if ts is None:
            counts["bad_timestamp"] += 1
        rec = {"_line": idx + 1, "timestamp": ts}
        for col in NUMERIC:
            v = _to_float(r.get(col))
            lo, hi = LIMITS[col]
            if v is None:
                counts["missing_values"] += 1
            elif col == "ambient_temp_c" and not (lo <= v <= hi):
                counts["out_of_range"] += 1
                issues.append(f"Row {idx + 1}: temperature {v} is implausible, replaced")
                v = None
            elif not (lo <= v <= hi):
                # humidity / solar: small overshoots are sensor noise -> clip
                counts["clipped"] += 1
                v = min(max(v, lo), hi)
            rec[col] = v
        rows.append(rec)

    # --- timestamps -------------------------------------------------------
    for i, rec in enumerate(rows):
        if rec["timestamp"] is None:
            prev = next((rows[j]["timestamp"] for j in range(i - 1, -1, -1)
                         if rows[j]["timestamp"] is not None), None)
            if prev is not None:
                steps = i - next(j for j in range(i - 1, -1, -1) if rows[j]["timestamp"] is not None)
                rec["timestamp"] = prev + timedelta(hours=steps)
                rec["_inferred"] = True
                counts["inferred_timestamp"] += 1
            else:
                nxt = next(((j, rows[j]["timestamp"]) for j in range(i + 1, len(rows))
                            if rows[j]["timestamp"] is not None), None)
                if nxt is not None:
                    rec["timestamp"] = nxt[1] - timedelta(hours=nxt[0] - i)
                    rec["_inferred"] = True
                    counts["inferred_timestamp"] += 1
    if counts["inferred_timestamp"]:
        issues.append(f"{counts['inferred_timestamp']} unreadable timestamp(s) inferred from neighbouring hours")

    if not preserve_order:
        before = len(rows)
        rows = [r for r in rows if r["timestamp"] is not None]
        if len(rows) < before:
            issues.append(f"{before - len(rows)} row(s) dropped: no usable timestamp")
        rows.sort(key=lambda r: r["timestamp"])
        dedup = {}
        for r in rows:
            dedup[r["timestamp"]] = r          # keep the last reading per hour
        counts["duplicates_dropped"] = len(rows) - len(dedup)
        if counts["duplicates_dropped"]:
            issues.append(f"{counts['duplicates_dropped']} duplicate hour(s) removed (kept latest)")
        rows = list(dedup.values())

    for r in rows:
        r["hour"] = r["timestamp"].hour if r["timestamp"] is not None else 12

    # --- impute numeric gaps by interpolation ------------------------------
    for col in NUMERIC:
        valid = [i for i, r in enumerate(rows) if r[col] is not None]
        for i, r in enumerate(rows):
            if r[col] is not None:
                continue
            if not valid:
                r[col] = float(HOURLY_DEFAULTS[col][r["hour"]])
                continue
            left = max((j for j in valid if j < i), default=None)
            right = min((j for j in valid if j > i), default=None)
            if left is not None and right is not None:
                w = (i - left) / (right - left)
                r[col] = rows[left][col] * (1 - w) + rows[right][col] * w
            else:
                r[col] = rows[left if left is not None else right][col]
            if col == "solar_irradiance_wm2" and (r["hour"] <= 2 or r["hour"] >= 20):
                r[col] = 0.0                       # no sun at night
            r[col] = round(r[col], 2)
    if counts["missing_values"]:
        issues.append(f"{counts['missing_values']} missing weather value(s) filled by interpolation")
    if counts["clipped"]:
        issues.append(f"{counts['clipped']} humidity/solar value(s) clipped to physical range")

    for r in rows:
        r.pop("_line", None)
    report = {"input_rows": len(raw_rows), "output_rows": len(rows),
              "issues": issues, "counts": counts}
    return rows, report
