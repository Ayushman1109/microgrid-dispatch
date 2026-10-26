"""
Calls the published QuickML endpoint. Tried in this order:

1. Catalyst Python SDK (internal auth, no tokens needed) -- works when this
   code runs inside AppSail and QUICKML_ENDPOINT_KEY is set.
2. REST + Zoho OAuth refresh-token flow -- works anywhere (env vars below).
3. Local reference model -- ONLY if ALLOW_LOCAL_FALLBACK=true. Every response
   says which source produced it, and the dashboard shows it.

Environment variables (never commit their values):
    QUICKML_ENDPOINT_URL      endpoint URL shown on the QuickML endpoint page
    QUICKML_ENDPOINT_KEY      X-QUICKML-ENDPOINT-KEY header value
    CATALYST_ORG              CATALYST-ORG header value
    QUICKML_ENVIRONMENT       Development | Production (default Production)
    ZOHO_CLIENT_ID / ZOHO_CLIENT_SECRET / ZOHO_REFRESH_TOKEN
    ZOHO_ACCOUNTS_URL         default https://accounts.zoho.in (India DC)
    QUICKML_PAYLOAD_STYLE     auto | flat | data_obj | data_list (default auto)
    ALLOW_LOCAL_FALLBACK      true | false (default false)
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .features import FEATURE_COLUMNS, build_feature_row

HERE = os.path.dirname(os.path.abspath(__file__))
_token_lock = threading.Lock()
_token = {"value": None, "expires": 0.0}
_style_cache = {"style": None}


def env(name, default=""):
    return os.environ.get(name, default).strip()


def rest_configured():
    return bool(env("QUICKML_ENDPOINT_URL") and env("QUICKML_ENDPOINT_KEY")
                and env("ZOHO_REFRESH_TOKEN") and env("ZOHO_CLIENT_ID"))


def fallback_allowed():
    return env("ALLOW_LOCAL_FALLBACK", "false").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------- response
PRIORITY_KEYS = ("predictions", "prediction", "predicted_value", "predicted_values",
                 "target_demand_kw", "result", "results", "output", "outputs", "data", "value")


def extract_predictions(obj):
    """Pull a flat list of numbers out of whatever JSON shape QuickML returns."""
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


# ---------------------------------------------------------------- auth
def get_access_token():
    with _token_lock:
        if _token["value"] and time.time() < _token["expires"] - 60:
            return _token["value"]
        resp = requests.post(
            env("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.in").rstrip("/") + "/oauth/v2/token",
            params={"refresh_token": env("ZOHO_REFRESH_TOKEN"),
                    "client_id": env("ZOHO_CLIENT_ID"),
                    "client_secret": env("ZOHO_CLIENT_SECRET"),
                    "grant_type": "refresh_token"},
            timeout=15)
        resp.raise_for_status()
        body = resp.json()
        if "access_token" not in body:
            raise RuntimeError(f"Zoho token refresh failed: {body}")
        _token["value"] = body["access_token"]
        _token["expires"] = time.time() + float(body.get("expires_in", 3600))
        return _token["value"]


def _headers():
    h = {"Authorization": f"Zoho-oauthtoken {get_access_token()}",
         "X-QUICKML-ENDPOINT-KEY": env("QUICKML_ENDPOINT_KEY"),
         "Content-Type": "application/json"}
    if env("CATALYST_ORG"):
        h["CATALYST-ORG"] = env("CATALYST_ORG")
    h["Environment"] = env("QUICKML_ENVIRONMENT", "Production")
    return h


def _body(style, records):
    if style == "data_list":
        return {"data": records}
    if style == "data_obj":
        return {"data": records[0]}
    return records[0]                                   # flat


def _post(style, records):
    resp = requests.post(env("QUICKML_ENDPOINT_URL"), headers=_headers(),
                         json=_body(style, records), timeout=30)
    resp.raise_for_status()
    vals = extract_predictions(resp.json())
    if not vals or len(vals) != len(records):
        raise ValueError(f"unexpected response shape: {str(resp.text)[:200]}")
    return vals


def _rest_predict(feature_rows):
    style = env("QUICKML_PAYLOAD_STYLE", "auto").lower()
    if style == "auto":
        style = _style_cache["style"]
    if style is None:                                   # discover once
        errors = []
        if len(feature_rows) > 1:
            try:
                vals = _post("data_list", feature_rows)
                _style_cache["style"] = "data_list"
                return vals
            except Exception as e:                        # noqa: BLE001
                errors.append(f"data_list: {e}")
        for candidate in ("data_obj", "flat"):
            try:
                _post(candidate, feature_rows[:1])
                _style_cache["style"] = style = candidate
                break
            except Exception as e:                        # noqa: BLE001
                errors.append(f"{candidate}: {e}")
        if style is None:
            raise RuntimeError("QuickML rejected every payload style: " + " | ".join(errors))
    if style == "data_list":
        return _post("data_list", feature_rows)
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda r: _post(style, [r])[0], feature_rows))


def _sdk_predict(zc_app, feature_rows):
    qml = zc_app.quick_ml()
    fn = getattr(qml, "run_inference", None) or getattr(qml, "predict")
    key = env("QUICKML_ENDPOINT_KEY")
    out = []
    for r in feature_rows:
        vals = extract_predictions(fn(key, r))
        if not vals:
            raise ValueError("SDK returned no numeric prediction")
        out.append(vals[0])
    return out


# ---------------------------------------------------------------- fallback
_reference = None


def _reference_predict(feature_rows):
    global _reference
    if _reference is None:
        with open(os.path.join(HERE, "reference_model.json")) as f:
            _reference = json.load(f)
    b, w = _reference["intercept"], _reference["coefficients"]
    return [b + sum(w[c] * float(r[c]) for c in FEATURE_COLUMNS) for r in feature_rows]


# ---------------------------------------------------------------- public
def predict_rows(clean_rows, zc_app=None):
    """clean_rows -> (predictions[list of float], source[str], errors[list])."""
    feats = [build_feature_row(r["hour"], r["ambient_temp_c"], r["humidity_pct"],
                               r["solar_irradiance_wm2"]) for r in clean_rows]
    if not feats:
        return [], "none", []
    errors = []
    if zc_app is not None and env("QUICKML_ENDPOINT_KEY"):
        try:
            return _sdk_predict(zc_app, feats), "quickml-sdk", errors
        except Exception as e:                            # noqa: BLE001
            errors.append(f"SDK call failed: {e}")
    if rest_configured():
        try:
            return _rest_predict(feats), "quickml-rest", errors
        except Exception as e:                            # noqa: BLE001
            errors.append(f"REST call failed: {e}")
    else:
        errors.append("QuickML REST credentials not configured")
    if fallback_allowed():
        return _reference_predict(feats), "local-reference", errors
    raise RuntimeError("; ".join(errors) or "No prediction source available")
