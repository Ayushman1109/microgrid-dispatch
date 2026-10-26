"""
HackSphere 3.0 microgrid backend (Catalyst AppSail, Python + Flask).

Routes
  GET  /                    dashboard (static/index.html)
  GET  /api/health          configuration status
  GET  /api/sample          24 sample hours (?date=YYYY-MM-DD, default a typical day)
  POST /api/forecast        JSON {"rows": [...], "battery": {...}} or multipart CSV
                            -> clean, predict via QuickML, run controller, simulate battery
  POST /api/predict-batch   JSON {"rows": [...]} -> {"predictions": [...]}  (order kept;
                            used by predict.py so judges need no QuickML credentials)
  GET  /api/logs            recent system events and runs
"""
import csv
import io
import os
import uuid

from flask import Flask, jsonify, request, send_from_directory

from core import storage
from core.cleaning import clean_rows
from core.controller import STORAGE, run_controller, simulate_battery
from core.quickml_client import fallback_allowed, predict_rows, rest_configured

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
SAMPLE_CSV = os.path.join(HERE, "data", "sample_test_input.csv")
FORECAST_HOURS = 24
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_BATCH_ROWS = 2000

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def catalyst_app():
    """Catalyst SDK handle when running on AppSail, else None (local dev)."""
    try:
        import zcatalyst_sdk
        return zcatalyst_sdk.initialize(req=request)
    except Exception:                                     # noqa: BLE001
        return None


def read_csv_text(text):
    text = text.lstrip("\ufeff")
    return [dict(r) for r in csv.DictReader(io.StringIO(text))]


def store_upload(zc_app, filename, raw_bytes):
    bucket = os.environ.get("STRATUS_BUCKET", "").strip()
    if not bucket or zc_app is None:
        return
    try:
        key = f"uploads/{storage.now_iso()[:10]}/{uuid.uuid4().hex[:8]}-{filename}"
        zc_app.stratus().bucket(bucket).put_object(key, raw_bytes)
        storage.log("info", f"CSV archived to Stratus as {key}")
    except Exception as e:                                # noqa: BLE001
        storage.log("warn", f"Stratus archive skipped: {e}")


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC, name)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok",
                    "quickml_rest_configured": rest_configured(),
                    "quickml_sdk_key_set": bool(os.environ.get("QUICKML_ENDPOINT_KEY")),
                    "local_fallback_allowed": fallback_allowed()})


@app.route("/api/sample")
def sample():
    with open(SAMPLE_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    days = {}
    for r in rows:
        days.setdefault(r["timestamp"][:10], []).append(r)
    full_days = {d: v for d, v in days.items() if len(v) == 24}
    wanted = request.args.get("date")
    if wanted not in full_days:
        # default: a typical day (median daily temperature) -- shows both modes
        means = sorted((sum(float(x["ambient_temp_c"]) for x in v) / 24, d) for d, v in full_days.items())
        wanted = means[len(means) // 2][1]
    return jsonify({"date": wanted, "available_dates": sorted(full_days), "rows": full_days[wanted]})


@app.route("/api/forecast", methods=["POST", "OPTIONS"])
def forecast():
    if request.method == "OPTIONS":
        return "", 204
    zc_app = catalyst_app()
    battery_params = {}
    if request.files.get("file"):
        upload = request.files["file"]
        raw = upload.read()
        try:
            raw_rows = read_csv_text(raw.decode("utf-8", errors="replace"))
        except csv.Error as e:
            return jsonify({"error": f"Could not read the CSV: {e}"}), 400
        store_upload(zc_app, upload.filename or "telemetry.csv", raw)
        input_kind = f"CSV upload ({upload.filename})"
        for k in request.form:
            if k.startswith("battery_"):
                battery_params[k[8:]] = request.form[k]
    else:
        body = request.get_json(silent=True) or {}
        raw_rows = body.get("rows") or []
        battery_params = body.get("battery") or {}
        input_kind = body.get("input_kind", "form input")

    if not raw_rows:
        return jsonify({"error": "No telemetry rows received. Upload a CSV or fill in the form."}), 400

    rows, report = clean_rows(raw_rows, preserve_order=False)
    if not rows:
        return jsonify({"error": "No usable rows after cleaning: every row lacked a readable timestamp.",
                        "cleaning": report}), 400
    if len(rows) > FORECAST_HOURS:
        report["issues"].append(f"{len(rows)} hours supplied; forecasting the first {FORECAST_HOURS}")
        rows = rows[:FORECAST_HOURS]
    elif len(rows) < FORECAST_HOURS:
        report["issues"].append(f"Only {len(rows)} hour(s) supplied; a full window is {FORECAST_HOURS}")

    try:
        preds, source, errors = predict_rows(rows, zc_app)
    except Exception as e:                                # noqa: BLE001
        storage.log("error", f"Forecast failed: {e}")
        return jsonify({"error": f"Forecast service unavailable: {e}", "cleaning": report}), 502
    for err in errors:
        storage.log("warn", err)

    decisions = run_controller(rows, preds)
    try:
        params = {k: float(v) for k, v in battery_params.items() if str(v).strip() != ""}
    except ValueError:
        params = {}
    battery, used_params = simulate_battery(rows, decisions, params)

    hourly = []
    for r, d, b in zip(rows, decisions, battery):
        hourly.append({
            "timestamp": r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"), "hour": r["hour"],
            "ambient_temp_c": r["ambient_temp_c"], "humidity_pct": r["humidity_pct"],
            "solar_irradiance_wm2": r["solar_irradiance_wm2"], **d, **b})

    overload = [h for h in hourly if h["overload_alert"]]
    summary = {
        "run_id": uuid.uuid4().hex[:10], "run_at": storage.now_iso(), "source": source,
        "input": input_kind, "rows": len(hourly),
        "peak_kw": max(h["predicted_demand_kw"] for h in hourly),
        "storage_hours": sum(h["grid_mode"] == STORAGE for h in hourly),
        "overload_hours": len(overload),
        "transitions": sum(h["dispatch_signal"].startswith("SWITCH") for h in hourly),
        "first_mode": hourly[0]["grid_mode"],
        "min_soc_pct": min(h["soc_pct"] for h in hourly),
    }
    persisted = storage.record_run(summary, zc_app)
    storage.log("alert" if overload else "info",
                f"Forecast {summary['run_id']} via {source}: peak {summary['peak_kw']:.1f} kW, "
                f"{summary['storage_hours']} storage h, {len(overload)} overload h")
    return jsonify({"summary": summary, "hourly": hourly, "cleaning": report,
                    "battery_params": used_params, "persisted": persisted, "warnings": errors})


@app.route("/api/predict-batch", methods=["POST", "OPTIONS"])
def predict_batch():
    if request.method == "OPTIONS":
        return "", 204
    body = request.get_json(silent=True) or {}
    raw_rows = body.get("rows") or []
    if not raw_rows:
        return jsonify({"error": "Send {\"rows\": [...]}"}), 400
    if len(raw_rows) > MAX_BATCH_ROWS:
        return jsonify({"error": f"At most {MAX_BATCH_ROWS} rows per request"}), 413
    rows, report = clean_rows(raw_rows, preserve_order=True)
    try:
        preds, source, errors = predict_rows(rows, catalyst_app())
    except Exception as e:                                # noqa: BLE001
        storage.log("error", f"predict-batch failed: {e}")
        return jsonify({"error": str(e)}), 502
    storage.log("info", f"predict-batch: {len(preds)} rows via {source}")
    return jsonify({"predictions": [round(p, 4) for p in preds], "source": source,
                    "cleaning": report["counts"], "warnings": errors})


@app.route("/api/logs")
def logs():
    return jsonify({"events": storage.recent_events(), "runs": storage.recent_runs()})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large: the limit is 2 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT", os.environ.get("PORT", 9000)))
    app.run(host="0.0.0.0", port=port)

