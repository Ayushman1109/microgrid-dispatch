"""
Local stand-in for Zoho accounts + a QuickML endpoint, for testing the
network code without real credentials. It enforces the documented headers
and only accepts ONE payload style ({"data": {...}}), so the client's
auto-discovery is exercised too. Predictions come from reference_model.json.

    python tests/mock_quickml.py   # listens on :8765
"""
import json
import os

from flask import Flask, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
REF = json.load(open(os.path.join(HERE, "..", "model", "reference_model.json")))
app = Flask(__name__)
CALLS = {"n": 0}


@app.post("/oauth/v2/token")
def token():
    if request.args.get("grant_type") != "refresh_token":
        return jsonify({"error": "invalid_grant"}), 400
    return jsonify({"access_token": "mock-token", "expires_in": 3600})


@app.post("/quickml/predict")
def predict():
    CALLS["n"] += 1
    if request.headers.get("Authorization") != "Zoho-oauthtoken mock-token":
        return jsonify({"error": "unauthorised"}), 401
    if not request.headers.get("X-QUICKML-ENDPOINT-KEY"):
        return jsonify({"error": "missing endpoint key"}), 400
    body = request.get_json(silent=True) or {}
    rec = body.get("data")
    if not isinstance(rec, dict):
        return jsonify({"error": "expected {'data': {feature: value}}"}), 400
    y = REF["intercept"] + sum(w * float(rec[c]) for c, w in REF["coefficients"].items())
    return jsonify({"status": "success", "result": [y]})


@app.get("/calls")
def calls():
    return jsonify(CALLS)


if __name__ == "__main__":
    app.run(port=8765)
