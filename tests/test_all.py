"""
Run:  python -m pytest tests/ -q      (or simply: python tests/test_all.py)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "appsail"))
sys.path.insert(0, ROOT)
os.environ["PREDICT_PROXY_URL"] = "unset"
os.environ["QUICKML_ENDPOINT_URL"] = "unset"

from core.cleaning import clean_rows                       # noqa: E402
from core.controller import MAIN_GRID, STORAGE, evaluate_hour, run_controller  # noqa: E402
from core.features import FEATURE_COLUMNS, build_feature_row  # noqa: E402
import predict as P                                          # noqa: E402


# ---- transition controller: every boundary in section 6 ----------------
def test_storage_threshold_is_strict():
    assert evaluate_hour(10, 160.00)["grid_mode"] == MAIN_GRID
    assert evaluate_hour(10, 160.01)["grid_mode"] == STORAGE


def test_peak_window_inclusive_both_ends():
    assert evaluate_hour(17, 50)["grid_mode"] == MAIN_GRID
    for h in range(18, 23):
        assert evaluate_hour(h, 50)["grid_mode"] == STORAGE
    assert evaluate_hour(23, 50)["grid_mode"] == MAIN_GRID


def test_overload_threshold_inclusive_and_mode_independent():
    assert evaluate_hour(3, 199.99)["overload_alert"] is False
    assert evaluate_hour(3, 200.00)["overload_alert"] is True
    assert evaluate_hour(20, 250)["overload_alert"] is True


def test_rounding_matches_display():
    # 160.004 is shown as 160.00, so it must be treated as 160.00
    assert evaluate_hour(10, 160.004)["grid_mode"] == MAIN_GRID


def test_dispatch_signals():
    rows = [{"hour": h} for h in (16, 17, 18, 23)]
    out = run_controller(rows, [100, 170, 100, 100])
    assert [o["dispatch_signal"] for o in out] == [
        "ENGAGE_GRID", "SWITCH_TO_STORAGE", "HOLD", "SWITCH_TO_GRID"]


# ---- training / inference consistency -----------------------------------
def test_feature_columns_match():
    assert P.FEATURE_COLUMNS == FEATURE_COLUMNS
    header = open(os.path.join(ROOT, "model", "quickml_train.csv")).readline().strip().split(",")
    assert header[:-1] == FEATURE_COLUMNS and header[-1] == "target_demand_kw"


def test_predict_py_features_equal_backend_features():
    df = pd.read_csv(os.path.join(ROOT, "sample_test_input.csv")).head(100)
    ours = P.build_features(P.clean_inputs(df))
    rows, _ = clean_rows(df.to_dict(orient="records"), preserve_order=True)
    theirs = pd.DataFrame([build_feature_row(r["hour"], r["ambient_temp_c"], r["humidity_pct"],
                                             r["solar_irradiance_wm2"]) for r in rows])[FEATURE_COLUMNS]
    assert np.allclose(ours.to_numpy(float), theirs.to_numpy(float))


# ---- contract robustness -------------------------------------------------
def test_predict_survives_messy_input():
    df = pd.read_csv(os.path.join(ROOT, "sample_test_input.csv")).head(30).astype(object)
    df.loc[2, "ambient_temp_c"] = np.nan
    df.loc[4, "timestamp"] = "not a time"
    df.loc[6, "humidity_pct"] = 400
    df.loc[8, "solar_irradiance_wm2"] = "oops"
    df.loc[10, "ambient_temp_c"] = 999
    preds = P.predict(df)
    assert len(preds) == len(df) and np.isfinite(preds).all()


def test_order_and_count_preserved_when_shuffled():
    df = pd.read_csv(os.path.join(ROOT, "sample_test_input.csv")).head(48)
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    a = pd.Series(P.predict(df), index=df["timestamp"])
    b = pd.Series(P.predict(shuffled), index=shuffled["timestamp"])
    assert np.allclose(a.loc[shuffled["timestamp"]].values, b.values)


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(fns)} tests passed")
