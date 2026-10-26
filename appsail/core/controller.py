"""
Transition controller (problem statement section 6) + battery simulation.

Rules, evaluated independently for every forecast hour:
  * Renewable Storage Mode  if demand > 160 kW  OR  18 <= hour <= 22
  * Main Grid Mode          otherwise (demand <= 160 kW AND outside 18-22)
  * Critical Overload Alert if demand >= 200 kW  (independent of mode)

Comparisons use the prediction rounded to 2 decimals -- the same number the
dashboard and predictions.csv show -- so what users see always matches the
decision that was made.
"""

STORAGE_THRESHOLD_KW = 160.0      # strictly greater than -> storage
OVERLOAD_THRESHOLD_KW = 200.0     # greater than or equal -> alert
PEAK_WINDOW = (18, 22)            # inclusive hours

MAIN_GRID = "MAIN_GRID"
STORAGE = "RENEWABLE_STORAGE"


def evaluate_hour(hour, demand_kw):
    demand = round(float(demand_kw), 2)
    in_window = PEAK_WINDOW[0] <= int(hour) <= PEAK_WINDOW[1]
    above = demand > STORAGE_THRESHOLD_KW
    mode = STORAGE if (above or in_window) else MAIN_GRID
    reasons = []
    if above:
        reasons.append(f"demand {demand:.2f} kW > {STORAGE_THRESHOLD_KW:.0f} kW")
    if in_window:
        reasons.append("evening peak window 18:00-22:00")
    if mode == MAIN_GRID:
        reasons.append(f"demand {demand:.2f} kW <= {STORAGE_THRESHOLD_KW:.0f} kW, outside peak window")
    overload = demand >= OVERLOAD_THRESHOLD_KW
    return {"grid_mode": mode, "overload_alert": overload, "reasons": reasons}


def run_controller(rows, predictions):
    """
    rows: cleaned rows (timestamp, hour, ...); predictions: list of floats.
    Adds grid_mode, overload_alert, reasons and a dispatch signal per hour.
    The signal records transitions: SWITCH_TO_STORAGE / SWITCH_TO_GRID / HOLD.
    """
    out, prev_mode = [], None
    for r, p in zip(rows, predictions):
        decision = evaluate_hour(r["hour"], p)
        mode = decision["grid_mode"]
        if prev_mode is None:
            signal = "ENGAGE_STORAGE" if mode == STORAGE else "ENGAGE_GRID"
        elif mode != prev_mode:
            signal = "SWITCH_TO_STORAGE" if mode == STORAGE else "SWITCH_TO_GRID"
        else:
            signal = "HOLD"
        prev_mode = mode
        out.append({**decision, "predicted_demand_kw": round(float(p), 2), "dispatch_signal": signal})
    return out


DEFAULT_BATTERY = {
    "capacity_kwh": 1000.0,     # usable pack size
    "start_soc_pct": 85.0,
    "min_soc_pct": 15.0,        # protective floor; below this the grid backs up
    "max_discharge_kw": 250.0,
    "grid_charge_kw": 80.0,     # charging rate while on main grid
    "pv_area_m2": 400.0,
    "pv_efficiency": 0.20,
}


def simulate_battery(rows, decisions, params=None):
    """
    Hour-by-hour state-of-charge (SOC) simulation. The grid mode itself is
    always decided by the rules above; the battery only reports how well it
    can serve those decisions (e.g. reserve exhausted -> grid backup).
    """
    p = {**DEFAULT_BATTERY, **(params or {})}
    cap = max(1.0, float(p["capacity_kwh"]))
    soc_kwh = cap * min(100.0, max(0.0, float(p["start_soc_pct"]))) / 100.0
    floor_kwh = cap * float(p["min_soc_pct"]) / 100.0
    results = []
    for r, d in zip(rows, decisions):
        demand = d["predicted_demand_kw"]
        solar_kw = r["solar_irradiance_wm2"] * p["pv_area_m2"] * p["pv_efficiency"] / 1000.0
        discharge = charge = grid_import = grid_backup = 0.0
        if d["grid_mode"] == STORAGE:
            need = max(0.0, demand - solar_kw)
            discharge = min(need, max(0.0, soc_kwh - floor_kwh), p["max_discharge_kw"])
            grid_backup = need - discharge
            surplus = max(0.0, solar_kw - demand)
            charge = min(surplus, cap - (soc_kwh - discharge))
            soc_kwh = soc_kwh - discharge + charge
            grid_import = grid_backup
        else:
            grid_import = demand
            charge = min(p["grid_charge_kw"] + solar_kw, cap - soc_kwh)
            soc_kwh += charge
            grid_import += max(0.0, charge - solar_kw)
        results.append({
            "solar_kw": round(solar_kw, 2),
            "battery_discharge_kw": round(discharge, 2),
            "battery_charge_kw": round(charge, 2),
            "grid_import_kw": round(grid_import, 2),
            "grid_backup_kw": round(grid_backup, 2),
            "soc_pct": round(100.0 * soc_kwh / cap, 1),
            "reserve_alert": grid_backup > 0.01,
        })
    return results, p
