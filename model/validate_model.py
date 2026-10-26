"""
Offline check of which QuickML algorithm to pick, using the exact CSV you
upload. Uses "rolling-origin" validation: train on the past, test on the next
500 hours, four times -- the same direction (past -> future) as the real test.

    pip install pandas scikit-learn
    python model/validate_model.py

Also writes model/reference_model.json: the linear model's coefficients.
The app uses them only as a clearly-labelled offline fallback.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "quickml_train.csv"))
y = df.pop("target_demand_kw")
X_all = df
X_core = df.drop(columns=["ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"])

FOLDS = [(2000, 2500), (2500, 3000), (3000, 3500), (3500, 4000)]


def rolling_rmse(model, X):
    scores = []
    for a, b in FOLDS:
        model.fit(X.iloc[:a], y.iloc[:a])
        err = model.predict(X.iloc[a:b]) - y.iloc[a:b]
        scores.append(np.sqrt(np.mean(err ** 2)))
    return np.mean(scores)


candidates = {
    "Linear Regression (all 27 features)": (LinearRegression(), X_all),
    "Linear Regression (hours + cooling only)": (LinearRegression(), X_core),
    "Ridge alpha=1 (all features)": (Ridge(alpha=1.0), X_all),
    "Random Forest (all features)": (RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0), X_all),
    "Gradient Boosting (all features)": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0), X_all),
}
print(f"{'Model':45s} rolling RMSE (kW)")
for name, (model, X) in candidates.items():
    print(f"{name:45s} {rolling_rmse(model, X):6.2f}")

final = LinearRegression().fit(X_all, y)
resid = y - final.predict(X_all)
ref = {
    "intercept": float(final.intercept_),
    "coefficients": {c: float(w) for c, w in zip(X_all.columns, final.coef_)},
    "train_rmse_kw": float(np.sqrt(np.mean(resid ** 2))),
}
with open(os.path.join(HERE, "reference_model.json"), "w") as f:
    json.dump(ref, f, indent=2)
print(f"\nCooling slope: {ref['coefficients']['cooling_deg_c']:.2f} kW per degree above 24 C")
print(f"In-sample RMSE: {ref['train_rmse_kw']:.2f} kW (roughly the noise floor)")
