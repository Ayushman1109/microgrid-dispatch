# Microgrid Dispatch — HackSphere 3.0

Forecasts hourly energy demand for the next 24 hours with a model trained in **Zoho Catalyst QuickML**, applies the grid-transition rules, simulates battery state-of-charge, and shows it all on a live dashboard hosted on **Catalyst AppSail**.

## Results

| Model (rolling validation on the last 2,000 training hours) | RMSE (kW) |
|---|---|
| Starter-kit dummy formula | 45.8 |
| Random Forest / Gradient Boosting on our features | 20–21 |
| **Linear Regression on our features (submitted)** | **8.69** |
| Irreducible noise floor (residual std) | ≈ 8.5 |

Why linear beats trees here: exploring the data shows demand follows a fixed daily schedule plus a cooling load that rises ~8.6 kW for every °C above 24 °C, at every hour of the day. The hidden test period is mid-summer, hotter than most of the training data. Tree models cannot predict beyond the hottest values they've seen; a linear model extends the cooling slope naturally. Humidity, solar, weekday, and lagged/rolling temperature add nothing (tested in `model/validate_model.py`).

## Architecture

```
Browser (dashboard: static/index.html, styled after ISA-101 operator displays)
   │  form edits / CSV upload / sample day
   ▼
AppSail · Python Flask (appsail/app.py)
   ├─ core/cleaning.py        parse, validate, interpolate, dedupe
   ├─ core/features.py        hour one-hots + cooling degrees (same as training)
   ├─ core/quickml_client.py  ──► QuickML published endpoint (SDK or OAuth REST)
   ├─ core/controller.py      grid-mode rules, overload alerts, dispatch signals, battery SOC
   └─ core/storage.py         system log + optional Data Store / Stratus persistence

predict.py (evaluator) ──► AppSail /api/predict-batch ──► QuickML
                       └─► or QuickML directly with OAuth env vars
```

| Problem-statement module | Implementation |
|---|---|
| Telemetry ingestion | Dashboard form (24 editable hours), CSV upload (optionally archived to Stratus) |
| Preprocessing & API | AppSail Flask backend, pure-Python (only `flask`, `requests`, `zcatalyst-sdk`) |
| Forecasting engine | QuickML Linear Regression pipeline + published endpoint |
| Transition controller | `core/controller.py`, unit-tested at every rule boundary |
| Live dashboard | Control-room style: annunciator alarm panel, live single-line diagram (breakers and power flows per hour), 24-hour trend with playback and keyboard scrubbing, sequence-of-events log, heatwave what-if (shift temperatures), hourly schedule with CSV export, input checks, system log |

## Repository layout

```
predict.py                 graded inference script (contract: predict(input_df))
test_eval.py               official self-test (unchanged)
sample_test_input.csv
model/
  train_telemetry.csv      starter-kit training data
  build_quickml_dataset.py → quickml_train.csv (upload this to QuickML)
  quickml_train.csv
  validate_model.py        algorithm comparison + reference_model.json
  reference_model.json
appsail/
  app.py, requirements.txt, app-config.example.json
  core/                    cleaning, features, controller, quickml_client, storage
  static/index.html        dashboard
  data/sample_test_input.csv
tests/
  test_all.py              rule boundaries, feature consistency, contract robustness
  mock_quickml.py          local fake of Zoho OAuth + QuickML for offline testing
```

## Setup

### 1. Train and publish the model in QuickML

1. `python model/build_quickml_dataset.py` (already done; `model/quickml_train.csv` is included).
2. QuickML → Datasets → upload `model/quickml_train.csv`.
3. Create an **ML pipeline** (not AutoML — AutoML may pick a tree model, which scores ~2.3× worse):
   - Target column: `target_demand_kw`, problem type: Regression.
   - Algorithm: **Linear Regression**. Leave all 27 feature columns in. Do not add normalisation or encoding stages; the columns are already model-ready.
4. Run the pipeline. The cross-validated RMSE QuickML reports should be about 8.5–9 kW. If it is far higher, check the algorithm and that no column was dropped.
5. Endpoints → create an endpoint from that model → **Publish**. Open the endpoint page and copy the endpoint URL, the endpoint key, and the **sample request** shown there.

### 2. Get a refresh token (so calls keep working after 1 hour)

Zoho access tokens expire after one hour, and judges will run `predict.py` days later. The code therefore stores a long-lived *refresh token* and fetches fresh access tokens itself.

1. Go to https://api-console.zoho.in → Add client → **Self Client**. Note the client ID and secret.
2. Generate a grant code with scope `QuickML.deployment.READ` (10-minute validity).
3. Exchange it within 10 minutes:
   ```bash
   curl -X POST "https://accounts.zoho.in/oauth/v2/token?grant_type=authorization_code&client_id=ID&client_secret=SECRET&code=GRANT_CODE"
   ```
   Save the `refresh_token` from the response. Use `accounts.zoho.com` instead if your Catalyst account is on the US data centre.

### 3. Deploy the backend to AppSail

```bash
npm install -g zcatalyst-cli
catalyst login
catalyst init            # choose AppSail, Python runtime, build path = appsail
cd appsail
python3 -m pip install -r requirements.txt -t .   # AppSail needs dependencies vendored
catalyst deploy
```

Set the startup command to `python3 -u app.py` (or copy `command` from `app-config.example.json` into the `app-config.json` that `catalyst init` created). Then add the environment variables below in the console (AppSail → your service → Configuration → Environment variables). Open the AppSail URL: the dashboard loads a sample day and forecasts it immediately.

### 4. Point `predict.py` at your deployment

Edit the top of `predict.py`:

```python
PREDICT_PROXY_URL = "https://<your-appsail-url>"
QUICKML_ENDPOINT_URL = "<endpoint URL from step 1>"
```

Then:

```bash
python predict.py --input sample_test_input.csv --output predictions.csv
python test_eval.py      # must print SUCCESS
```

The first stderr line says which path answered (`proxy:quickml-...`, `quickml-rest(...)` or the offline warning). **Before submitting, confirm it says `proxy:quickml-...`.**

## Environment variables

Never commit values; `.env.example` lists them.

| Variable | Where | Purpose |
|---|---|---|
| `QUICKML_ENDPOINT_URL` | AppSail, optional for predict.py | Endpoint URL from the QuickML endpoint page |
| `QUICKML_ENDPOINT_KEY` | AppSail, optional for predict.py | `X-QUICKML-ENDPOINT-KEY` header |
| `CATALYST_ORG` | AppSail, optional for predict.py | `CATALYST-ORG` header (org ID) |
| `ZOHO_CLIENT_ID`, `ZOHO_CLIENT_SECRET`, `ZOHO_REFRESH_TOKEN` | AppSail, optional for predict.py | OAuth refresh flow |
| `ZOHO_ACCOUNTS_URL` | both | `https://accounts.zoho.in` (default) or `.com` |
| `QUICKML_ENVIRONMENT` | both | `Production` (default) or `Development` |
| `QUICKML_PAYLOAD_STYLE` | both | `auto` (default) tries `{"data":[...]}`, `{"data":{...}}`, then a flat record. Set it to match your endpoint's sample request once known |
| `ALLOW_LOCAL_FALLBACK` | both | Backend default `false`. predict.py default `true`: if QuickML is unreachable it uses an offline copy of the same linear model and prints a warning, so the evaluation never crashes. Set `false` if the rules require QuickML-only predictions |
| `DATASTORE_TABLE` | AppSail | e.g. `ForecastRuns`; stores each run summary (columns in `core/storage.py`) |
| `STRATUS_BUCKET` | AppSail | optional; archives uploaded CSVs |
| `PREDICT_PROXY_URL` | predict.py | Overrides the constant in the file |

## Transition controller

Evaluated per forecast hour on the prediction rounded to 2 decimals (the same value shown and exported):

| Condition | Output |
|---|---|
| demand > 160 kW **or** 18 ≤ hour ≤ 22 | Renewable Storage Mode |
| otherwise | Main Grid Mode |
| demand ≥ 200 kW | Critical Overload Alert (independent of mode) |

Each hour also carries a dispatch signal: `ENGAGE_GRID` / `ENGAGE_STORAGE` for the first hour, then `SWITCH_TO_STORAGE`, `SWITCH_TO_GRID` or `HOLD`. The battery simulation (adjustable in the dashboard) reports charge level and flags hours where the reserve floor forces grid backup; it never overrides the rule-based mode.

## Data cleaning

Header aliases (`temp`, `humidity`, `solar`…), many timestamp formats, unreadable timestamps inferred from neighbouring hours, implausible temperatures treated as missing, humidity/solar clipped to physical ranges, gaps filled by linear interpolation (night-time solar forced to 0), duplicate hours removed (dashboard only). `predict.py` and `/api/predict-batch` never drop or reorder rows. Every repair is listed in the dashboard's data-quality panel.

## Testing

```bash
python tests/test_all.py          # 9 tests: rule boundaries, feature consistency, messy input
python tests/mock_quickml.py      # fake QuickML on :8765 for end-to-end tests without credentials
```

## Assumptions to verify on your account

- The exact request body your endpoint expects (see its sample request); auto-detection covers the three common shapes.
- Data Store and Stratus calls are best-effort and log a warning instead of failing if the table or bucket is missing.
