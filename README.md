# dsp-project — Soccer ML Pipeline (Defense 1 + Defense 2)

End-to-end **MLOps** for tabular soccer match prediction: ingest CSV batches with
**Great Expectations**, store data-quality stats, run **scheduled predictions** against a
**sklearn** model served by a **FastAPI** API, retrain & promote models with **MLflow**, and
monitor data quality + drift in **Grafana** — all orchestrated by **Airflow 3.x** in Docker.

The model predicts **home win** (`home_team_goal > away_team_goal`) from 5 features:
`home_team_id, away_team_id, home_odds (B365H), draw_odds (B365D), away_odds (B365A)`.

## Architecture

```
                 scripts/split_dataset.py + generate_errors.py
                                  |
                                  v
data/raw_data ──► Airflow: soccer_ingestion (every 1 min)
                    GX validation → 5-category error stats → Postgres
                    good_data / bad_data split → Teams (Data Quality Alerts)
                    GX Data Docs → nginx (http://localhost:8081)
                                  |
                                  v
data/good_data ──► Airflow: soccer_prediction (every 2 min) ──► API /predict
                                  |
data/good_data ──► Airflow: soccer_training (daily + on drift)
                    train → MLflow log/register → save baseline stats
                    evaluate vs @champion → promote (@champion alias)
                    → API /reload-model  + archive good_data
                                  |
                                  v
model_service (FastAPI) ── loads models:/soccer_model@champion from MLflow
                            /predict, /reload-model, /past-predictions, /ingestion-stats
                                  |
                                  v
PostgreSQL ── predictions, ingestion_stats, ingestion_errors,
              training_runs, training_feature_stats
                  |                         |
                  v                         v
            Grafana dashboards         Grafana alerts → Teams
            (data quality + drift)     (Data Quality Alerts / ML Alerts)
```

## Services & URLs

| Service       | URL                              | Notes                                  |
|---------------|----------------------------------|----------------------------------------|
| API           | http://localhost:8000/health     | FastAPI model service                  |
| Streamlit     | http://localhost:8501            | Multipage web app                      |
| Airflow       | http://localhost:8080            | `admin` / `admin`                      |
| MLflow        | http://localhost:5000            | Tracking server + model registry       |
| Grafana       | http://localhost:3000            | `admin` / `admin`                      |
| nginx (Docs)  | http://localhost:8081            | Great Expectations Data Docs           |
| PostgreSQL    | localhost:5432                   | `dsp_db` (+ `mlflow_db` backend store) |

## Prerequisites

- Docker Desktop / Docker Engine + Compose v2
- Python **3.12** on the host (only for the helper `scripts/`)
- Run everything from the **repository root** (same level as `docker-compose.yml`)

## Setup & run

```bat
copy .env.example .env
```

Edit `.env` and set the Teams webhook URLs (leave blank to disable Teams alerts —
alerts will still log):

```
TEAMS_WEBHOOK_URL=          # ingestion (data quality) alerts from Airflow
DATA_QUALITY_WEBHOOK_URL=   # Grafana "Data Quality Alerts" channel
ML_ALERTS_WEBHOOK_URL=      # training DAG + Grafana "ML Alerts" channel
```

Start the full stack (builds the MLflow image on first run):

```bat
docker compose up --build -d
```

Prepare data on the host:

```bat
py -3.12 scripts\split_dataset.py      :: create batches in data\raw_data
py -3.12 scripts\generate_errors.py    :: optionally inject data-quality issues
py -3.12 scripts\make_demo_files.py     :: create 5 demo files in demo_data\
```

The DAGs run on a schedule. To trigger manually (demo/testing):

```bat
docker compose exec airflow airflow dags trigger soccer_ingestion
docker compose exec airflow airflow dags trigger soccer_prediction
docker compose exec airflow airflow dags trigger soccer_training
```

> The first `soccer_training` run has no champion yet, so it **bootstraps**: it trains,
> registers `soccer_model`, sets the `@champion` alias, and the API auto-loads it on the
> next `/reload-model`.

Optional — seed backdated monitoring data so Grafana shows history immediately:

```bat
py -3.12 scripts\seed_monitoring_data.py
```

## Training pipeline (`soccer_training`)

```
load_data → train_model → save_training_stats → evaluate_candidate
   → (conditional) promote_to_champion → notify_api_reload
                                       → archive_data
```

- **load_data** — skips the run unless `MIN_NEW_FILES_FOR_TRAINING` (default 3) new files
  have accumulated in `good_data`. Also computes drift (bonus trigger).
- **train_model** — trains a RandomForest, logs params/metrics/model to MLflow, registers a
  new `soccer_model` version.
- **save_training_stats** — writes per-feature baseline stats (`home_odds` mean/std, `favorite`
  category frequencies) to `training_feature_stats` (drift baseline source).
- **evaluate_candidate** — compares candidate vs `@champion` on the **same held-out test set**;
  promotes only if accuracy ≥ champion **and** inference time < `INFERENCE_MS_THRESHOLD` (50 ms).
  Handles the first-run (no champion) case.
- **promote_to_champion** — sets the `@champion` alias and marks the promoted model's stats as
  the only `is_baseline = TRUE`. On rejection it logs to MLflow and alerts the **ML Alerts** channel.
- **notify_api_reload** / **archive_data** — reload the API and move trained files to
  `archived_data` (only after a successful promotion).

**Bonus — drift-triggered retraining:** `load_data` compares the recent serving mean of
`home_odds` to the training baseline (z-score). Retraining requires **both** drift **and**
enough new data.

## Monitoring (Grafana)

Two provisioned dashboards (auto-loaded, query Postgres directly, temporal, threshold colors):

- **Data Quality Monitoring** — invalid-row rate, errors by the 5 categories
  (completeness, validity, consistency, schema, type), criticality over time, recent ingestions.
- **Data Drift & Prediction Monitoring** — `home_odds` mean vs baseline (numerical drift),
  `favorite=HOME` frequency vs baseline (categorical drift), home-win rate, predicted-class
  counts, and confidence by model version.

Provisioned alerts (2 per dashboard) route to Teams:

| Channel               | Alerts                                                        |
|-----------------------|---------------------------------------------------------------|
| `Data Quality Alerts` | All ingested data has errors; high validation error volume    |
| `ML Alerts`           | Significant input drift; predictions collapsed to a constant  |

## CI / CD

- **CI** (`.github/workflows/ci.yml`) — flake8 + pytest on PRs to `develop`.
- **CD** (`.github/workflows/cd.yml`) — automated GitHub Release on every push/merge to `main`.

## Stack

Python 3.12 · Docker Compose · PostgreSQL 15 · FastAPI · SQLAlchemy · scikit-learn ·
MLflow 2.16 · Streamlit · Apache Airflow 3.0.2 · Great Expectations · nginx · Grafana 11 ·
GitHub Actions
