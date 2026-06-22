# Defense 2 — Demo Runbook (10-minute demo)

## Before the room fills (T − 15 min)

```bat
:: 1. Start the full stack (if not already running)
docker compose up -d

:: 2. Wait for all services to be healthy (~60 s), then verify:
py -3.12 scripts\healthcheck.py
```

All 7 rows must be GREEN. Fix anything red before proceeding.

```bat
:: 3. Clean state + fresh baseline
py -3.12 scripts\demo_reset.py --db --yes
py -3.12 scripts\seed_monitoring_data.py
```

> **Why seed?** — Seeds a training baseline (`home_odds` mean=2.63, std=1.45) and
> 240 historical predictions so the Grafana dashboards show meaningful history and
> the drift z-score calculation has a reference point.

```bat
:: 4. Open browser tabs (leave them in this order, full screen Grafana last)
::    Tab 1: http://localhost:8080   (Airflow)
::    Tab 2: http://localhost:5000   (MLflow)
::    Tab 3: http://localhost:3000   (Grafana — Drift & Prediction dashboard)
::    Tab 4: http://localhost:8081   (Data Docs)
::    Tab 5: http://localhost:8000/health  (API)

:: 5. Start the demo conductor (keep it visible in a terminal)
py -3.12 scripts\demo_run.py
```

---

## Beat-by-Beat Script (≈ 10 minutes total)

### Beat 1 — Clean ingestion → good_data + Data Docs (≈ 1.5 min)

**Say:** *"The pipeline starts with CSV batches landing in `raw_data`. Airflow picks
them up every minute, runs Great Expectations validation, and routes rows to
`good_data` or `bad_data`."*

- Press ENTER in the conductor → `demo_clean_1.csv` (30 clean rows) is copied and
  the `soccer_ingestion` DAG is triggered.
- Switch to **Tab 1** (Airflow) → show the DAG running.
- When the conductor shows `Split complete`, switch to **Tab 4** (Data Docs) and
  show the validation report.

**Say:** *"Zero errors. All 30 rows pass every expectation — moved to `good_data`
for prediction."*

**URL:** `http://localhost:8081`

---

### Beat 2 — All-errors batch → High criticality (≈ 1.5 min)

**Say:** *"Now I inject a completely corrupt file — every row has a null B365H odds
field, which is a completeness error."*

- Press ENTER → `demo_all_errors.csv` injected, ingestion triggered.
- **Tab 1**: show `save_statistics` and `send_alerts` tasks.
- **Tab 3**: switch to **Data Quality Monitoring** → invalid-row rate hits 100%.

**Say:** *"Criticality is HIGH — that fires a Grafana alert and a Teams message
if the webhook is configured. All 30 rows land in `bad_data`, nothing goes to the
model."*

**URL:** `http://localhost:3000/d/dsp_data_quality`

---

### Beat 3 — Mixed batch → row-level split (≈ 1.5 min)

**Say:** *"Real-world data is messier — some rows are good, some aren't. The
pipeline splits at the row level."*

- Press ENTER → `demo_mixed.csv` (20 valid + 10 bad rows) injected.
- **Tab 3**: show the error category breakdown (completeness, validity, type).

**Say:** *"10 rows go to `bad_data` — 3 completeness errors (null odds), 5 validity
errors (negative goals or zero odds), 2 type errors (string in goal column). The 20
clean rows go to `good_data` and are available for prediction."*

**URL:** `http://localhost:3000/d/dsp_data_quality`

---

### Beat 4 — Predictions + model version stamp (≈ 1.5 min)

**Say:** *"With enough good rows accumulated, the prediction DAG sends them to the
FastAPI model service, which loads the champion from the MLflow registry."*

- Press ENTER → `demo_clean_2.csv` and `demo_clean_3.csv` ingested, prediction DAG
  triggered.
- **Tab 5** (`/health`): show `model_version` and `model_source=mlflow`.
- **Tab 3**: switch to **Drift & Prediction Monitoring** → prediction counts appear.

**Say:** *"Every prediction row in Postgres carries the `model_version` that made
it — this is how we draw the before/after line on the drift chart."*

**URLs:**
- `http://localhost:8000/health`
- `http://localhost:8000/past-predictions?limit=20&source=scheduled`
- `http://localhost:3000/d/dsp_drift_predictions`

---

### Beat 5 — Inject input drift (≈ 1 min)

**Say:** *"I'll now simulate a distribution shift: odds patterns that look different
from what the model trained on. In production this would come from changing betting
markets."*

- Press ENTER → 300 synthetic predictions with `home_odds` ≈ 7.2 (vs training
  baseline 2.63) are written directly to the DB.
- **Tab 3**: set time range to **Last 15 minutes** → the `home_odds serving mean`
  panel spikes above the dashed baseline line.

**Say:** *"The z-score exceeds 2.0, which is our drift threshold. The system logs
`drift_detected=True` in the training run and this can trigger an automatic retrain
— today I'm going to trigger it manually to walk you through the full flow."*

**URL:** `http://localhost:3000/d/dsp_drift_predictions`

---

### Beat 6 — Retrain → promote → /reload-model → new champion (≈ 2 min)

**Say:** *"We have 4 new good files since the last training AND drift detected —
both conditions satisfied. Triggering the training pipeline."*

- Press ENTER → `soccer_training` DAG triggered.
- **Tab 1**: open the `soccer_training` DAG and watch tasks turn green:
  `load_data → train_model → save_training_stats → evaluate_candidate
   → promote_to_champion → notify_api_reload → archive_data → post_grafana_annotation`
- **Tab 2** (MLflow): show the new run in the `soccer_training` experiment; the
  `@champion` alias on the model registry.

**Say:** *"The candidate beat the previous champion on accuracy **and** inference
time. `@champion` alias is re-pointed, and `notify_api_reload` calls `POST /reload-model`
on the FastAPI service — zero downtime, no container restart."*

- Press ENTER (after tasks are green) → conductor checks `/health` and prints the
  new `model_version`.
- **Tab 5**: refresh `/health` → `model_version` shows the new version number.
- **Tab 3**: a vertical orange **"Model Retrained"** annotation line appears on
  the drift panels — before/after the line you can see the model_version change in
  the confidence panel.

**URLs:**
- `http://localhost:5000/#/models/soccer_model`
- `http://localhost:8000/health`
- `http://localhost:3000/d/dsp_drift_predictions`

---

## Fallback: training run is slow or skipped

**If `promote_to_champion` takes > 3 min:** The training DAG uses Airflow's
LocalExecutor inside the container, which can be slow if the machine is loaded.
- Show MLflow `soccer_training` experiment → point to the *in-progress* run and
  explain what each logged metric means.
- While waiting, demonstrate the **Streamlit** app at `http://localhost:8501`.

**If the task is `skipped`:** Check the `load_data` task logs. Common causes:
1. `Only N new file(s) < 3` → you skipped a beat or ingestion failed. Copy one more
   clean file: `copy demo_data\demo_clean_1.csv data\raw_data\extra_clean.csv` then
   trigger ingestion and re-trigger training.
2. `Not enough usable/labelled rows` → unlikely with demo files; re-run `demo_reset`
   and start over.

**If `/reload-model` is never called:** Manually reload:
```bat
curl -s -X POST http://localhost:8000/reload-model
```

---

## Command Quick-Reference

| Command | Purpose |
|---|---|
| `py -3.12 scripts\healthcheck.py` | Pre-demo service check |
| `py -3.12 scripts\demo_reset.py --db --yes` | Wipe to clean state |
| `py -3.12 scripts\seed_monitoring_data.py` | Seed baseline + history |
| `py -3.12 scripts\demo_run.py` | Paced demo conductor |
| `docker compose logs -f model_service` | Watch API logs |
| `docker compose exec airflow airflow dags trigger soccer_training` | Manual retrain |

## Key URLs at a glance

| Service | URL | Credentials |
|---|---|---|
| API | http://localhost:8000/health | — |
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | admin / admin |
| Data Docs | http://localhost:8081 | — |
| Streamlit | http://localhost:8501 | — |
