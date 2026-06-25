# Defense 2 — Demo Runbook (step by step)


Logins: **Airflow** admin / (password from logs) · **Grafana** admin / admin

| Service | Link |
|---|---|
| API health | http://localhost:8000/health |
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |
| Streamlit | http://localhost:8501 |
| Data Docs (nginx) | http://localhost:8081 |

---

## 0 · Before the slot (clean slate)

1. Open **Docker Desktop**.
2. Reset and start fresh:
   ```bat
   docker compose down -v --remove-orphans
   docker compose up --build -d
   ```
   **Also empty the data folders** 
   ```powershell
   Get-ChildItem data\good_data, data\bad_data, data\raw_data, data\archived_data -File -Exclude .gitkeep | Remove-Item -Force
   ```
3. Confirm all 7 are up/healthy:
   ```bat
   docker compose ps
   ```
4. Create the input batches:
   ```bat
   python scripts\split_dataset.py
   ```
5. Log in to Airflow (username is `admin`). `standalone` generated its own password:
   ```powershell
   docker compose exec airflow cat /opt/airflow/simple_auth_manager_passwords.json.generated
   ```
   That prints `{"admin": "<password>"}` — use that password.

> 
---

## 1 · Ingestion + data quality  -- data is validated before it reaches the model

1. Open **Airflow** → http://localhost:8080 → **Dags**.
2. Turn **ON** `soccer_ingestion` and `soccer_prediction` .
3. Click `soccer_ingestion` → Great Expectations validates each CSV, splits good/bad rows, and saves the 5 error categories to Postgres.
4. Links:
   - **Streamlit** → http://localhost:8501 → *Ingestion Stats* and *Past Predictions*.
   - **Data Docs** → http://localhost:8081 → the Great Expectations report.

run ~2–3 minutes so predictions build up.

---

## 2 · Confirm predictions are flowing

```bat
docker compose exec db psql -U user -d dsp_db -c "SELECT count(*) FROM predictions;"
```

> If it's **0**: the prediction DAG skipped because of a leftover state file. Fix:
> ```bat
> docker compose exec airflow rm -f /opt/airflow/data/.prediction_state
> docker compose exec airflow airflow dags trigger soccer_prediction
> ```
> wait 30s, re-check the count.

---

## 3 · Train + promote the champion  -- the model retrains itself

1. In Airflow, **pause `soccer_prediction`** (toggle OFF) so it doesn't compete during training.
2. Open `soccer_training` → click **▶ Trigger** 
3. API hot-swapped to the new champion:
   ```bat
   curl.exe http://localhost:8000/health
   ```
   Expect `"model_source":"mlflow"`, `"model_version":"soccer_model_v..."`.
5. Show **MLflow** → http://localhost:5000 → **Models** → `soccer_model` with the `@champion`
   alias; **Experiments** → the run's accuracy / F1 / inference-time metrics.


## 4 · Monitoring in Grafana -- data quality and model drift live

1. Open **Grafana** → http://localhost:3000 (admin / admin).
2. Top-right time range → **Last 1 hour**.
3. **Data Quality Monitoring** dashboard:
   - invalid-row rate per ingestion, errors by the **5 categories**, criticality over time.
4. **Data Drift & Prediction Monitoring** dashboard:
   - `home_odds` mean vs training baseline (numerical drift),
   - `favorite=HOME` frequency vs baseline (categorical drift),
   - home-win rate + predicted-class counts (model behavior),
   - the orange **"Model Retrained"** annotation line at the moment of promotion.

**2 alerts per dashboard** are configured and route to Teams channels
(`Data Quality Alerts`, `ML Alerts`).

---

## 5 · Wrap

- **CI/CD**: flake8 + pytest on PRs to `develop`; automated **GitHub Release** on push to `main`.
- **Not working: Teams alert *delivery* uses placeholder webhook URLs, so it doesn't
  post to a live channel — everything else is functional. (Don't demo Teams delivery.)

