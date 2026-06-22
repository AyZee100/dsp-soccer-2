# Model service (FastAPI)

## What this package does

1. **Serves predictions** over HTTP for the Streamlit UI and the Airflow prediction DAG.
2. **Owns the database schema** for `predictions`, `ingestion_stats`, and `training_stats` (SQLAlchemy models in `main.py`).
3. **Loads the sklearn model once** when the process starts (`lifespan`), from `model.pkl` or a small fallback fit if the file is missing or incompatible.
4. **Writes every prediction** (batch or single) with `source` (`webapp` vs `scheduled`) so the UI can filter history.

## File map

| Area | Where |
|------|--------|
| Env / DB engine | `DATABASE_URL`, `create_engine`, `SessionLocal` |
| Tables | `PredictionRecord`, `IngestionStat`, `TrainingStat` |
| Request/response shapes | `FeatureInput`, `PredictRequest`, `PredictionRow`, `PredictResponse` |
| Session injection | `get_db()` + `Depends(get_db)` |
| Startup | `lifespan` → `create_all`, optional column migration, `_load_model()` |
| Routes | `/health`, `/predict`, `/past-predictions`, `/reload-model`, `/ingestion-stats` |

## Defense talking points

- **Pydantic** validates inputs; **422** is automatic on bad JSON/types.
- **400** when `features` is empty; **500** if the model failed to load.
- **Sync `def` endpoints** with sync SQLAlchemy: OK for this course (FastAPI runs in a thread pool).
- **`/reload-model`**: Defense 1 reloads local `pickle`; Defense 2 would swap in an MLflow champion.
