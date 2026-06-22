-- Application tables pre-created on first Postgres init so Grafana annotation
-- queries never fail with "relation does not exist" before the training DAG runs.
-- All tables also use CREATE TABLE IF NOT EXISTS in code; this is belt-and-suspenders.

\c dsp_db

CREATE TABLE IF NOT EXISTS model_annotations (
    id              SERIAL PRIMARY KEY,
    annotation_text TEXT NOT NULL,
    tags            TEXT DEFAULT 'model-retrain',
    model_version   TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
