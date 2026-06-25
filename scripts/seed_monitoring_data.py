"""
OPTIONAL demo helper: seed the database with backdated monitoring data so the
Grafana dashboards show meaningful history immediately (instead of waiting for
the pipelines to accumulate data over time).

It inserts, spread across the last ~2 hours:
  - ingestion_errors rows (per-category counts + criticality)
  - a training baseline in training_feature_stats (is_baseline = TRUE)
  - a training_runs summary row
  - predictions rows, including a recent "drift" window where home_odds shifts up

Usage (after `docker compose up`, from the repo root):
    py -3.12 scripts/seed_monitoring_data.py
Connection comes from DATABASE_URL or defaults to the local compose Postgres.
"""
import json
import os
import random
from datetime import datetime, timedelta

import psycopg2

DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/dsp_db",
)
MODEL_VERSION = os.getenv("SEED_MODEL_VERSION", "soccer_model_v1")
random.seed(7)


def _conn():
    return psycopg2.connect(DSN)


def ensure_tables(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_errors (
            id SERIAL PRIMARY KEY, file_name TEXT NOT NULL, row_count INTEGER NOT NULL,
            completeness INTEGER DEFAULT 0, validity INTEGER DEFAULT 0, consistency INTEGER DEFAULT 0,
            schema_errors INTEGER DEFAULT 0, type_errors INTEGER DEFAULT 0, total_errors INTEGER DEFAULT 0,
            criticality TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_feature_stats (
            id SERIAL PRIMARY KEY, model_version TEXT NOT NULL, feature_name TEXT NOT NULL,
            feature_type TEXT NOT NULL, category TEXT, metric TEXT NOT NULL, value DOUBLE PRECISION NOT NULL,
            is_baseline BOOLEAN DEFAULT FALSE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id SERIAL PRIMARY KEY, run_id TEXT, model_version TEXT, accuracy DOUBLE PRECISION,
            f1 DOUBLE PRECISION, inference_ms DOUBLE PRECISION, n_total INTEGER,
            drift_detected BOOLEAN DEFAULT FALSE, drift_details TEXT, promoted BOOLEAN DEFAULT FALSE,
            reason TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY, source VARCHAR(32) DEFAULT 'webapp', model_version VARCHAR(64) DEFAULT 'local_v1',
            input_data TEXT NOT NULL, prediction DOUBLE PRECISION NOT NULL, proba_home_win DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_annotations (
            id SERIAL PRIMARY KEY, annotation_text TEXT NOT NULL,
            tags TEXT DEFAULT 'model-retrain', model_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """)


def seed_ingestion(cur, now):
    for i in range(40):
        ts = now - timedelta(minutes=120 - i * 3)
        rows = random.randint(25, 35)
        comp = random.randint(0, 4)
        val = random.randint(0, 5)
        cons = random.randint(0, 2)
        sch = 1 if random.random() < 0.08 else 0
        typ = random.randint(0, 2)
        # A spike of all-errors near the middle of the window.
        if 18 <= i <= 20:
            comp, val, total = rows, 0, rows
            crit = "High"
        else:
            total = min(rows, comp + val + cons + (sch * rows) + typ)
            ratio = total / rows
            crit = "High" if ratio > 0.5 else "Medium" if ratio >= 0.1 else "Low" if total else "None"
        cur.execute(
            """INSERT INTO ingestion_errors
               (file_name,row_count,completeness,validity,consistency,schema_errors,type_errors,total_errors,criticality,created_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (f"batch_{i}.csv", rows, comp, val, cons, sch, typ, total, crit, ts),
        )


def seed_baseline(cur, now):
    cur.execute("UPDATE training_feature_stats SET is_baseline = FALSE;")
    base = [
        ("home_odds", "numeric", None, "mean", 2.63), ("home_odds", "numeric", None, "std", 1.45),
        ("draw_odds", "numeric", None, "mean", 3.95), ("draw_odds", "numeric", None, "std", 0.95),
        ("away_odds", "numeric", None, "mean", 4.40), ("away_odds", "numeric", None, "std", 3.20),
        ("favorite", "categorical", "HOME", "freq", 0.73),
        ("favorite", "categorical", "DRAW", "freq", 0.01),
        ("favorite", "categorical", "AWAY", "freq", 0.26),
    ]
    for fname, ftype, cat, metric, val in base:
        cur.execute(
            """INSERT INTO training_feature_stats
               (model_version,feature_name,feature_type,category,metric,value,is_baseline,created_at)
               VALUES (%s,%s,%s,%s,%s,%s,TRUE,%s)""",
            (MODEL_VERSION, fname, ftype, cat, metric, val, now - timedelta(hours=3)),
        )
    cur.execute(
        """INSERT INTO training_runs
           (run_id,model_version,accuracy,f1,inference_ms,n_total,drift_detected,drift_details,promoted,reason,created_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        ("seed-run", MODEL_VERSION, 0.61, 0.56, 6.8, 22592, False,
         json.dumps({"drift": False, "reason": "seed"}), True, "bootstrap_first_model",
         now - timedelta(hours=3)),
    )


def seed_predictions(cur, now):
    for i in range(240):
        ts = now - timedelta(minutes=120) + timedelta(seconds=i * 30)
        recent = ts > now - timedelta(minutes=25)  # induce drift in recent window
        home = round(random.gauss(3.4 if recent else 2.6, 0.6), 2)
        home = max(1.05, home)
        draw = round(random.gauss(3.9, 0.4), 2)
        away = round(random.gauss(4.2, 1.0), 2)
        feats = {
            "home_team_id": random.randint(8000, 10100),
            "away_team_id": random.randint(8000, 10100),
            "home_odds": home, "draw_odds": draw, "away_odds": away,
        }
        proba = max(0.02, min(0.98, 1.6 / home - 0.1))
        pred = 1.0 if proba >= 0.5 else 0.0
        cur.execute(
            """INSERT INTO predictions (source,model_version,input_data,prediction,proba_home_win,created_at)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            ("scheduled", MODEL_VERSION, json.dumps(feats), pred, round(proba, 3), ts),
        )


def main():
    now = datetime.utcnow()
    with _conn() as conn:
        with conn.cursor() as cur:
            ensure_tables(cur)
            seed_ingestion(cur, now)
            seed_baseline(cur, now)
            seed_predictions(cur, now)
        conn.commit()
    print("Seeded ingestion_errors, training baseline, training_runs, and predictions.")
    print("Open Grafana (http://localhost:3000) and set the range to 'Last 3 hours'.")


if __name__ == "__main__":
    main()
