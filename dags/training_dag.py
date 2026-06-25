"""
Training pipeline (Defense 2).

    load_data -> train_model -> save_training_stats -> evaluate_candidate
        -> (conditional) promote_to_champion -> notify_api_reload
                                              -> archive_data
                                              -> post_grafana_annotation

- Trains a RandomForest on good_data, logs everything to MLflow, registers a new
  model version, and promotes it to @champion only if it beats the current champion
  and meets the inference-time budget.
- BONUS: drift detection (recent serving distribution vs the champion's training
  baseline) is computed each run; drift + enough new data forces a retrain.
- BONUS: after promotion, writes a Grafana annotation (via HTTP API + Postgres
  model_annotations table) so the drift dashboard shows a vertical "retrained" line.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from airflow.providers.postgres.hooks.postgres import PostgresHook

import ml_utils

GOOD_PATH = Path("/opt/airflow/data/good_data")
ARCHIVE_PATH = Path("/opt/airflow/data/archived_data")
CACHE_DIR = Path("/opt/airflow/data/.training_cache")
SEEN_FILE = Path("/opt/airflow/data/.training_seen")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "soccer_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "soccer_training")

MIN_NEW_FILES = int(os.getenv("MIN_NEW_FILES_FOR_TRAINING", "3"))
INFERENCE_MS_THRESHOLD = float(os.getenv("INFERENCE_MS_THRESHOLD", "50"))
DRIFT_Z_THRESHOLD = float(os.getenv("DRIFT_Z_THRESHOLD", "2.0"))

API_BASE_URL = os.getenv("API_BASE_URL", "http://model_service:8000")
GRAFANA_URL = os.getenv("GRAFANA_INTERNAL_URL", "http://grafana:3000")
ML_ALERTS_WEBHOOK_URL = os.getenv("ML_ALERTS_WEBHOOK_URL", "").strip()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _mlflow():
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow


def _read_seen() -> set:
    if not SEEN_FILE.exists():
        return set()
    return {line.strip() for line in SEEN_FILE.read_text().splitlines() if line.strip()}


def _write_seen(names: set) -> None:
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEEN_FILE.write_text("\n".join(sorted(names)))


def _get_champion_version(client):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except Exception:
        return None


def _send_ml_alert(message: str) -> None:
    print(f"[ML ALERT] {message}")
    if not ML_ALERTS_WEBHOOK_URL:
        print("ML_ALERTS_WEBHOOK_URL not configured; skipping Teams alert.")
        return
    try:
        card = {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.4",
            "body": [
                {"type": "TextBlock", "text": "ML Alert", "weight": "bolder", "size": "medium"},
                {"type": "TextBlock", "text": message, "wrap": True},
            ],
        }
        requests.post(ML_ALERTS_WEBHOOK_URL, json=card, timeout=10).raise_for_status()
    except Exception as exc:
        print(f"Failed to send ML alert: {exc}")


def _recent_serving_frame(pg_hook, limit: int = 500) -> pd.DataFrame:
    """Pull recent prediction inputs into a feature frame for drift checks."""
    records = pg_hook.get_records(
        "SELECT input_data FROM predictions ORDER BY created_at DESC LIMIT %s;" % int(limit)
    )
    rows = []
    for (raw,) in records or []:
        try:
            rows.append(json.loads(raw))
        except Exception:
            continue
    return pd.DataFrame(rows)


def _detect_drift(pg_hook) -> dict:
    """
    Compare recent serving distribution of a numerical feature (home_odds) against
    the champion's training baseline using a z-score. Returns drift flag + details.
    """
    base = pg_hook.get_first(
        """
        SELECT
            MAX(CASE WHEN metric='mean' THEN value END),
            MAX(CASE WHEN metric='std'  THEN value END)
        FROM training_feature_stats
        WHERE is_baseline = TRUE AND feature_name = 'home_odds';
        """
    )
    if not base or base[0] is None:
        return {"drift": False, "reason": "no_baseline", "z": None}

    base_mean, base_std = float(base[0]), float(base[1] or 0.0)
    serving = _recent_serving_frame(pg_hook)
    if serving.empty or "home_odds" not in serving.columns:
        return {"drift": False, "reason": "no_serving_data", "z": None}

    serving_mean = float(pd.to_numeric(serving["home_odds"], errors="coerce").dropna().mean())
    denom = base_std if base_std > 1e-9 else 1.0
    z = abs(serving_mean - base_mean) / denom
    return {
        "drift": bool(z > DRIFT_Z_THRESHOLD),
        "reason": "z_threshold",
        "z": round(z, 3),
        "base_mean": round(base_mean, 3),
        "serving_mean": round(serving_mean, 3),
    }


def _ensure_tables(pg_hook) -> None:
    pg_hook.run(
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            id SERIAL PRIMARY KEY,
            run_id TEXT,
            model_version TEXT,
            accuracy DOUBLE PRECISION,
            f1 DOUBLE PRECISION,
            inference_ms DOUBLE PRECISION,
            n_total INTEGER,
            drift_detected BOOLEAN DEFAULT FALSE,
            drift_details TEXT,
            promoted BOOLEAN DEFAULT FALSE,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    pg_hook.run(
        """
        CREATE TABLE IF NOT EXISTS training_feature_stats (
            id SERIAL PRIMARY KEY,
            model_version TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            feature_type TEXT NOT NULL,
            category TEXT,
            metric TEXT NOT NULL,
            value DOUBLE PRECISION NOT NULL,
            is_baseline BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    pg_hook.run(
        """
        CREATE TABLE IF NOT EXISTS model_annotations (
            id             SERIAL PRIMARY KEY,
            annotation_text TEXT NOT NULL,
            tags           TEXT DEFAULT 'model-retrain',
            model_version  TEXT,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


# --------------------------------------------------------------------------- #
# DAG
# --------------------------------------------------------------------------- #
with DAG(
    dag_id="soccer_training",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",  # daily 02:00; also triggered on drift + enough new data
    catchup=False,
    max_active_runs=1,  # never train concurrently with itself (avoids registry races / OOM)
    tags=["ml", "training"],
) as dag:

    @task(multiple_outputs=False)
    def load_data() -> dict:
        """Gate the run: require enough NEW good_data files since the last training."""
        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        _ensure_tables(pg_hook)

        good_files = sorted(GOOD_PATH.glob("*.csv")) if GOOD_PATH.exists() else []
        seen = _read_seen()
        new_files = [f for f in good_files if f.name not in seen]

        drift = _detect_drift(pg_hook)
        enough = len(new_files) >= MIN_NEW_FILES

        # Core: enough new data triggers training.
        # Bonus: drift detected also justifies a retrain (still needs data to learn from).
        if not enough:
            raise AirflowSkipException(
                f"Only {len(new_files)} new file(s) (< {MIN_NEW_FILES}); "
                f"drift={drift.get('drift')}. Skipping training run."
            )

        df = ml_utils.load_good_data(GOOD_PATH)
        X, y = ml_utils.build_features(df)
        if len(X) < 10 or y.nunique() < 2:
            raise AirflowSkipException("Not enough usable/labelled rows to train.")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        run_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        data_path = CACHE_DIR / f"train_{run_stamp}.parquet"
        full = X.copy()
        full["__target__"] = y.values
        full.to_parquet(data_path, index=False)

        return {
            "data_path": str(data_path),
            "new_files": [f.name for f in new_files],
            "drift": drift,
            "n_rows": int(len(X)),
        }

    @task(multiple_outputs=False)
    def train_model(meta: dict) -> dict:
        """Train candidate, log params/metrics/model to MLflow, register a new version."""
        mlflow = _mlflow()
        from mlflow import MlflowClient

        full = pd.read_parquet(meta["data_path"])
        y = full["__target__"]
        X = full.drop(columns=["__target__"])

        model, X_test, y_test, metrics = ml_utils.train_random_forest(X, y)

        # Persist the held-out test set so evaluate_candidate scores both
        # candidate and champion on the SAME current data (fair comparison).
        test_path = Path(meta["data_path"]).with_name(
            Path(meta["data_path"]).stem + "_test.parquet"
        )
        test_df = X_test.copy()
        test_df["__target__"] = y_test.values
        test_df.to_parquet(test_path, index=False)

        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"train_{datetime.utcnow():%Y%m%d_%H%M%S}") as run:
            mlflow.log_params({
                "n_estimators": 200,
                "n_total": metrics["n_total"],
                "n_train": metrics["n_train"],
                "n_test": metrics["n_test"],
                "features": ",".join(ml_utils.FEATURES),
            })
            mlflow.log_metrics({
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "inference_ms": metrics["inference_ms"],
            })
            mlflow.sklearn.log_model(
                model, artifact_path="model", registered_model_name=MODEL_NAME
            )
            run_id = run.info.run_id

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.search_model_versions(f"run_id='{run_id}'")
        version = max((int(v.version) for v in versions), default=None)

        return {
            "run_id": run_id,
            "version": version,
            "metrics": metrics,
            "data_path": meta["data_path"],
            "test_path": str(test_path),
            "drift": meta["drift"],
            "new_files": meta["new_files"],
        }

    @task(multiple_outputs=False)
    def save_training_stats(train_meta: dict) -> dict:
        """Save per-feature baseline stats for the candidate (drift baseline source)."""
        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        _ensure_tables(pg_hook)

        # Compute baseline from the full training set (representative distribution).
        full = pd.read_parquet(train_meta["data_path"])
        model_version = f"{MODEL_NAME}_v{train_meta['version']}"
        X = full.drop(columns=["__target__"], errors="ignore")
        stats = ml_utils.compute_feature_stats(X, model_version)

        for s in stats:
            pg_hook.run(
                """
                INSERT INTO training_feature_stats
                    (model_version, feature_name, feature_type, category, metric, value, is_baseline)
                VALUES (%s, %s, %s, %s, %s, %s, FALSE);
                """,
                parameters=(
                    s["model_version"], s["feature_name"], s["feature_type"],
                    s["category"], s["metric"], s["value"],
                ),
            )
        return {**train_meta, "model_version": model_version}

    @task(multiple_outputs=False)
    def evaluate_candidate(train_meta: dict) -> dict:
        """Compare candidate vs champion on the same test set + inference budget."""
        mlflow = _mlflow()
        from mlflow import MlflowClient
        from sklearn.metrics import accuracy_score

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        test = pd.read_parquet(train_meta["test_path"])
        y_test = test["__target__"]
        X_test = test.drop(columns=["__target__"])

        candidate = mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}/{train_meta['version']}"
        )
        cand_acc = float(accuracy_score(y_test, candidate.predict(X_test)))
        cand_inf = float(train_meta["metrics"]["inference_ms"])

        champ = _get_champion_version(client)
        if champ is None:
            return {
                **train_meta,
                "promote": True,
                "reason": "bootstrap_first_model",
                "candidate_acc": cand_acc,
                "champion_acc": None,
                "inference_ms": cand_inf,
            }

        champ_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        champ_acc = float(accuracy_score(y_test, champ_model.predict(X_test)))

        meets_perf = cand_acc >= champ_acc
        meets_latency = cand_inf < INFERENCE_MS_THRESHOLD
        promote = meets_perf and meets_latency

        if promote:
            reason = f"candidate_acc {cand_acc:.3f} >= champion {champ_acc:.3f}, inf {cand_inf:.1f}ms OK"
        elif not meets_perf:
            reason = f"candidate_acc {cand_acc:.3f} < champion {champ_acc:.3f}"
        else:
            reason = f"inference {cand_inf:.1f}ms >= threshold {INFERENCE_MS_THRESHOLD}ms"

        # Log comparison to the candidate's MLflow run.
        try:
            with mlflow.start_run(run_id=train_meta["run_id"]):
                mlflow.log_metrics({"champion_accuracy": champ_acc, "candidate_accuracy": cand_acc})
                mlflow.set_tag("promotion_decision", "promote" if promote else "reject")
                mlflow.set_tag("promotion_reason", reason)
        except Exception as exc:
            print(f"Could not log comparison to MLflow: {exc}")

        return {
            **train_meta,
            "promote": promote,
            "reason": reason,
            "candidate_acc": cand_acc,
            "champion_acc": champ_acc,
            "inference_ms": cand_inf,
        }

    @task(multiple_outputs=False)
    def promote_to_champion(ev: dict) -> dict:
        """Set @champion alias if evaluation passed; record the run + baseline."""
        from mlflow import MlflowClient

        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        _ensure_tables(pg_hook)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        metrics = ev["metrics"]
        drift = ev.get("drift", {})

        pg_hook.run(
            """
            INSERT INTO training_runs
                (run_id, model_version, accuracy, f1, inference_ms, n_total,
                 drift_detected, drift_details, promoted, reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            parameters=(
                ev["run_id"], ev["model_version"], metrics["accuracy"], metrics["f1"],
                metrics["inference_ms"], metrics["n_total"],
                bool(drift.get("drift")), json.dumps(drift),
                bool(ev["promote"]), ev["reason"],
            ),
        )

        if not ev["promote"]:
            _send_ml_alert(
                f"Candidate {ev['model_version']} NOT promoted. Reason: {ev['reason']}. "
                f"Champion accuracy={ev.get('champion_acc')}, candidate={ev.get('candidate_acc')}."
            )
            raise AirflowSkipException(f"Not promoted: {ev['reason']}")

        # Promote: set alias + make this model's stats the only drift baseline.
        # MLflow's registry API expects the version as a string, not an int.
        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, str(ev["version"]))
        pg_hook.run("UPDATE training_feature_stats SET is_baseline = FALSE;")
        pg_hook.run(
            "UPDATE training_feature_stats SET is_baseline = TRUE WHERE model_version = %s;",
            parameters=(ev["model_version"],),
        )
        print(f"Promoted {ev['model_version']} to @{MODEL_ALIAS}.")
        return {"promoted": True, "version": ev["version"], "new_files": ev["new_files"]}

    @task
    def notify_api_reload(promo: dict) -> str:
        """Tell the API to reload the new champion from the registry."""
        if not promo or not promo.get("promoted"):
            raise AirflowSkipException("No promotion; skipping API reload.")
        resp = requests.post(f"{API_BASE_URL}/reload-model", timeout=30)
        resp.raise_for_status()
        print(f"API reload response: {resp.json()}")
        return resp.text

    @task(trigger_rule="none_failed_min_one_success")
    def archive_data(promo: dict) -> str:
        """Move trained good_data files to archived_data (only after promotion)."""
        if not promo or not promo.get("promoted"):
            raise AirflowSkipException("No promotion; keeping good_data as-is.")
        ARCHIVE_PATH.mkdir(parents=True, exist_ok=True)
        moved = []
        for name in promo.get("new_files", []):
            src = GOOD_PATH / name
            if src.exists():
                shutil.move(str(src), str(ARCHIVE_PATH / name))
                moved.append(name)
        # Reset the seen-set: archived files are gone from good_data.
        remaining = {f.name for f in GOOD_PATH.glob("*.csv")} if GOOD_PATH.exists() else set()
        _write_seen(remaining)
        return f"Archived {len(moved)} file(s)."

    @task(trigger_rule="none_failed_min_one_success")
    def post_grafana_annotation(promo: dict) -> str:
        """
        After a successful promotion write a Grafana annotation so the drift
        and prediction dashboards show a vertical 'model retrained' line.

        Two complementary approaches (both attempted; failures are non-fatal):
          1. INSERT into model_annotations — Postgres-backed annotation query
             embedded in the dashboard JSON; persists across Grafana restarts.
          2. POST to Grafana HTTP API — appears immediately without a page reload.
        """
        import base64

        if not promo or not promo.get("promoted"):
            raise AirflowSkipException("No promotion; skipping annotation.")

        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        model_version = f"{MODEL_NAME}_v{promo.get('version', '?')}"
        annotation_text = f"Model retrained → @champion: {model_version}"
        tags = "model-retrain"

        # 1 — Postgres row (dashboard annotation query)
        try:
            pg_hook.run(
                """
                INSERT INTO model_annotations (annotation_text, tags, model_version)
                VALUES (%s, %s, %s);
                """,
                parameters=(annotation_text, tags, model_version),
            )
            print(f"Inserted annotation into model_annotations: {annotation_text}")
        except Exception as exc:
            print(f"Warning: could not write to model_annotations: {exc}")

        # 2 — Grafana HTTP API (grafana:3000 reachable within Docker network)
        try:
            now_ms = int(datetime.utcnow().timestamp() * 1000)
            payload = json.dumps({
                "time": now_ms,
                "text": annotation_text,
                "tags": [tags, model_version],
            }).encode()
            creds = base64.b64encode(b"admin:admin").decode()
            req = requests.post(
                f"{GRAFANA_URL}/api/annotations",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Basic {creds}",
                },
                timeout=10,
            )
            req.raise_for_status()
            print(f"Grafana API annotation posted: {req.json()}")
        except Exception as exc:
            print(f"Warning: Grafana API annotation failed (non-fatal): {exc}")

        return annotation_text

    # Wiring
    loaded = load_data()
    trained = train_model(loaded)
    stats = save_training_stats(trained)
    evaluated = evaluate_candidate(stats)
    promoted = promote_to_champion(evaluated)
    notify_api_reload(promoted)
    archive_data(promoted)
    post_grafana_annotation(promoted)
