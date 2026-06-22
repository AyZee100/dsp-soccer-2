from datetime import datetime
import ast
import os
from pathlib import Path

import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator, get_current_context


GOOD_DATA_PATH = Path("/opt/airflow/data/good_data")
STATE_FILE = Path("/opt/airflow/data/.prediction_state")
API_URL = os.getenv("PREDICTION_API_URL", "http://model_service:8000/predict")


def _extract_batch_index(filename: str) -> int | None:
    # Expected format: batch_<int>.csv
    stem = Path(filename).stem  # e.g. "batch_3"
    if not stem.startswith("batch_"):
        return None
    try:
        return int(stem.split("batch_", 1)[1])
    except Exception:
        return None


def _read_state() -> str:
    if not STATE_FILE.exists():
        return ""
    return STATE_FILE.read_text(encoding="utf-8").strip()


def _write_state(value: str) -> None:
    STATE_FILE.write_text(value, encoding="utf-8")


def _check_for_new_data() -> bool:
    """
    Airflow compliance: when there is no newly ingested data, downstream tasks must be skipped,
    resulting in a "skipped" DAG run in the UI (or at least all tasks skipped).
    """
    context = get_current_context()
    ti = context["ti"]

    if not GOOD_DATA_PATH.exists():
        ti.xcom_push(key="file_paths", value=[])
        return False

    last_seen = _read_state()
    files = [f for f in GOOD_DATA_PATH.iterdir() if f.suffix == ".csv"]
    # Sort numerically when possible so `batch_10` comes after `batch_2`.
    files = sorted(
        files,
        key=lambda p: (_extract_batch_index(p.name) is None, _extract_batch_index(p.name) or -1, p.name),
    )
    if not files:
        ti.xcom_push(key="file_paths", value=[])
        return False

    existing_names = {f.name for f in files}

    # If state points to a batch that doesn't exist anymore (common when we wiped `good_data`),
    # we must re-process everything that is currently available.
    if last_seen and last_seen not in existing_names:
        new_files = [str(f) for f in files]
        ti.xcom_push(key="file_paths", value=new_files)
        return True

    last_idx = _extract_batch_index(last_seen) if last_seen else None
    idxs = [_extract_batch_index(f.name) for f in files]
    idxs = [i for i in idxs if i is not None]
    max_idx = max(idxs) if idxs else None

    if last_idx is not None and max_idx is not None:
        # If the state is ahead of what exists now, re-process what we have.
        if last_idx > max_idx:
            new_files = [str(f) for f in files]
        else:
            new_files = [
                str(f)
                for f in files
                if (idx := _extract_batch_index(f.name)) is not None and idx > last_idx
            ]
    else:
        # Fallback: lexicographic compare (less reliable, but prevents total blockage).
        if not last_seen:
            new_files = [str(f) for f in files]
        else:
            new_files = [str(f) for f in files if f.name > last_seen]

    ti.xcom_push(key="file_paths", value=new_files)
    return bool(new_files)


def _make_predictions(file_paths: list[str]) -> None:
    # Airflow templating may pass list-like XCom values as a string
    # (e.g. "['/path/batch_7.csv']"). Convert back to a Python list.
    if isinstance(file_paths, str):
        try:
            parsed = ast.literal_eval(file_paths)
            if isinstance(parsed, list):
                file_paths = parsed
            else:
                file_paths = [str(parsed)]
        except Exception:
            file_paths = [file_paths]

    newest_processed = ""
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        required = ["home_team_api_id", "away_team_api_id", "B365H", "B365D", "B365A"]
        if not set(required).issubset(df.columns):
            continue

        clean_df = df[required].dropna()
        payload = {
            "source": "scheduled",
            "features": [
                {
                    "home_team_id": int(row["home_team_api_id"]),
                    "away_team_id": int(row["away_team_api_id"]),
                    "home_odds": float(row["B365H"]),
                    "draw_odds": float(row["B365D"]),
                    "away_odds": float(row["B365A"]),
                }
                for _, row in clean_df.iterrows()
            ],
        }
        if payload["features"]:
            response = requests.post(API_URL, json=payload, timeout=30)
            if not response.ok:
                raise RuntimeError(
                    f"Prediction API failed for {Path(file_path).name} "
                    f"with status {response.status_code}: {response.text}"
                )

        # Mark file as processed even if it produced no features (e.g. empty CSV),
        # otherwise the DAG will keep reprocessing the same empty batch forever.
        newest_processed = Path(file_path).name

    if newest_processed:
        _write_state(newest_processed)


with DAG(
    dag_id="soccer_prediction",
    start_date=datetime(2026, 1, 1),
    schedule="*/2 * * * *",
    catchup=False,
) as dag:
    check_for_new_data = ShortCircuitOperator(
        task_id="check_for_new_data",
        python_callable=_check_for_new_data,
    )

    make_predictions = PythonOperator(
        task_id="make_predictions",
        python_callable=_make_predictions,
        op_kwargs={"file_paths": "{{ ti.xcom_pull(task_ids='check_for_new_data', key='file_paths') }}"},
    )

    check_for_new_data >> make_predictions
