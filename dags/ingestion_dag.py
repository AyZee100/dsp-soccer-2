from datetime import datetime
import json
import os
import random
import shutil
from pathlib import Path

import pandas as pd
import requests
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from airflow.providers.postgres.hooks.postgres import PostgresHook


RAW_PATH = Path("/opt/airflow/data/raw_data")
GOOD_PATH = Path("/opt/airflow/data/good_data")
BAD_PATH = Path("/opt/airflow/data/bad_data")
GX_DOCS_DIR = Path("/opt/airflow/data/gx_data_docs")
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL", "").strip()


def _fallback_validation(df: pd.DataFrame) -> dict:
    row_count = len(df)
    required_columns = {"home_team_api_id", "away_team_api_id",
                        "home_team_goal", "away_team_goal", "B365H", "B365D", "B365A"}
    missing_columns = sorted(list(required_columns - set(df.columns)))

    invalid_mask = pd.Series([False] * row_count)
    if missing_columns:
        invalid_mask = pd.Series([True] * row_count)
    else:
        invalid_mask = (
            df["B365H"].isna()
            | (df["home_team_goal"] < 0)
            | (~df["away_team_goal"].apply(lambda x: str(x).lstrip("-").isdigit()))
        )

    return {
        "missing_columns": missing_columns,
        "invalid_mask": invalid_mask,
        "error_summary": {
            "null_B365H": int(df["B365H"].isna().sum()) if "B365H" in df.columns else row_count,
            "negative_home_team_goal": (
                int((df["home_team_goal"] < 0).sum()) if "home_team_goal" in df.columns else row_count
            ),
            "invalid_away_team_goal_type": (
                int((~df["away_team_goal"].apply(lambda x: str(x).lstrip("-").isdigit())).sum())
                if "away_team_goal" in df.columns
                else row_count
            ),
            "missing_required_columns": missing_columns,
        },
    }


def _gx_validate_with_checkpoint(df: pd.DataFrame, file_name: str) -> dict:
    """
    Best-effort GX Core v1 flow:
    - Expectation Suite
    - Validation Definition equivalent via Validator + suite
    - Checkpoint-like execution
    - Data Docs build
    If GX API differs at runtime, caller falls back safely.
    """
    import great_expectations as gx

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name="ingestion_source")
    data_asset = data_source.add_dataframe_asset(name="ingestion_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_df")

    suite_name = "soccer_quality_suite"
    try:
        suite = context.suites.get(name=suite_name)
    except Exception:
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    validator = context.get_validator(batch=batch, expectation_suite=suite)

    # Add expectations through the Validator API (GX 1.15 compatible).
    validator.expect_table_columns_to_match_set(
        column_set=["home_team_api_id", "away_team_api_id",
                    "home_team_goal", "away_team_goal", "B365H", "B365D", "B365A"],
        exact_match=True,
    )
    validator.expect_column_values_to_not_be_null(column="B365H")
    validator.expect_column_values_to_be_between(column="home_team_goal", min_value=0, max_value=20)
    validator.expect_column_values_to_be_of_type(column="away_team_goal", type_="int64")

    # Persist suite into the (ephemeral) context so Data Docs can render it with expectations.
    try:
        context.suites.add_or_update(validator.get_expectation_suite())
    except Exception:
        pass

    validation_result = validator.validate()

    # Build GX Data Docs into a deterministic container folder.
    GX_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    site_name = "gx_local_site"
    site_config = {
        "class_name": "SiteBuilder",
        "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
        "store_backend": {
            "class_name": "TupleFilesystemStoreBackend",
            "base_directory": str(GX_DOCS_DIR),
        },
    }
    # Ephemeral context is recreated each run, so we can safely add the site every time.
    context.add_data_docs_site(site_name=site_name, site_config=site_config)

    built = context.build_data_docs(site_names=[site_name])

    fallback = _fallback_validation(df)
    fallback["gx_success"] = bool(getattr(validation_result, "success", False))
    fallback["gx_results"] = json.loads(validation_result.to_json_dict()) if hasattr(
        validation_result, "to_json_dict") else {}
    # Best-effort: store the index path so the UI can reference it.
    site_info = built.get(site_name, {}) if isinstance(built, dict) else {}
    fallback["report_file_name"] = (
        site_info.get("index_path")
        or site_info.get("site_index_path")
        or site_info.get("site_url")
        or f"{site_name}/index.html"
    )
    return fallback


REQUIRED_COLUMNS = [
    "home_team_api_id", "away_team_api_id",
    "home_team_goal", "away_team_goal", "B365H", "B365D", "B365A",
]


def _categorize_errors(df: pd.DataFrame, missing_columns: list) -> dict:
    """
    Bucket row-level data quality problems into the 5 defense categories:
    completeness, validity, consistency, schema, type.
    Returns per-category boolean masks + the union mask used for the good/bad split.
    """
    n = len(df)
    false = pd.Series([False] * n, index=df.index)

    # SCHEMA: a required column is absent -> the whole batch is structurally invalid.
    schema_mask = pd.Series([bool(missing_columns)] * n, index=df.index)

    # COMPLETENESS: required value is null/empty.
    completeness_mask = false.copy()
    for col in ["B365H", "B365D", "B365A", "home_team_api_id", "away_team_api_id"]:
        if col in df.columns:
            completeness_mask = completeness_mask | df[col].isna()

    # TYPE: a value is the wrong type (e.g. "ERROR" injected into away_team_goal).
    type_mask = false.copy()
    if "away_team_goal" in df.columns:
        type_mask = type_mask | df["away_team_goal"].apply(
            lambda x: pd.notna(x) and not str(x).lstrip("-").isdigit()
        )

    # VALIDITY: value present and right type but out of allowed range
    # (negative goals, non-positive odds).
    validity_mask = false.copy()
    if "home_team_goal" in df.columns:
        validity_mask = validity_mask | (
            pd.to_numeric(df["home_team_goal"], errors="coerce").fillna(0) < 0
        )
    for col in ["B365H", "B365D", "B365A"]:
        if col in df.columns:
            validity_mask = validity_mask | (
                pd.to_numeric(df[col], errors="coerce").fillna(1) <= 0
            )

    # CONSISTENCY: duplicated rows (keep the first occurrence, flag the copies).
    consistency_mask = df.duplicated(keep="first")
    if not isinstance(consistency_mask, pd.Series):
        consistency_mask = false.copy()

    invalid_mask = (
        schema_mask | completeness_mask | type_mask | validity_mask | consistency_mask
    )

    return {
        "counts": {
            "completeness": int(completeness_mask.sum()),
            "validity": int(validity_mask.sum()),
            "consistency": int(consistency_mask.sum()),
            "schema": int(schema_mask.sum()),
            "type": int(type_mask.sum()),
        },
        "invalid_mask": invalid_mask,
    }


def _send_teams_alert(message: str) -> None:
    if not TEAMS_WEBHOOK_URL:
        print("Teams webhook not configured. Skipping Teams alert.")
        return
    try:
        adaptive_card = {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.4",
            "body": [
                {"type": "TextBlock", "text": "Data Quality Alert", "weight": "bolder", "size": "medium"},
                {"type": "TextBlock", "text": message, "wrap": True},
            ],
        }
        # The provided flow expects a valid AdaptiveCard JSON at the root.
        response = requests.post(TEAMS_WEBHOOK_URL, json=adaptive_card, timeout=10)
        response.raise_for_status()
    except Exception as exc:
        print(f"Failed to send Teams alert: {exc}")


with DAG(
    dag_id="soccer_ingestion",
    start_date=datetime(2026, 1, 1),
    schedule="*/1 * * * *",
    catchup=False,
) as dag:
    @task
    def read_data():
        files = [p for p in RAW_PATH.glob("*.csv")]
        if not files:
            raise AirflowSkipException("No raw data files available")
        selected = random.choice(files)
        df = pd.read_csv(selected)
        return {"file_path": str(selected), "file_name": selected.name, "rows": df.to_dict(orient="records")}

    @task
    def validate_data(payload: dict):
        rows = payload["rows"]
        df = pd.DataFrame(rows)
        row_count = len(df)
        try:
            validation_details = _gx_validate_with_checkpoint(df, payload["file_name"])
        except Exception:
            validation_details = _fallback_validation(df)
            validation_details["gx_success"] = False
            validation_details["gx_results"] = {}
            validation_details["report_file_name"] = ""

        missing_columns = validation_details["missing_columns"]

        # Categorize errors into the 5 defense buckets and use that as the
        # authoritative invalid mask for the good/bad split + stats.
        categorized = _categorize_errors(df, missing_columns)
        invalid_mask = categorized["invalid_mask"]
        error_categories = categorized["counts"]

        invalid_rows = int(invalid_mask.sum())
        if missing_columns or (row_count > 0 and invalid_rows / row_count > 0.5):
            criticality = "High"
        elif row_count > 0 and invalid_rows / row_count >= 0.1:
            criticality = "Medium"
        elif invalid_rows > 0:
            criticality = "Low"
        else:
            criticality = "None"

        return {
            **payload,
            "row_count": row_count,
            "invalid_rows": invalid_rows,
            "missing_columns": missing_columns,
            "criticality": criticality,
            "error_categories": error_categories,
            "error_summary": validation_details["error_summary"],
            "gx_success": validation_details["gx_success"],
            "gx_results": validation_details["gx_results"],
            "report_file_name": validation_details["report_file_name"],
            "good_rows": df.loc[~invalid_mask].to_dict(orient="records"),
            "bad_rows": df.loc[invalid_mask].to_dict(orient="records"),
        }

    @task
    def save_statistics(validation: dict):
        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        pg_hook.run(
            """
            CREATE TABLE IF NOT EXISTS ingestion_stats (
                id SERIAL PRIMARY KEY,
                file_name TEXT NOT NULL,
                status TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                error_count INTEGER NOT NULL,
                criticality TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        # Backward-compatible fix when table existed without DB-level default.
        pg_hook.run(
            """
            ALTER TABLE ingestion_stats
            ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP;
            """
        )
        status = "Good" if validation["invalid_rows"] == 0 else "Bad"
        pg_hook.run(
            """
            INSERT INTO ingestion_stats (file_name, status, row_count, error_count, criticality, created_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP);
            """,
            parameters=(
                validation["file_name"],
                status,
                validation["row_count"],
                validation["invalid_rows"],
                validation["criticality"],
            ),
        )

        # Per-category error counts (powers the Grafana data-quality dashboard).
        pg_hook.run(
            """
            CREATE TABLE IF NOT EXISTS ingestion_errors (
                id SERIAL PRIMARY KEY,
                file_name TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                completeness INTEGER NOT NULL DEFAULT 0,
                validity INTEGER NOT NULL DEFAULT 0,
                consistency INTEGER NOT NULL DEFAULT 0,
                schema_errors INTEGER NOT NULL DEFAULT 0,
                type_errors INTEGER NOT NULL DEFAULT 0,
                total_errors INTEGER NOT NULL DEFAULT 0,
                criticality TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cats = validation.get("error_categories", {})
        pg_hook.run(
            """
            INSERT INTO ingestion_errors (
                file_name, row_count, completeness, validity, consistency,
                schema_errors, type_errors, total_errors, criticality, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP);
            """,
            parameters=(
                validation["file_name"],
                validation["row_count"],
                cats.get("completeness", 0),
                cats.get("validity", 0),
                cats.get("consistency", 0),
                cats.get("schema", 0),
                cats.get("type", 0),
                validation["invalid_rows"],
                validation["criticality"],
            ),
        )

    @task
    def send_alerts(validation: dict):
        # Defense rules: send only for Medium/High to avoid alert fatigue.
        if validation["criticality"] in {"High", "Medium"}:
            docs_base = os.getenv("GX_DATA_DOCS_URL", "http://localhost:8081").rstrip("/")
            report = validation["report_file_name"] or "index.html"
            report_url = report if str(report).startswith("http") else f"{docs_base}/{report}"
            message = (
                f"Data Quality Alert\n"
                f"- File: {validation['file_name']}\n"
                f"- Criticality: {validation['criticality']}\n"
                f"- Invalid rows: {validation['invalid_rows']}/{validation['row_count']}\n"
                f"- Categories: {validation.get('error_categories', {})}\n"
                f"- Data Docs: {report_url}\n"
                f"- Summary: {validation['error_summary']}"
            )
            print(f"[ALERT] {message}")
            _send_teams_alert(message)

    @task
    def split_and_save_data(validation: dict):
        GOOD_PATH.mkdir(parents=True, exist_ok=True)
        BAD_PATH.mkdir(parents=True, exist_ok=True)
        file_path = Path(validation["file_path"])
        file_name = validation["file_name"]

        if validation["invalid_rows"] == 0:
            shutil.move(str(file_path), str(GOOD_PATH / file_name))
            return
        if validation["invalid_rows"] == validation["row_count"]:
            shutil.move(str(file_path), str(BAD_PATH / file_name))
            return

        pd.DataFrame(validation["good_rows"]).to_csv(GOOD_PATH / file_name, index=False)
        pd.DataFrame(validation["bad_rows"]).to_csv(BAD_PATH / file_name, index=False)
        file_path.unlink(missing_ok=True)

    validation = validate_data(read_data())
    save_statistics(validation)
    send_alerts(validation)
    split_and_save_data(validation)
