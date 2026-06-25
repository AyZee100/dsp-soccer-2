"""
Demo fallback: promote the latest registered model version of `soccer_model`
to the @champion alias in MLflow.

Use this only if the training DAG's `promote_to_champion` task fails to set the
alias (the model still trained and registered). Run it from inside the
model_service container, then reload the API:

    docker compose exec model_service python /app/data/set_champion.py
    curl.exe -X POST http://localhost:8000/reload-model
    curl.exe http://localhost:8000/health
"""
import os

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "soccer_model")
ALIAS = os.getenv("MODEL_ALIAS", "champion")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise SystemExit(f"No registered versions found for model '{MODEL_NAME}'.")

    latest = max(int(v.version) for v in versions)
    # set_registered_model_alias requires the version as a string.
    client.set_registered_model_alias(MODEL_NAME, ALIAS, str(latest))
    print(f"Set alias @{ALIAS} -> {MODEL_NAME} version {latest}")


if __name__ == "__main__":
    main()
