import os
import random
from typing import Any

import pandas as pd
import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://model_service:8000")
REQUIRED_COLUMNS = ["home_team_id", "away_team_id", "home_odds", "draw_odds", "away_odds"]
ALIAS_TO_REQUIRED = {
    "home_team_api_id": "home_team_id",
    "away_team_api_id": "away_team_id",
    "B365H": "home_odds",
    "B365D": "draw_odds",
    "B365A": "away_odds",
}


def api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any], timeout: int = 30) -> Any:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def normalize_batch_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Accept both webapp and ingestion column naming conventions."""
    normalized = df.rename(columns=ALIAS_TO_REQUIRED)
    return normalized


def build_batch_payload(df: pd.DataFrame, source: str = "webapp") -> dict[str, Any]:
    normalized = normalize_batch_dataframe(df)
    cleaned = normalized[REQUIRED_COLUMNS].dropna()
    if cleaned.empty:
        raise ValueError("No valid rows to predict after filtering empty values in required columns")
    return {"source": source, "features": cleaned.to_dict(orient="records")}


_CITIES = [
    "Paris",
    "Lyon",
    "Marseille",
    "Toulouse",
    "Nice",
    "Nantes",
    "Lille",
    "Rennes",
    "Bordeaux",
    "Montpellier",
    "Strasbourg",
    "Reims",
    "Lens",
    "Brest",
    "Metz",
    "Dijon",
    "Angers",
    "Grenoble",
    "Rouen",
    "Caen",
    "Tours",
    "Nancy",
    "Amiens",
]
_MASCOTS = [
    "Falcons",
    "Wolves",
    "Eagles",
    "Tigers",
    "Lions",
    "Sharks",
    "Bulls",
    "Ravens",
    "Panthers",
    "Foxes",
    "Dragons",
    "Hawks",
    "Giants",
    "Pirates",
    "Knights",
    "Spartans",
    "Titans",
]


def team_name(team_id: int | float | str | None) -> str:
    """
    Deterministic team-name generator from an ID.
    We don't have a real mapping in the dataset, so we create stable, human-friendly names.
    """
    if team_id is None or (isinstance(team_id, float) and pd.isna(team_id)):
        return "Unknown team"
    try:
        tid = int(team_id)
    except Exception:
        return f"Team {team_id}"

    rng = random.Random(tid)  # stable across runs
    return f"{rng.choice(_CITIES)} {rng.choice(_MASCOTS)}"


def prediction_label(prediction_value: Any) -> str:
    """
    Model outputs 1.0 for 'home win' class, 0.0 otherwise (draw or away win).
    """
    try:
        p = float(prediction_value)
    except Exception:
        return "Unknown"
    if p == 1.0:
        return "Home win"
    if p == 0.0:
        return "Not home win (Draw/Away)"
    return "Unknown"
