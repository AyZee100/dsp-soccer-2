import os
import random
from typing import Any

import pandas as pd
import requests
import streamlit as st


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


# --------------------------------------------------------------------------- #
# Presentation helpers (shared look & feel across pages)
# --------------------------------------------------------------------------- #

PITCH_GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#ef4444"
SLATE = "#64748b"


def get_health() -> dict:
    """Return the API /health payload, or an empty dict if unreachable."""
    try:
        return api_get("/health")
    except Exception:
        return {}


def inject_base_css() -> None:
    """Inject the shared stylesheet used by hero banners and stat cards."""
    st.markdown(
        """
        <style>
        .app-hero {
            background: linear-gradient(135deg, #064e3b 0%, #10b981 100%);
            color: #ffffff;
            padding: 1.4rem 1.6rem;
            border-radius: 14px;
            margin-bottom: 1.3rem;
            box-shadow: 0 6px 18px rgba(16,185,129,0.18);
        }
        .app-hero h1 { color:#fff; margin:0; font-size:1.7rem; }
        .app-hero p  { color:#d1fae5; margin:.35rem 0 0 0; font-size:.95rem; }
        .stat-card {
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-left:5px solid var(--accent, #10b981);
            border-radius:12px;
            padding:.9rem 1.05rem;
            height:100%;
            box-shadow: 0 1px 3px rgba(15,23,42,0.05);
        }
        .stat-card .label { color:#64748b; font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; }
        .stat-card .value { color:#0f172a; font-size:1.45rem; font-weight:700; margin-top:.2rem; line-height:1.15; }
        .stat-card .sub   { color:#94a3b8; font-size:.78rem; margin-top:.2rem; }
        .badge { display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.78rem; font-weight:600; }
        .badge-ok   { background:#dcfce7; color:#166534; }
        .badge-warn { background:#fef9c3; color:#854d0e; }
        .badge-err  { background:#fee2e2; color:#991b1b; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: Any, sub: str = "", accent: str = PITCH_GREEN) -> str:
    """Return HTML for a single accent-bordered stat card."""
    return (
        f'<div class="stat-card" style="--accent:{accent}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div>'
        f"</div>"
    )


def render_cards(cards: list[str]) -> None:
    """Render a row of stat cards in equal-width columns."""
    cols = st.columns(len(cards))
    for col, html in zip(cols, cards):
        with col:
            st.markdown(html, unsafe_allow_html=True)


def criticality_accent(criticality: str | None) -> str:
    """Map an ingestion criticality string to a card accent color."""
    value = (criticality or "").lower()
    if value in {"high", "critical"}:
        return RED
    if value in {"medium", "warning"}:
        return AMBER
    return PITCH_GREEN
