"""
Shared ML helpers for the training pipeline (used by dags/training_dag.py).

Kept dependency-light and import-safe so the Airflow DAG parser does not choke:
heavy libs (mlflow, sklearn) are imported lazily inside the functions that use them.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Model input features, in the exact order the API/model expects.
FEATURES = ["home_team_id", "away_team_id", "home_odds", "draw_odds", "away_odds"]

# Raw soccer CSV column -> model feature name.
RAW_TO_FEATURE = {
    "home_team_api_id": "home_team_id",
    "away_team_api_id": "away_team_id",
    "B365H": "home_odds",
    "B365D": "draw_odds",
    "B365A": "away_odds",
}

NUMERIC_DRIFT_FEATURES = ["home_odds", "draw_odds", "away_odds"]
CATEGORICAL_DRIFT_FEATURE = "favorite"  # engineered: cheapest-odds outcome (HOME/DRAW/AWAY)


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Turn a raw soccer batch into model features X and the home-win target y."""
    df = df.rename(columns=RAW_TO_FEATURE)

    needed = set(FEATURES) | {"home_team_goal", "away_team_goal"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for training: {sorted(missing)}")

    for col in FEATURES + ["home_team_goal", "away_team_goal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + ["home_team_goal", "away_team_goal"])
    # Odds must be positive to be meaningful.
    df = df[(df["home_odds"] > 0) & (df["draw_odds"] > 0) & (df["away_odds"] > 0)]

    X = df[FEATURES].reset_index(drop=True)
    y = (df["home_team_goal"] > df["away_team_goal"]).astype(int).reset_index(drop=True)
    return X, y


def favorite_from_odds(home_odds: float, draw_odds: float, away_odds: float) -> str:
    """Cheapest odds = bookmaker's favorite. Engineered categorical feature."""
    smallest = min(home_odds, draw_odds, away_odds)
    if home_odds == smallest:
        return "HOME"
    if draw_odds == smallest:
        return "DRAW"
    return "AWAY"


def load_good_data(good_dir: str | Path) -> pd.DataFrame:
    """Concatenate every CSV currently in good_data into one frame."""
    good_dir = Path(good_dir)
    files = sorted(good_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_feature_stats(X: pd.DataFrame, model_version: str) -> List[dict]:
    """
    Tidy (long) baseline rows for the drift dashboard.
    One row per (feature, metric/category) so Grafana can compare serving vs baseline.
    """
    rows: List[dict] = []

    for feat in NUMERIC_DRIFT_FEATURES:
        col = X[feat].astype(float)
        for metric, value in {
            "mean": col.mean(),
            "std": col.std(ddof=0),
            "min": col.min(),
            "max": col.max(),
        }.items():
            rows.append({
                "model_version": model_version,
                "feature_name": feat,
                "feature_type": "numeric",
                "category": None,
                "metric": metric,
                "value": float(value),
            })

    fav = X.apply(
        lambda r: favorite_from_odds(r["home_odds"], r["draw_odds"], r["away_odds"]),
        axis=1,
    )
    freqs = fav.value_counts(normalize=True)
    for category in ["HOME", "DRAW", "AWAY"]:
        rows.append({
            "model_version": model_version,
            "feature_name": CATEGORICAL_DRIFT_FEATURE,
            "feature_type": "categorical",
            "category": category,
            "metric": "freq",
            "value": float(freqs.get(category, 0.0)),
        })

    return rows


def train_random_forest(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25, seed: int = 42):
    """Train + evaluate a RandomForest. Returns (model, X_test, y_test, metrics)."""
    import time

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    model = RandomForestClassifier(n_estimators=200, random_state=seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, zero_division=0))

    # Mean inference time per single-row prediction (ms).
    sample = X_test.iloc[[0]] if len(X_test) else X.iloc[[0]]
    t0 = time.perf_counter()
    for _ in range(50):
        model.predict(sample)
    inference_ms = (time.perf_counter() - t0) / 50 * 1000.0

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "inference_ms": float(inference_ms),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_total": int(len(X)),
    }
    return model, X_test, y_test, metrics
