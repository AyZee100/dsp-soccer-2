from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from common import (
    AMBER,
    PITCH_GREEN,
    RED,
    SLATE,
    api_get,
    criticality_accent,
    get_health,
    inject_base_css,
    render_cards,
    stat_card,
)

st.set_page_config(page_title="Soccer Prediction Dashboard", page_icon="⚽", layout="wide")
inject_base_css()

st.markdown(
    '<div class="app-hero">'
    "<h1>⚽ Soccer Prediction Dashboard</h1>"
    "<p>Live MLOps overview — model serving, prediction activity, and ingestion quality.</p>"
    "</div>",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=10)
def _load_predictions() -> pd.DataFrame:
    try:
        rows = api_get("/past-predictions", params={"limit": 500, "source": "all"})
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_convert(None)
    return df


@st.cache_data(ttl=10)
def _load_ingestion() -> pd.DataFrame:
    try:
        rows = api_get("/ingestion-stats", params={"limit": 200})
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Row 1 — System status
# --------------------------------------------------------------------------- #
health = get_health()
api_ok = health.get("status") == "healthy"
model_version = health.get("model_version") or "—"
model_source = (health.get("model_source") or "unknown").lower()

api_badge = (
    '<span class="badge badge-ok">healthy</span>'
    if api_ok
    else '<span class="badge badge-err">unreachable</span>'
)
source_accent = PITCH_GREEN if model_source == "mlflow" else AMBER
source_sub = "MLflow registry @champion" if model_source == "mlflow" else "local fallback (no champion yet)"

st.markdown("##### System status")
render_cards(
    [
        stat_card("Model service API", api_badge, "FastAPI · :8000", PITCH_GREEN if api_ok else RED),
        stat_card("Champion model", model_version, "served by the API", PITCH_GREEN),
        stat_card("Model source", model_source, source_sub, source_accent),
    ]
)

st.write("")

# --------------------------------------------------------------------------- #
# Row 2 — Prediction activity
# --------------------------------------------------------------------------- #
preds = _load_predictions()

st.markdown("##### Prediction activity")
if preds.empty:
    st.info("No predictions yet. Run the prediction DAG or use the **Prediction** page to generate some.")
else:
    now = datetime.utcnow()
    last_24h = preds[preds["created_at"] >= now - timedelta(hours=24)]
    home_rate = (pd.to_numeric(preds["prediction"], errors="coerce") == 1.0).mean() * 100
    auto = int((preds["source"] == "scheduled").sum())
    manual = int((preds["source"] == "webapp").sum())

    render_cards(
        [
            stat_card("Total predictions", f"{len(preds):,}", "most recent 500", SLATE),
            stat_card("Last 24 hours", f"{len(last_24h):,}", "rolling window", PITCH_GREEN),
            stat_card(
                "Home-win rate",
                f"{home_rate:.0f}%",
                "share predicted Home win",
                AMBER if home_rate > 80 or home_rate < 20 else PITCH_GREEN,
            ),
            stat_card("Automatic / Manual", f"{auto} / {manual}", "Airflow / Streamlit", SLATE),
        ]
    )

    # Temporal trend: predictions per 10-minute bucket, split by predicted class.
    trend = preds.dropna(subset=["created_at"]).copy()
    if not trend.empty:
        trend["bucket"] = trend["created_at"].dt.floor("10min")
        trend["Predicted class"] = pd.to_numeric(trend["prediction"], errors="coerce").map(
            {1.0: "Home win", 0.0: "Draw / Away"}
        )
        pivot = (
            trend.groupby(["bucket", "Predicted class"]).size().unstack(fill_value=0).sort_index()
        )
        st.caption("Predictions over time (10-minute buckets)")
        st.area_chart(pivot, height=240, color=[AMBER, PITCH_GREEN])

st.write("")

# --------------------------------------------------------------------------- #
# Row 3 — Ingestion health
# --------------------------------------------------------------------------- #
ingest = _load_ingestion()

st.markdown("##### Ingestion health")
if ingest.empty:
    st.info("No ingestion runs yet. Drop a CSV into `data/raw_data` and run the ingestion DAG.")
else:
    latest = ingest.sort_values("created_at", ascending=False).iloc[0]
    total_rows = int(ingest["row_count"].sum())
    total_errors = int(ingest["error_count"].sum())
    err_rate = (total_errors / total_rows * 100) if total_rows else 0.0
    last_crit = latest.get("criticality") or "—"

    render_cards(
        [
            stat_card("Files ingested", f"{len(ingest):,}", "most recent 200", SLATE),
            stat_card(
                "Overall error rate",
                f"{err_rate:.1f}%",
                f"{total_errors:,} / {total_rows:,} rows",
                criticality_accent("high" if err_rate > 50 else "medium" if err_rate > 10 else None),
            ),
            stat_card(
                "Last file status",
                str(latest.get("status", "—")),
                str(latest.get("file_name", "")),
                PITCH_GREEN,
            ),
            stat_card("Last criticality", str(last_crit), "latest ingestion run", criticality_accent(last_crit)),
        ]
    )

st.write("")
st.divider()

# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
st.markdown("##### Explore")
nav = st.columns(3)
with nav[0]:
    st.page_link("pages/1_Prediction.py", label="Make a prediction", icon="🎯")
    st.caption("Single match or batch CSV upload.")
with nav[1]:
    st.page_link("pages/2_Past_Predictions.py", label="Past predictions", icon="📜")
    st.caption("Readable history with filters.")
with nav[2]:
    st.page_link("pages/3_Ingestion_Stats.py", label="Ingestion stats", icon="📥")
    st.caption("Data-quality trends per ingestion.")
