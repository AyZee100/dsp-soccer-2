from datetime import datetime, time

import pandas as pd
import streamlit as st

from common import api_get, inject_base_css, prediction_label, team_name


st.set_page_config(page_title="Past Predictions", page_icon="📜", layout="wide")
inject_base_css()

st.markdown(
    '<div class="app-hero">'
    "<h1>📜 Past Predictions</h1>"
    "<p>Readable prediction history for non-technical users.</p>"
    "</div>",
    unsafe_allow_html=True,
)

controls = st.columns([1.1, 1, 1, 1, 1.2])
with controls[0]:
    source = st.selectbox("Source", ["all", "webapp", "scheduled"], index=0)
with controls[1]:
    start_date = st.date_input("Start date", value=datetime.utcnow().date())
with controls[2]:
    end_date = st.date_input("End date", value=datetime.utcnow().date())
with controls[3]:
    view_mode = st.selectbox("View", ["Simple", "Table"], index=0)
with controls[4]:
    show_debug = st.toggle("Debug (raw payload)", value=False)

if start_date > end_date:
    st.error("Start date must be <= End date.")
    st.stop()

refresh_col1, refresh_col2 = st.columns([1, 5])
with refresh_col1:
    refresh_now = st.button("Refresh now")
with refresh_col2:
    st.caption("Use this after batch prediction to fetch latest rows immediately.")

if refresh_now:
    st.cache_data.clear()
    st.rerun()


@st.cache_data(ttl=2)
def _load_predictions(src: str) -> pd.DataFrame:
    data = api_get("/past-predictions", params={"limit": 500, "source": src})
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_convert(None)
    return df


try:
    df = _load_predictions(source)
except Exception as exc:
    st.error(f"Cannot load past predictions: {exc}")
    st.stop()

if df.empty:
    st.info("No predictions found.")
    st.stop()

# Filter by date (inclusive). We treat `created_at` as UTC-naive in UI for simplicity.
start_dt = datetime.combine(start_date, time.min)
end_dt = datetime.combine(end_date, time.max)
df = df.loc[df["created_at"].between(start_dt, end_dt, inclusive="both")].copy()

if df.empty:
    st.info("No predictions in the selected period.")
    st.stop()

# Expand features into dedicated columns for readability.
features_df = pd.json_normalize(df["features"]).add_prefix("feat_")
df_view = pd.concat([df.drop(columns=["features"], errors="ignore"), features_df], axis=1)

# Friendly columns
rename_map = {
    "created_at": "timestamp",
    "prediction": "prediction",
    "proba_home_win": "proba_home_win",
    "model_version": "model",
    "source": "source",
    "feat_home_team_id": "home_team_id",
    "feat_away_team_id": "away_team_id",
    "feat_home_odds": "home_odds",
    "feat_draw_odds": "draw_odds",
    "feat_away_odds": "away_odds",
}
df_view = df_view.rename(columns=rename_map)

df_view["source_label"] = df_view.get("source").map(
    {
        "webapp": "Manual (Streamlit)",
        "scheduled": "Automatic (Airflow)",
    }
).fillna(df_view.get("source"))

# Format for display
if "timestamp" in df_view.columns:
    df_view["timestamp_dt"] = pd.to_datetime(df_view["timestamp"], errors="coerce")
    df_view["timestamp"] = df_view["timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

for c in ["home_odds", "draw_odds", "away_odds"]:
    if c in df_view.columns:
        df_view[c] = pd.to_numeric(df_view[c], errors="coerce").round(2)

df_view["pred_label"] = df_view.get("prediction").apply(prediction_label)
if "proba_home_win" in df_view.columns:
    df_view["proba_home_win"] = pd.to_numeric(df_view["proba_home_win"], errors="coerce")
    df_view["confidence"] = (df_view["proba_home_win"] * 100).round(0).astype("Int64").astype(str) + "%"
else:
    df_view["confidence"] = ""

df_view["home_team"] = df_view.get("home_team_id").apply(team_name)
df_view["away_team"] = df_view.get("away_team_id").apply(team_name)


def _fmt_match(row: pd.Series) -> str:
    ht = row.get("home_team", "Unknown team")
    at = row.get("away_team", "Unknown team")
    return f"{ht} vs {at}"


def _fmt_odds(row: pd.Series) -> str:
    h, d, a = row.get("home_odds"), row.get("draw_odds"), row.get("away_odds")
    parts: list[str] = []
    if pd.notna(h):
        parts.append(f"Home {float(h):.2f}")
    if pd.notna(d):
        parts.append(f"Draw {float(d):.2f}")
    if pd.notna(a):
        parts.append(f"Away {float(a):.2f}")
    return " | ".join(parts) if parts else "N/A"


df_view = df_view.sort_values(by=["timestamp_dt"], ascending=False, na_position="last")

st.caption(f"Showing **{len(df_view)}** predictions.")

if view_mode == "Simple":
    st.markdown("#### Quick explanation")
    st.write(
        "- **Match** uses generated team names (stable from IDs).\n"
        "- **Odds** are bookmaker odds for Home/Draw/Away.\n"
        "- **Prediction** is what the model thinks will happen.\n"
        "- **Manual** = you clicked Predict in the app, **Automatic** = Airflow scheduled run."
    )

    max_cards = 80
    if len(df_view) > max_cards:
        st.info(f"Showing the latest {max_cards}. Use **Table** view for the full list.")

    for _, row in df_view.head(max_cards).iterrows():
        a, b = st.columns([2.2, 1])
        with a:
            st.markdown(f"**{_fmt_match(row)}**")
            st.write(f"Odds: {_fmt_odds(row)}")
        with b:
            st.write(f"Time: {row.get('timestamp', '')}")
            st.write(f"Source: {row.get('source_label', '')}")
            conf = row.get("confidence", "")
            conf_txt = f" ({conf})" if conf and conf != "<NA>%" else ""
            st.write(f"Prediction: **{row.get('pred_label', '')}**{conf_txt}")
        st.divider()
else:
    st.markdown("#### Table view (for scanning)")
    table = df_view.copy()
    keep = [
        "timestamp",
        "source_label",
        "pred_label",
        "confidence",
        "home_team",
        "away_team",
        "home_odds",
        "draw_odds",
        "away_odds",
    ]
    keep = [c for c in keep if c in table.columns]
    table = table[keep].rename(
        columns={
            "timestamp": "Time",
            "source_label": "How it was generated",
            "pred_label": "Model prediction",
            "confidence": "Confidence",
            "home_team": "Home team",
            "away_team": "Away team",
            "home_odds": "Home odds",
            "draw_odds": "Draw odds",
            "away_odds": "Away odds",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True, height=720)

if show_debug:
    with st.expander("Debug: raw API payload (first 5 rows)"):
        st.write(df.head(5).to_dict(orient="records"))
