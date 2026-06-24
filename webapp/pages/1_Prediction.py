import pandas as pd
import streamlit as st

from common import (
    REQUIRED_COLUMNS, api_get, api_post, build_batch_payload,
    inject_base_css, normalize_batch_dataframe, prediction_label, team_name
)

st.set_page_config(page_title="Prediction", page_icon="🎯", layout="wide")
inject_base_css()

st.markdown(
    '<div class="app-hero">'
    "<h1>🎯 Prediction</h1>"
    "<p>Predict a single match or score a batch of fixtures from a CSV.</p>"
    "</div>",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def _recent_team_ids() -> list[int]:
    # Build team options from recent predictions so UI feels realistic.
    ids: set[int] = set()
    try:
        rows = api_get("/past-predictions", params={"limit": 500, "source": "all"})
        for row in rows:
            feats = row.get("features", {}) or {}
            for key in ("home_team_id", "away_team_id"):
                value = feats.get(key)
                if value is not None:
                    try:
                        ids.add(int(value))
                    except Exception:
                        pass
    except Exception:
        pass

    if not ids:
        ids = {9987, 10000, 9993, 8342, 9851, 10249}
    return sorted(ids)


st.subheader("Single prediction")
with st.form("single_prediction_form"):
    team_ids = _recent_team_ids()
    home_team_id = st.selectbox(
        "Home team",
        options=team_ids,
        index=0,
        format_func=lambda tid: team_name(tid),
    )
    away_team_id = st.selectbox(
        "Away team",
        options=team_ids,
        index=1 if len(team_ids) > 1 else 0,
        format_func=lambda tid: team_name(tid),
    )
    if home_team_id == away_team_id:
        st.warning("Choose two different teams.")
    home_odds = st.number_input("Home odds", value=2.0)
    draw_odds = st.number_input("Draw odds", value=3.0)
    away_odds = st.number_input("Away odds", value=3.5)
    st.caption(f"Match preview: **{team_name(home_team_id)} vs {team_name(away_team_id)}**")
    submit = st.form_submit_button("Predict")

if submit:
    if home_team_id == away_team_id:
        st.error("Home team and Away team must be different.")
        st.stop()
    payload = {
        "source": "webapp",
        "features": [
            {
                "home_team_id": int(home_team_id),
                "away_team_id": int(away_team_id),
                "home_odds": float(home_odds),
                "draw_odds": float(draw_odds),
                "away_odds": float(away_odds),
            }
        ],
    }
    try:
        result = api_post("/predict", payload)
        preds = pd.DataFrame(result.get("predictions", []))
        if preds.empty:
            st.warning("No prediction returned.")
        else:
            row = preds.iloc[0].to_dict()
            features = row.get("features", {}) or {}
            home_id = features.get("home_team_id")
            away_id = features.get("away_team_id")
            st.markdown("#### Result")
            st.markdown(f"**{team_name(home_id)} vs {team_name(away_id)}**")
            home_o = features.get('home_odds')
            draw_o = features.get('draw_odds')
            away_o = features.get('away_odds')
            label = prediction_label(row.get('prediction'))
            proba = row.get("proba_home_win")

            res_left, res_right = st.columns([1.3, 1])
            with res_left:
                oc = st.columns(3)
                oc[0].metric("Home odds", f"{home_o}")
                oc[1].metric("Draw odds", f"{draw_o}")
                oc[2].metric("Away odds", f"{away_o}")
                if label == "Home win":
                    st.success(f"Prediction: **{label}**")
                else:
                    st.info(f"Prediction: **{label}**")
            with res_right:
                if proba is not None:
                    pct = max(0.0, min(1.0, float(proba)))
                    st.metric("Home-win confidence", f"{round(pct * 100)}%")
                    st.progress(pct)
                    st.caption("Model-estimated probability of a home win.")
                else:
                    st.caption("This model does not expose class probabilities.")

            with st.expander("Details (table)"):
                st.dataframe(preds, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"API error: {exc}")

st.divider()
st.subheader("Batch prediction (CSV upload)")
st.caption(
    "Required columns: home_team_id, away_team_id, home_odds, draw_odds, away_odds "
    "(ingestion aliases also supported: home_team_api_id, away_team_api_id, B365H, B365D, B365A)"
)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
chunk_size = st.selectbox("Batch chunk size", [25, 50, 100, 200], index=2, help="Smaller chunks avoid API timeout")
if st.button("Predict batch") and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        normalized = normalize_batch_dataframe(df)
        required = set(REQUIRED_COLUMNS)
        if not required.issubset(normalized.columns):
            missing = sorted(required - set(normalized.columns))
            st.error(f"Missing required columns: {missing}")
        else:
            payload = build_batch_payload(normalized)
            features = payload.get("features", [])
            if not features:
                st.warning("No valid rows to predict.")
                st.stop()

            all_predictions: list[dict] = []
            progress = st.progress(0, text="Running batch predictions...")
            total_chunks = (len(features) + chunk_size - 1) // chunk_size

            for i in range(total_chunks):
                chunk = features[i * chunk_size: (i + 1) * chunk_size]
                chunk_payload = {"source": payload.get("source", "webapp"), "features": chunk}
                result = api_post("/predict", chunk_payload, timeout=120)
                all_predictions.extend(result.get("predictions", []))
                progress.progress((i + 1) / total_chunks, text=f"Processed chunk {i + 1}/{total_chunks}")

            progress.empty()
            preds = pd.DataFrame(all_predictions)
            if preds.empty:
                st.warning("No predictions returned.")
            else:
                # Expand features for readability
                features_df = pd.json_normalize(preds["features"]).add_prefix("feat_")
                out = pd.concat([preds.drop(columns=["features"], errors="ignore"), features_df], axis=1)
                out["Home team"] = out.get("feat_home_team_id").apply(team_name)
                out["Away team"] = out.get("feat_away_team_id").apply(team_name)
                out["Prediction"] = out.get("prediction").apply(prediction_label)
                if "proba_home_win" in out.columns:
                    out["Confidence"] = (
                        pd.to_numeric(out["proba_home_win"], errors="coerce") * 100
                    ).round(0).astype("Int64")
                keep = [
                    "id",
                    "created_at",
                    "source",
                    "Prediction",
                    "Confidence",
                    "Home team",
                    "Away team",
                    "feat_home_odds",
                    "feat_draw_odds",
                    "feat_away_odds",
                ]
                keep = [c for c in keep if c in out.columns]
                out = out[keep].rename(
                    columns={
                        "created_at": "Time",
                        "source": "Source",
                        "feat_home_odds": "Home odds",
                        "feat_draw_odds": "Draw odds",
                        "feat_away_odds": "Away odds",
                    }
                )
                st.markdown("#### Batch results")
                st.dataframe(out, use_container_width=True, hide_index=True, height=720)
                st.success("Batch prediction completed.")
                st.info(
                    "Go to **Past Predictions** to see full history. "
                    "If needed, click **Refresh now** on that page."
                )
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")
