import pandas as pd
import streamlit as st

from common import (
    AMBER,
    PITCH_GREEN,
    RED,
    SLATE,
    api_get,
    criticality_accent,
    inject_base_css,
    render_cards,
    stat_card,
)

st.set_page_config(page_title="Ingestion Stats", page_icon="📥", layout="wide")
inject_base_css()

st.markdown(
    '<div class="app-hero">'
    "<h1>📥 Ingestion Stats</h1>"
    "<p>Data-quality trends across recent ingestion runs.</p>"
    "</div>",
    unsafe_allow_html=True,
)

refresh = st.button("Refresh now")
if refresh:
    st.cache_data.clear()


@st.cache_data(ttl=5)
def _load() -> pd.DataFrame:
    data = api_get("/ingestion-stats", params={"limit": 500})
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["row_count"] = pd.to_numeric(df["row_count"], errors="coerce").fillna(0).astype(int)
    df["error_count"] = pd.to_numeric(df["error_count"], errors="coerce").fillna(0).astype(int)
    df["error_rate"] = (df["error_count"] / df["row_count"].where(df["row_count"] > 0)).fillna(0.0)
    return df


try:
    df = _load()
except Exception as exc:
    st.error(f"Cannot load ingestion stats: {exc}")
    st.stop()

if df.empty:
    st.info("No ingestion stats yet. Drop a CSV into `data/raw_data` and run the ingestion DAG.")
    st.stop()

# --------------------------------------------------------------------------- #
# Summary cards
# --------------------------------------------------------------------------- #
total_rows = int(df["row_count"].sum())
total_errors = int(df["error_count"].sum())
overall_rate = (total_errors / total_rows * 100) if total_rows else 0.0
rate_accent = RED if overall_rate > 50 else AMBER if overall_rate > 10 else PITCH_GREEN

render_cards(
    [
        stat_card("Files ingested", f"{len(df):,}", "most recent 500", SLATE),
        stat_card("Rows processed", f"{total_rows:,}", "across all runs", SLATE),
        stat_card("Rows with errors", f"{total_errors:,}", "flagged by validation", AMBER),
        stat_card("Overall error rate", f"{overall_rate:.1f}%", "errors / total rows", rate_accent),
    ]
)

st.write("")

# --------------------------------------------------------------------------- #
# Temporal charts
# --------------------------------------------------------------------------- #
ts = df.dropna(subset=["created_at"]).sort_values("created_at").copy()

left, right = st.columns(2)
with left:
    st.markdown("###### Error rate over time")
    if not ts.empty:
        series = ts.set_index("created_at")["error_rate"] * 100
        st.line_chart(series, height=240, color=AMBER)
    else:
        st.caption("No timestamped runs to plot.")

with right:
    st.markdown("###### Rows vs errors over time")
    if not ts.empty:
        rows_errs = ts.set_index("created_at")[["row_count", "error_count"]].rename(
            columns={"row_count": "Rows", "error_count": "Errors"}
        )
        st.bar_chart(rows_errs, height=240, color=[PITCH_GREEN, RED])
    else:
        st.caption("No timestamped runs to plot.")

# --------------------------------------------------------------------------- #
# Status / criticality breakdown
# --------------------------------------------------------------------------- #
st.markdown("###### Ingestions by status")
status_counts = df["status"].value_counts()
st.bar_chart(status_counts, height=220, color=PITCH_GREEN)

# --------------------------------------------------------------------------- #
# Recent runs table (color-coded error rate)
# --------------------------------------------------------------------------- #
st.markdown("###### Recent ingestion runs")
table = df.sort_values("created_at", ascending=False).head(50).copy()
table["error_rate"] = (table["error_rate"] * 100).round(1)
table = table.rename(
    columns={
        "created_at": "Time",
        "file_name": "File",
        "status": "Status",
        "row_count": "Rows",
        "error_count": "Errors",
        "error_rate": "Error %",
        "criticality": "Criticality",
    }
)
st.dataframe(
    table[["Time", "File", "Status", "Rows", "Errors", "Error %", "Criticality"]],
    use_container_width=True,
    hide_index=True,
    height=420,
    column_config={
        "Error %": st.column_config.ProgressColumn(
            "Error %",
            help="Share of rows flagged with errors",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
    },
)
