import pandas as pd
import streamlit as st

from common import api_get


st.title("Ingestion Stats")

try:
    data = api_get("/ingestion-stats", params={"limit": 500})
    df = pd.DataFrame(data)
    if df.empty:
        st.info("No ingestion stats yet.")
    else:
        st.dataframe(df, use_container_width=True)
except Exception as exc:
    st.error(f"Cannot load ingestion stats: {exc}")
