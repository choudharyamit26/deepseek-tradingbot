import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import json, os, glob
from signal_logger import LOG_DIR, _CSV_FIELDS

st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st_autorefresh(key="auto", interval=60*1000, limit=100)

st.title("📈 Intraday Trading Bot Dashboard")

def load_signals():
    csv_files = sorted(glob.glob(os.path.join(LOG_DIR, "signals_*.csv")))
    if not csv_files:
        return []
    rows = []
    for f in csv_files:
        df = pd.read_csv(f)
        rows.extend(df.to_dict("records"))
    return rows

st.subheader("Recent Signals")
signals = load_signals()
if signals:
    df = pd.DataFrame(signals[-50:])
    st.dataframe(df)
else:
    st.info("No signals yet.")
