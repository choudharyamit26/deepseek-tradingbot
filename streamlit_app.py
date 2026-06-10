"""Trading bot dashboard: live reversal watch on open positions,
recent signals, and reflection-agent history. Read-only — no order actions.

Run: streamlit run streamlit_app.py
"""

import glob
import json
import os

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from indicators import calculate_technical_indicators
from reversal_detector import detect_reversals
from signal_logger import LOG_DIR

st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st_autorefresh(key="auto", interval=60 * 1000, limit=200)

st.title("📈 Intraday Trading Bot Dashboard")

HYPOTHESES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "kronos_integrated_bot", "state", "hypotheses.jsonl")


@st.cache_resource
def get_dhan():
    try:
        from dhan_integration import DhanStockTradingBot
        return DhanStockTradingBot()
    except Exception as e:
        st.warning(f"Dhan client unavailable: {e}")
        return None


def load_signals():
    csv_files = sorted(glob.glob(os.path.join(LOG_DIR, "signals_*.csv")))
    rows = []
    for f in csv_files[-5:]:  # last 5 sessions
        try:
            rows.extend(pd.read_csv(f).to_dict("records"))
        except Exception:
            continue
    return rows


def load_hypotheses():
    records = []
    if os.path.isfile(HYPOTHESES_FILE):
        with open(HYPOTHESES_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


tab_positions, tab_signals, tab_reflection = st.tabs(
    ["🔄 Open Positions — Reversal Watch", "📊 Recent Signals", "🧠 Reflection History"]
)

# ── Tab 1: live reversal detection on open positions ─────────────────────────
with tab_positions:
    dhan = get_dhan()
    if dhan is None:
        st.info("Set DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN in .env to enable live positions.")
    else:
        try:
            positions = dhan.fetch_positions()
        except Exception as e:
            positions = {}
            st.error(f"Could not fetch positions: {e}")

        if not positions:
            st.info("No open positions.")
        else:
            rows = []
            details = {}
            for symbol, pos in positions.items():
                is_buy = pos["transaction_type"] == dhan.dhan.BUY
                row = {
                    "Symbol": symbol,
                    "Side": "LONG" if is_buy else "SHORT",
                    "Qty": pos["quantity"],
                    "Entry": pos["entry_price"],
                    "uPnL": pos["unrealized_pnl"],
                    "Reversal Score": None,
                    "Recommendation": "n/a",
                }
                try:
                    df = dhan.get_historical_data(pos["security_id"], "3minute", min_bars=20)
                    if df is not None and len(df) >= 20:
                        ind = calculate_technical_indicators(df)
                        report = detect_reversals(df.copy(), is_buy=is_buy, indicators=ind)
                        row["Reversal Score"] = report.score
                        row["Recommendation"] = report.recommendation
                        details[symbol] = report
                    else:
                        row["Recommendation"] = "insufficient bars"
                except Exception as e:
                    row["Recommendation"] = f"error: {e}"
                rows.append(row)

            df_pos = pd.DataFrame(rows)

            def _color(rec: str):
                if "EXIT" in str(rec):
                    return "background-color: #7f1d1d"
                if "CAUTION" in str(rec):
                    return "background-color: #78350f"
                return ""

            st.dataframe(
                df_pos.style.map(_color, subset=["Recommendation"]),
                use_container_width=True,
            )

            for symbol, report in details.items():
                if report.signals:
                    with st.expander(f"{symbol}: contributing signals "
                                     f"(score {report.score})"):
                        for s in report.signals:
                            st.write(f"- **{s.name}** (severity {s.severity}): {s.description}")

# ── Tab 2: recent signals from CSV logs ──────────────────────────────────────
with tab_signals:
    signals = load_signals()
    if signals:
        df_sig = pd.DataFrame(signals[-100:])
        show_cols = [c for c in ["timestamp", "symbol", "signal_type", "direction",
                                 "entry_price", "exit_price", "confidence", "pnl",
                                 "mode", "market_regime"] if c in df_sig.columns]
        st.dataframe(df_sig[show_cols].iloc[::-1], use_container_width=True)

        closed = df_sig[pd.to_numeric(df_sig.get("pnl"), errors="coerce").notna()
                        & (df_sig.get("exit_price", "").astype(str).str.strip() != "")]
        if len(closed):
            pnl = pd.to_numeric(closed["pnl"], errors="coerce").dropna()
            c1, c2, c3 = st.columns(3)
            c1.metric("Closed trades (last 5 sessions)", len(pnl))
            c2.metric("Win rate", f"{(pnl > 0).mean() * 100:.0f}%")
            c3.metric("Total PnL", f"{pnl.sum():+,.2f}")
    else:
        st.info("No signals yet.")

# ── Tab 3: reflection-agent history ──────────────────────────────────────────
with tab_reflection:
    hyps = load_hypotheses()
    if not hyps:
        st.info("No reflection records yet. Run: python -m kronos_integrated_bot.run_reflection")
    else:
        df_h = pd.DataFrame([{
            "When": h.get("timestamp"),
            "Action": h.get("action", "change"),
            "Version": f"{h.get('old_version')} → {h.get('new_version')}",
            "Parameter": h.get("parameter_changed"),
            "Change": (f"{h.get('old_value')} → {h.get('new_value')}"
                       if h.get("parameter_changed") else ""),
            "Win rate at decision": (h.get("metrics") or {}).get("win_rate"),
            "Closed trades": (h.get("metrics") or {}).get("closed_trades"),
        } for h in hyps])
        st.dataframe(df_h.iloc[::-1], use_container_width=True)

        for h in reversed(hyps[-10:]):
            label = h.get("parameter_changed") or h.get("action", "record")
            with st.expander(f"{h.get('timestamp')} — {label}"):
                st.write(f"**Hypothesis:** {h.get('hypothesis', '')}")
                if h.get("reasoning"):
                    st.write(f"**Reasoning:** {h.get('reasoning')}")
                if h.get("metrics"):
                    st.json(h["metrics"])
