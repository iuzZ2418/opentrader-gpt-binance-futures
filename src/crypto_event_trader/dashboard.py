from __future__ import annotations

import pandas as pd
import streamlit as st

from crypto_event_trader.config import Settings
from crypto_event_trader.database import Repository

st.set_page_config(page_title="Crypto Event Trader", page_icon="📊", layout="wide")
settings = Settings.from_env()
repository = Repository(settings.sqlite_path())
repository.initialize(settings.initial_cash)

st.title("Crypto Event Research & Paper Trading")
st.caption("Research system — not investment advice. Execution is paper-only by default.")

portfolio = repository.portfolio()
left, middle, right, fourth = st.columns(4)
left.metric("Equity", f"${portfolio['equity']:,.2f}")
middle.metric("Cash", f"${portfolio['cash']:,.2f}")
right.metric("Gross exposure", f"${portfolio['gross_exposure']:,.2f}")
fourth.metric("Drawdown", f"{portfolio['drawdown']:.2%}")

signals = repository.list_signals(200)
orders = repository.list_orders(200)
curve = repository.equity_curve(2000)

signal_tab, portfolio_tab, execution_tab = st.tabs(["Signals", "Portfolio", "Execution log"])
with signal_tab:
    if signals:
        signal_frame = pd.DataFrame(signals)
        st.bar_chart(signal_frame.groupby("threshold_bucket").size())
        st.dataframe(
            signal_frame[
                [
                    "created_at",
                    "symbol",
                    "event_type",
                    "polarity",
                    "score",
                    "threshold_bucket",
                    "title",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No signals yet. Run the sample pipeline or ingest a configured feed.")

with portfolio_tab:
    if curve:
        curve_frame = pd.DataFrame(curve)
        curve_frame["recorded_at"] = pd.to_datetime(curve_frame["recorded_at"])
        st.line_chart(curve_frame.set_index("recorded_at")[["equity"]])
    st.dataframe(pd.DataFrame(portfolio["positions"]), hide_index=True, use_container_width=True)

with execution_tab:
    if orders:
        st.dataframe(pd.DataFrame(orders), hide_index=True, use_container_width=True)
    else:
        st.info("No paper orders have been generated.")
