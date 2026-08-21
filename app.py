"""
Streamlit App – Phase 4 Portfolio Construction & Risk Analytics
Integrates Phase 1 (download) + Phase 2 (cleaning) + Phase 3 (construction) + Phase 4 (Historical VaR)
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_collection.download_portfolio import download_asset_prices
from src.data_cleaning.cleaning import run_data_cleaning_pipeline, data_overview
from src.portfolio_construction.portfolio import (
    construct_portfolio,
    resolve_target_weights,
    PortfolioResult,
)
from src.historical_simulation_var.historical_var import HistoricalSimulation

st.set_page_config(
    page_title="Portfolio Risk & Construction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Portfolio Construction & Risk Analytics")
st.caption(
    "Build multi-asset portfolios with configurable rebalancing and evaluate "
    "historical VaR, Expected Shortfall, and rolling tail risk."
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Inputs")

    tickers_raw = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN, META",
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date",
            value=date.today() - timedelta(days=365 * 3),
            max_value=date.today() - timedelta(days=30),
        )
    with col2:
        end_date = st.date_input(
            "End date",
            value=date.today() - timedelta(days=1),
            max_value=date.today(),
        )

    initial_value = st.number_input(
        "Initial portfolio value ($)",
        min_value=1_000.0,
        max_value=100_000_000.0,
        value=100_000.0,
        step=10_000.0,
        format="%.0f",
    )

    rebalance_freq = st.selectbox(
        "Rebalancing frequency",
        options=["none", "daily", "weekly", "monthly", "quarterly"],
        index=3,
    )

    transaction_cost_bps = st.number_input(
        "Transaction cost (bps)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
    )
    volatility_window = st.slider("Rolling volatility window", 10, 750, 21)

    st.divider()
    st.subheader("🛡️ Risk Parameters (Phase 4)")
    conf_str = st.selectbox("Confidence Level", ['95%', '99%', '99.5%'], index=0)
    confidence_level = float(conf_str.replace("%", "")) / 100.0
    rolling_window = st.selectbox("Rolling Lookback Window (Days)", options=[250, 500, 750], index=0)

    st.divider()
    st.subheader("Portfolio scheme")

    scheme = st.radio(
        "Weighting method",
        options=["equal", "user", "market_cap", "long_short"],
        format_func=lambda x: {
            "equal": "Equal-weighted",
            "user": "User-defined weights",
            "market_cap": "Market-cap weighted",
            "long_short": "Long-short (optional)",
        }[x],
        index=0,
    )

    user_weights: dict[str, float] = {}
    long_tickers: list[str] = []
    short_tickers: list[str] = []
    long_weight = 0.5
    short_weight = 0.5

    if scheme == "user":
        st.markdown("**Enter weights** (will be normalised to sum = 1)")
        for t in tickers:
            user_weights[t] = st.number_input(
                f"Weight – {t}",
                min_value=0.0,
                max_value=1.0,
                value=round(1.0 / max(len(tickers), 1), 4),
                step=0.01,
                key=f"w_{t}",
            )

    elif scheme == "long_short":
        st.markdown("**Long leg**")
        long_sel = st.multiselect(
            "Long tickers",
            options=tickers,
            default=tickers[: max(1, len(tickers) // 2)],
        )
        long_tickers = long_sel
        long_weight = st.slider("Total long weight", 0.0, 1.0, 0.5, 0.05)

        st.markdown("**Short leg**")
        short_sel = st.multiselect(
            "Short tickers",
            options=[t for t in tickers if t not in long_tickers],
            default=[t for t in tickers if t not in long_tickers][:1],
        )
        short_tickers = short_sel
        short_weight = st.slider("Total short weight (absolute)", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    run_btn = st.button("🚀 Build Portfolio", type="primary", use_container_width=True)

    with st.expander("Advanced – Cleaning options"):
        apply_winsor = st.checkbox("Winsorise returns", value=False)
        stale_window = st.slider("Stale-price window", 2, 10, 3)

# ── Main ───────────────────────────────────────────────────────────────────
if not run_btn:
    st.info(
        "Configure the inputs in the sidebar and click **Build Portfolio** "
        "to download data, clean it and construct the portfolio."
    )
    st.stop()

if not tickers:
    st.error("Please enter at least one ticker.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

progress = st.progress(0, text="Starting…")
status = st.empty()

try:
    status.info("Resolving target weights…")
    progress.progress(10, text="Weights")

    target_w = resolve_target_weights(
        scheme=scheme,
        tickers=tickers,
        user_weights=user_weights if scheme == "user" else None,
        long_tickers=long_tickers if scheme == "long_short" else None,
        short_tickers=short_tickers if scheme == "long_short" else None,
        long_weight=long_weight,
        short_weight=short_weight,
    )

    status.info(f"Downloading prices for {', '.join(tickers)} …")
    progress.progress(25, text="Downloading")

    prices_raw = download_asset_prices(
        tickers=tickers,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        market_benchmark="^GSPC",
    )
    prices = prices_raw[tickers].copy()

    status.info("Running data-cleaning pipeline…")
    progress.progress(50, text="Cleaning")

    cleaned = run_data_cleaning_pipeline(
        prices=prices,
        weights=target_w,
        align_dates=True,
        missing_method="ffill_bfill",
        stale_window=stale_window,
        apply_winsorize=apply_winsor,
        vol_window=volatility_window,
    )
    clean_prices = cleaned.clean_prices

    status.info("Constructing portfolio with rebalancing…")
    progress.progress(75, text="Construction")

    result: PortfolioResult = construct_portfolio(
        prices=clean_prices,
        target_weights=target_w,
        initial_value=float(initial_value),
        rebalance_freq=rebalance_freq,  # type: ignore
        transaction_cost_bps=float(transaction_cost_bps),
    )
    
    risk_sim = HistoricalSimulation(
        portfolio_values=result.portfolio_value,
        portfolio_pnl=result.portfolio_pnl,
        confidence_level=confidence_level,
    )
    risk_metrics = risk_sim.pnl_calculation()
    rolling_df = risk_sim.rolling_var_series(window=rolling_window)
    
    progress.progress(100, text="Done")
    status.success("Portfolio constructed successfully!")

except Exception as e:
    progress.empty()
    status.empty()
    st.error(f"Pipeline failed: {e}")
    st.exception(e)
    st.stop()

# ── Results ────────────────────────────────────────────────────────────────
meta = result.metadata

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Final Value", f"${meta['final_value']:,.0f}")
k2.metric("Total Return", f"{meta['total_return']:.2%}")
k3.metric("Rebalances", meta["n_rebalances"])
k4.metric("Trading Days", meta["n_days"])
k5.metric(
    "Max Drawdown",
    f"{(result.portfolio_value / result.portfolio_value.cummax() - 1).min():.2%}",
)

st.divider()
st.subheader(f"🛡️ Risk Analytics (Historical VaR at {conf_str} Confidence)")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Historical VaR ($)", f"${risk_metrics['var_dollar']:,.2f}")
r2.metric("Historical VaR (%)", f"{risk_metrics['var_pct']:.2%}")
r3.metric("Expected Shortfall ($)", f"${risk_metrics['es_dollar']:,.2f}")
r4.metric("Expected Shortfall (%)", f"{risk_metrics['es_pct']:.2%}")

st.divider()

tab_val, tab_pnl, tab_risk, tab_ret, tab_w, tab_data = st.tabs(
    ["📊 Portfolio Value", "💰 Daily P&L", "🛡️ Risk & VaR", "📈 Returns", "⚖️ Weights", "📋 Data"]
)

with tab_val:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.portfolio_value.index,
            y=result.portfolio_value,
            mode="lines",
            name="Portfolio Value",
            line=dict(width=2, color="#1f77b4"),
        )
    )
    if len(result.rebalance_dates) > 0:
        rebal_vals = result.portfolio_value.reindex(result.rebalance_dates)
        fig.add_trace(
            go.Scatter(
                x=result.rebalance_dates,
                y=rebal_vals,
                mode="markers",
                name="Rebalance",
                marker=dict(size=8, color="red", symbol="diamond"),
            )
        )
    fig.update_layout(
        title="Daily Portfolio Value",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode="x unified",
        height=480,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_pnl:
    colors = np.where(result.portfolio_pnl >= 0, "#2ca02c", "#d62728")
    fig_pnl = go.Figure(
        go.Bar(
            x=result.portfolio_pnl.index,
            y=result.portfolio_pnl,
            marker_color=colors,
            name="Daily P&L",
        )
    )
    fig_pnl.update_layout(
        title="Daily Portfolio P&L ($)",
        xaxis_title="Date",
        yaxis_title="P&L ($)",
        height=450,
        template="plotly_white",
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

with tab_risk:
    st.subheader(f"Historical P&L Distribution & Tail Risk Cutoffs ({conf_str})")
    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Histogram(
            x=result.portfolio_pnl,
            nbinsx=50,
            name="Daily P&L",
            marker_color="skyblue",
            opacity=0.75,
        )
    )
    fig_dist.add_vline(
        x=-risk_metrics["var_dollar"],
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR ({conf_str}): -${risk_metrics['var_dollar']:,.0f}",
        annotation_position="top left",
    )
    fig_dist.add_vline(
        x=-risk_metrics["es_dollar"],
        line_dash="dot",
        line_color="darkred",
        annotation_text=f"Expected Shortfall: -${risk_metrics['es_dollar']:,.0f}",
        annotation_position="bottom left",
    )
    fig_dist.update_layout(
        title="Portfolio Daily P&L Frequency Distribution",
        xaxis_title="Daily Profit / Loss ($)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader(f"Rolling {rolling_window}-Day Risk Time Series")
    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df["rolling_var"],
        mode="lines", name=f"Rolling VaR ({conf_str})", line=dict(color="red", width=1.5)
    ))
    fig_rolling.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df["rolling_es"],
        mode="lines", name="Rolling Expected Shortfall", line=dict(color="darkred", width=1.5, dash="dash")
    ))
    fig_rolling.update_layout(
        title=f"Rolling VaR vs Expected Shortfall ($)",
        xaxis_title="Date",
        yaxis_title="Risk Measure ($)",
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig_rolling, use_container_width=True)

    st.subheader("Tail Loss Events Exceeding VaR Threshold")
    tail_losses_series = result.portfolio_pnl[result.portfolio_pnl <= -risk_metrics["var_dollar"]]
    fig_tail = go.Figure(
        go.Bar(
            x=tail_losses_series.index,
            y=tail_losses_series,
            marker_color="darkred",
            name="Tail Losses"
        )
    )
    fig_tail.update_layout(
        title=f"Filtered Tail Losses Below -${risk_metrics['var_dollar']:,.0f}",
        xaxis_title="Date",
        yaxis_title="Loss Amount ($)",
        template="plotly_white",
        height=350,
    )
    st.plotly_chart(fig_tail, use_container_width=True)

with tab_ret:
    fig_ret = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Daily Returns", "Cumulative Returns"),
        row_heights=[0.4, 0.6],
    )
    fig_ret.add_trace(
        go.Scatter(
            x=result.portfolio_returns.index,
            y=result.portfolio_returns,
            mode="lines", name="Daily return",
            line=dict(width=1, color="#ff7f0e"),
        ),
        row=1, col=1,
    )
    cum_ret = (1 + result.portfolio_returns).cumprod() - 1
    fig_ret.add_trace(
        go.Scatter(
            x=cum_ret.index, y=cum_ret, mode="lines", name="Cumulative",
            line=dict(width=2, color="#1f77b4"), fill="tozeroy",
        ),
        row=2, col=1,
    )
    fig_ret.update_layout(height=600, template="plotly_white", showlegend=False)
    fig_ret.update_yaxes(tickformat=".1%", row=1, col=1)
    fig_ret.update_yaxes(tickformat=".1%", row=2, col=1)
    st.plotly_chart(fig_ret, use_container_width=True)

    rets = result.portfolio_returns.dropna()
    if len(rets) > 1:
        ann_factor = 252
        stats = {
            "Mean daily": f"{rets.mean():.4%}",
            "Std daily": f"{rets.std():.4%}",
            "Ann. Return": f"{rets.mean() * ann_factor:.2%}",
            "Ann. Vol": f"{rets.std() * np.sqrt(ann_factor):.2%}",
            "Sharpe (rf=0)": f"{(rets.mean() / rets.std()) * np.sqrt(ann_factor):.2f}",
            "Best day": f"{rets.max():.2%}",
            "Worst day": f"{rets.min():.2%}",
            "Skew": f"{rets.skew():.2f}",
            "Kurtosis": f"{rets.kurtosis():.2f}",
        }
        st.subheader("Return statistics")
        st.table(pd.Series(stats, name="Value"))

with tab_w:
    st.subheader("Target weights")
    tw = result.target_weights.sort_values(ascending=False)
    fig_w = px.bar(
        x=tw.index, y=tw.values,
        labels={"x": "Ticker", "y": "Weight"},
        title="Target Allocation",
        color=tw.values, color_continuous_scale="RdYlGn",
    )
    fig_w.update_layout(height=360, template="plotly_white", showlegend=False)
    st.plotly_chart(fig_w, use_container_width=True)

    st.subheader("Weight evolution")
    show_cols = list(result.weights_history.columns)[:8]
    fig_wh = px.line(result.weights_history[show_cols], title="Daily weights over time")
    fig_wh.update_layout(height=400, template="plotly_white", yaxis_tickformat=".0%")
    st.plotly_chart(fig_wh, use_container_width=True)

    with st.expander("Holdings (shares) – last 10 days"):
        st.dataframe(result.holdings.tail(10).style.format("{:.2f}"), use_container_width=True)

with tab_data:
    st.subheader("Daily series")
    df_out = pd.DataFrame({
        "portfolio_value": result.portfolio_value,
        "portfolio_pnl": result.portfolio_pnl,
        "portfolio_returns": result.portfolio_returns,
    })
    st.dataframe(
        df_out.style.format({
            "portfolio_value": "{:,.2f}",
            "portfolio_pnl": "{:,.2f}",
            "portfolio_returns": "{:.4%}",
        }),
        use_container_width=True,
        height=400,
    )
    csv = df_out.to_csv().encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="portfolio_daily.csv", mime="text/csv")

    st.subheader("Cleaning summary")
    ov = data_overview(clean_prices)
    st.text(ov.summary_text())

st.divider()
st.caption(
    f"Scheme: **{scheme}** · Rebalance: **{rebalance_freq}** · "
    f"Cost: **{transaction_cost_bps} bps** · Period: {start_date} → {end_date}"
)