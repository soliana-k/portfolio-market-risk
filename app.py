"""
Streamlit App - Portfolio Construction & Risk Analytics
Professional-grade UI inspired by Bloomberg Terminal / Yahoo Finance
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

from src.data_collection.download_portfolio import (
    download_asset_prices_cached,
    download_market_caps_cached,
)
from src.data_cleaning.cleaning import run_data_cleaning_pipeline, data_overview
from src.portfolio_construction.portfolio import (
    construct_portfolio,
    resolve_target_weights,
    PortfolioResult,
)
from src.historical_simulation_var.historical_var import HistoricalSimulation
from src.garch_var.garch import fit_garch
from src.gjr_garch.gjr import fit_gjr_garch
from src.ewma_var.ewma import fit_ewma
from src.var_backtesting.backtest import run_all_backtests, backtest_model
from src.model_comparison.compare import assemble_comparison, MODEL_ORDER
from src.stress_testing.stress import run_stress_tests
from src.analytics.portfolio_metrics import (
    asset_correlation_matrix,
    asset_return_attribution,
    concentration_metrics,
    multi_confidence_var_es,
    portfolio_variance,
    rolling_exception_series,
)
from src.analytics.report import build_html_report

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Risk Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("# Portfolio Construction & Risk Analytics")
st.caption(
    "Multi-asset portfolio construction with configurable rebalancing, "
    "Historical VaR, Expected Shortfall, GARCH volatility modelling, and stress testing."
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Portfolio Configuration")

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

    st.markdown("---")
    st.markdown("### Risk Parameters")
    conf_str = st.selectbox("Confidence Level", ['95%', '99%', '99.5%'], index=0)
    confidence_level = float(conf_str.replace("%", "")) / 100.0
    rolling_window = st.selectbox("Rolling Lookback Window (Days)", options=[250, 500, 750], index=0)

    st.markdown("---")
    st.markdown("### Portfolio Scheme")

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
        st.markdown("**Enter weights** (normalised to sum = 1)")
        for t in tickers:
            user_weights[t] = st.number_input(
                f"Weight - {t}",
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

    st.markdown("---")
    run_btn = st.button("Build Portfolio", type="primary", width="stretch")
    force_refresh = st.checkbox(
        "Force fresh data download",
        value=False,
        help="Bypass the cache and re-download prices & market caps from Yahoo Finance.",
    )

    with st.expander("Cleaning options"):
        apply_winsor = st.checkbox("Winsorise returns", value=False)
        stale_window = st.slider("Stale-price window", 2, 10, 3)

# ── Main ────────────────────────────────────────────────────────────────────
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

progress = st.progress(0, text="Starting...")
status = st.empty()

stale_flags: list = []

try:
    status.info("Resolving target weights...")
    progress.progress(10, text="Weights")

    market_caps_series = None
    if scheme == "market_cap":
        market_caps_series = download_market_caps_cached(
            tickers,
            force_refresh=force_refresh,
            on_stale=lambda m: stale_flags.append(m.get("saved_at")),
        )

    target_w = resolve_target_weights(
        scheme=scheme,
        tickers=tickers,
        user_weights=user_weights if scheme == "user" else None,
        long_tickers=long_tickers if scheme == "long_short" else None,
        short_tickers=short_tickers if scheme == "long_short" else None,
        long_weight=long_weight,
        short_weight=short_weight,
        market_caps=market_caps_series,
    )

    status.info(f"Downloading prices for {', '.join(tickers)}...")
    progress.progress(25, text="Downloading")

    prices_raw = download_asset_prices_cached(
        tickers=tickers,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        market_benchmark="^GSPC",
        force_refresh=force_refresh,
        on_stale=lambda m: stale_flags.append(m.get("saved_at")),
    )
    prices = prices_raw[tickers].copy()

    status.info("Running data-cleaning pipeline...")
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

    status.info("Constructing portfolio with rebalancing...")
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

    status.info("Fitting GARCH(1,1) volatility model...")
    progress.progress(90, text="GARCH")

    garch_res = fit_garch(
        result.portfolio_returns,
        confidence_level=confidence_level,
        portfolio_value=float(initial_value),
    )

    status.info("Fitting GJR-GARCH volatility model (asymmetric)...")
    progress.progress(95, text="GJR-GARCH")

    gjr_res = fit_gjr_garch(
        result.portfolio_returns,
        confidence_level=confidence_level,
        portfolio_value=float(initial_value),
    )

    status.info("Running VaR backtests (rolling out-of-sample)...")
    progress.progress(97, text="Backtest")

    bt_window = int(max(250, min(750, len(result.portfolio_returns) - 250)))
    bt_results = run_all_backtests(
        result.portfolio_returns,
        confidence_level=confidence_level,
        estimation_window=bt_window,
        portfolio_value=float(initial_value),
    )

    status.info("Comparing all VaR models (incl. EWMA)...")
    progress.progress(98, text="Compare")

    ewma_bt = backtest_model(
        result.portfolio_returns,
        lambda tr, cl, pv: fit_ewma(tr, confidence_level=cl, portfolio_value=pv).var_pct,
        model_name="EWMA",
        confidence_level=confidence_level,
        estimation_window=bt_window,
        portfolio_value=float(initial_value),
    )
    cmp_results = assemble_comparison(
        {**bt_results, "EWMA": ewma_bt}, confidence_level
    )

    status.info("Running stress tests (historical + hypothetical scenarios)...")
    progress.progress(99, text="Stress")

    asset_returns = clean_prices.pct_change().dropna()
    ewma_full = fit_ewma(
        result.portfolio_returns,
        confidence_level=confidence_level,
        portfolio_value=float(initial_value),
    )
    stress = run_stress_tests(
        portfolio_value=float(initial_value),
        weights=result.target_weights,
        portfolio_returns=result.portfolio_returns,
        asset_returns=asset_returns,
        baseline_var_pct=gjr_res.var_pct,
        baseline_es_pct=gjr_res.es_pct,
        confidence_level=confidence_level,
    )

    # ── Additional analytics (correlation, diversification, attribution) ──
    asset_rets = clean_prices.pct_change().dropna()
    corr_matrix = asset_correlation_matrix(asset_rets)
    div_metrics = concentration_metrics(result.target_weights)
    multi_var_es = multi_confidence_var_es(
        result.portfolio_returns,
        result.portfolio_value,
        confidences=[0.95, 0.975, 0.99, 0.995],
    )
    attribution = asset_return_attribution(asset_rets, result.target_weights)
    portfolio_var = portfolio_variance(result.target_weights, asset_rets.cov())

    # Benchmark (S&P 500) series aligned to the portfolio, if available.
    benchmark_col = None
    for cand in ("^GSPC", "GSPC"):
        if cand in prices_raw.columns:
            benchmark_col = cand
            break
    benchmark_rets = None
    if benchmark_col is not None:
        bench_prices = (
            prices_raw[benchmark_col]
            .reindex(result.portfolio_value.index)
            .ffill()
        )
        benchmark_rets = bench_prices.pct_change().dropna()

    from datetime import datetime as _dt

    refresh_ts = _dt.now().strftime("%Y-%m-%d %H:%M")

    progress.progress(100, text="Done")
    status.success("Portfolio constructed successfully.")

    # Persist the run so switching tabs doesn't recompute the pipeline.
    st.session_state["results"] = {
        "meta": result.metadata,
        "result": result,
        "risk_metrics": risk_metrics,
        "rolling_df": rolling_df,
        "garch_res": garch_res,
        "gjr_res": gjr_res,
        "ewma_full": ewma_full,
        "bt_results": bt_results,
        "cmp_results": cmp_results,
        "stress": stress,
        "corr_matrix": corr_matrix,
        "div_metrics": div_metrics,
        "multi_var_es": multi_var_es,
        "attribution": attribution,
        "portfolio_var": portfolio_var,
        "benchmark_rets": benchmark_rets,
        "clean_prices": clean_prices,
        "asset_rets": asset_rets,
        "refresh_ts": refresh_ts,
        "params": {
            "scheme": scheme,
            "rebalance_freq": rebalance_freq,
            "transaction_cost_bps": transaction_cost_bps,
            "start_date": start_date,
            "end_date": end_date,
            "conf_str": conf_str,
        },
    }

except Exception as e:
    progress.empty()
    status.empty()
    st.error(f"Pipeline failed: {e}")
    st.exception(e)
    st.stop()

# Clear the transient progress bar and status message once results render,
# so the dashboard front stays clean.
progress.empty()
status.empty()

if stale_flags:
    from datetime import datetime as _dt

    for saved_at in stale_flags:
        if saved_at:
            ts = _dt.fromtimestamp(float(saved_at)).strftime("%Y-%m-%d %H:%M")
        else:
            ts = "an unknown time"
        st.warning(
            f"Live data is currently unavailable - showing the last cached "
            f"data from {ts}. Re-run later to refresh."
        )

# ── Chart Theme ─────────────────────────────────────────────────────────────
CHART_BG = "#ffffff"
CHART_PAPER = "#ffffff"
CHART_GRID = "#f0f0f0"
CHART_FONT = "#333333"
CHART_FONT_SIZE = 11

COLOR_PRIMARY = "#0d6efd"
COLOR_NEGATIVE = "#dc3545"
COLOR_POSITIVE = "#198754"
COLOR_SECONDARY = "#6c757d"
COLOR_HIST = "#adb5bd"
COLOR_GARCH = "#6f42c1"
COLOR_GJR = "#20c997"
COLOR_EWMA = "#fd7e14"
COLOR_ACCENT = "#0dcaf0"

MODEL_COLORS = {
    "Historical": COLOR_PRIMARY,
    "EWMA": COLOR_EWMA,
    "GARCH": COLOR_GARCH,
    "GJR-GARCH": COLOR_GJR,
}


def _apply_chart_layout(fig, title="", height=420, yformat="", x_title="", y_title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=CHART_FONT, family="Inter"), x=0.01),
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_BG,
        font=dict(size=CHART_FONT_SIZE, color=CHART_FONT, family="Inter"),
        xaxis=dict(
            gridcolor=CHART_GRID, gridwidth=0.5,
            linecolor="#dee2e6", linewidth=0.5,
            title=dict(text=x_title, font=dict(size=11)),
        ),
        yaxis=dict(
            gridcolor=CHART_GRID, gridwidth=0.5,
            linecolor="#dee2e6", linewidth=0.5,
            title=dict(text=y_title, font=dict(size=11)),
            tickformat=yformat,
        ),
        margin=dict(l=50, r=20, t=40, b=40),
        height=height,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=11, font_family="Inter"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=10), bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    return fig


# ── Results: Dashboard KPI Header ──────────────────────────────────────────
meta = result.metadata

# Header bar: last refresh timestamp + US market open/close status.
try:
    from datetime import datetime as _nowmod
    import pytz as _pytz  # type: ignore

    now_utc = _nowmod.now(_pytz.timezone("UTC"))
    ny = _nowmod.now(_pytz.timezone("America/New_York"))
    _mark_open = 9.5 <= ny.hour + ny.minute / 60 < 16.0 and ny.weekday() < 5
    _status = "Market Open" if _mark_open else "Market Closed"
    _status_color = COLOR_POSITIVE if _mark_open else COLOR_NEGATIVE
    hb1, hb2, hb3 = st.columns([2, 2, 3])
    hb1.metric("Last Data Refresh", refresh_ts)
    hb2.metric("US Market", _status)
    hb3.write("")
    st.caption(
        f"Time in New York: {ny.strftime('%Y-%m-%d %H:%M')} — "
        "prices reflect the most recently cached session."
    )
except Exception:
    hb1, hb2, hb3 = st.columns([2, 2, 3])
    hb1.metric("Last Data Refresh", refresh_ts)
    hb2.metric("", "")
    hb3.write("")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Final Value", f"${meta['final_value']:,.0f}")
k2.metric("Total Return", f"{meta['total_return']:.2%}")
k3.metric("Rebalances", meta["n_rebalances"])
k4.metric("Trading Days", meta["n_days"])
k5.metric(
    "Max Drawdown",
    f"{(result.portfolio_value / result.portfolio_value.cummax() - 1).min():.2%}",
)

# ── Risk & Volatility Model Summary (collapsible) ──────────────────────────
# Kept behind an accordion so the dashboard front stays clean and professional.
with st.expander(
    "VaR & Volatility Model Summary "
    f"(Historical / GARCH / GJR-GARCH - {conf_str} Confidence)"
):
    summary_rows = [
        {
            "Metric": "VaR (%)",
            "Historical": f"{risk_metrics['var_pct']:.2%}",
            "GARCH": f"{garch_res.var_pct:.2%}",
            "GJR-GARCH": f"{gjr_res.var_pct:.2%}",
        },
        {
            "Metric": "VaR ($)",
            "Historical": f"${risk_metrics['var_dollar']:,.2f}",
            "GARCH": f"${garch_res.var_dollar:,.2f}",
            "GJR-GARCH": f"${gjr_res.var_dollar:,.2f}",
        },
        {
            "Metric": "Expected Shortfall (%)",
            "Historical": f"{risk_metrics['es_pct']:.2%}",
            "GARCH": f"{garch_res.es_pct:.2%}",
            "GJR-GARCH": f"{gjr_res.es_pct:.2%}",
        },
        {
            "Metric": "Expected Shortfall ($)",
            "Historical": f"${risk_metrics['es_dollar']:,.2f}",
            "GARCH": f"${garch_res.es_dollar:,.2f}",
            "GJR-GARCH": f"${gjr_res.es_dollar:,.2f}",
        },
        {
            "Metric": "1-Day sigma Forecast",
            "Historical": "-",
            "GARCH": f"{garch_res.sigma_forecast:.2%}",
            "GJR-GARCH": f"{gjr_res.sigma_forecast:.2%}",
        },
    ]
    st.dataframe(
        pd.DataFrame(summary_rows).set_index("Metric"),
        use_container_width=True,
        hide_index=False,
    )

    st.markdown("**Estimated model parameters**")

    def _fmt(v):
        return "-" if v is None else f"{float(v):.6f}"

    param_rows = [
        {
            "Parameter": "omega",
            "GARCH(1,1)": _fmt(garch_res.params.get("omega")),
            "GJR-GARCH": _fmt(gjr_res.omega),
        },
        {
            "Parameter": "alpha[1]",
            "GARCH(1,1)": _fmt(garch_res.params.get("alpha[1]")),
            "GJR-GARCH": _fmt(gjr_res.alpha),
        },
        {
            "Parameter": "beta[1]",
            "GARCH(1,1)": _fmt(garch_res.params.get("beta[1]")),
            "GJR-GARCH": _fmt(gjr_res.beta),
        },
        {
            "Parameter": "gamma[1] (leverage)",
            "GARCH(1,1)": "-",
            "GJR-GARCH": _fmt(gjr_res.leverage_parameter),
        },
    ]
    st.dataframe(
        pd.DataFrame(param_rows).set_index("Parameter"),
        use_container_width=True,
        hide_index=False,
    )

st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────────────
(
    tab_val,
    tab_pnl,
    tab_risk,
    tab_bt,
    tab_cmp,
    tab_stress,
    tab_ret,
    tab_ana,
    tab_w,
    tab_data,
) = st.tabs(
    [
        "Portfolio Value",
        "Daily P&L",
        "Risk & VaR",
        "Backtest",
        "Model Comparison",
        "Stress Testing",
        "Returns",
        "Analytics",
        "Weights",
        "Data",
    ]
)

# ── Tab: Portfolio Value ────────────────────────────────────────────────────
with tab_val:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.portfolio_value.index,
            y=result.portfolio_value,
            mode="lines",
            name="Portfolio Value",
            line=dict(width=1.8, color=COLOR_PRIMARY),
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
                marker=dict(size=6, color=COLOR_NEGATIVE, symbol="diamond-open", line=dict(width=1.5)),
            )
        )
    _apply_chart_layout(fig, title="Daily Portfolio Value", y_title="Value ($)", yformat="$.0f")
    fig.update_layout(height=440)
    st.plotly_chart(fig, width="stretch")

    if benchmark_rets is not None:
        st.markdown("#### Portfolio vs S&P 500 (^GSPC)")
        port_norm = result.portfolio_value / result.portfolio_value.iloc[0]
        bench_norm = (1 + benchmark_rets).cumprod()
        cmp_df = pd.DataFrame(
            {
                "Portfolio": port_norm.reindex(bench_norm.index).ffill(),
                "S&P 500": bench_norm,
            }
        ).dropna()
        fig_cmp = go.Figure()
        fig_cmp.add_trace(
            go.Scatter(
                x=cmp_df.index, y=cmp_df["Portfolio"], mode="lines",
                name="Portfolio", line=dict(width=1.8, color=COLOR_PRIMARY),
            )
        )
        fig_cmp.add_trace(
            go.Scatter(
                x=cmp_df.index, y=cmp_df["S&P 500"], mode="lines",
                name="S&P 500", line=dict(width=1.6, color=COLOR_SECONDARY, dash="dash"),
            )
        )
        _apply_chart_layout(
            fig_cmp,
            title="Growth of $1 (Portfolio vs Benchmark)",
            y_title="Growth",
        )
        fig_cmp.update_layout(height=400)
        st.plotly_chart(fig_cmp, width="stretch")

        port_ret = port_norm.iloc[-1] - 1
        bench_ret = bench_norm.iloc[-1] - 1
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio Return", f"{port_ret:.2%}")
        c2.metric("S&P 500 Return", f"{bench_ret:.2%}")
        c3.metric("Excess Return", f"{port_ret - bench_ret:+.2%}")

# ── Tab: Daily P&L ─────────────────────────────────────────────────────────
with tab_pnl:
    colors = [COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE for v in result.portfolio_pnl]
    fig_pnl = go.Figure(
        go.Bar(
            x=result.portfolio_pnl.index,
            y=result.portfolio_pnl,
            marker_color=colors,
            name="Daily P&L",
            marker_line_width=0,
        )
    )
    _apply_chart_layout(fig_pnl, title="Daily Portfolio P&L ($)", y_title="P&L ($)", yformat="$,.0f")
    fig_pnl.update_layout(height=420)
    st.plotly_chart(fig_pnl, width="stretch")

# ── Tab: Risk & VaR ────────────────────────────────────────────────────────
with tab_risk:
    st.markdown(f"#### P&L Distribution & Tail Risk Cutoffs ({conf_str})")
    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Histogram(
            x=result.portfolio_pnl,
            nbinsx=50,
            name="Daily P&L",
            marker_color=COLOR_HIST,
            marker_line=dict(width=0.5, color="#dee2e6"),
            opacity=0.8,
        )
    )
    fig_dist.add_vline(
        x=-risk_metrics["var_dollar"],
        line_dash="dash",
        line_color=COLOR_NEGATIVE,
        line_width=1.5,
        annotation_text=f"VaR ({conf_str}): -${risk_metrics['var_dollar']:,.0f}",
        annotation_position="top left",
        annotation_font=dict(size=10),
    )
    fig_dist.add_vline(
        x=-risk_metrics["es_dollar"],
        line_dash="dot",
        line_color="#8b0000",
        line_width=1.5,
        annotation_text=f"ES: -${risk_metrics['es_dollar']:,.0f}",
        annotation_position="bottom left",
        annotation_font=dict(size=10),
    )
    _apply_chart_layout(fig_dist, title="Portfolio Daily P&L Frequency Distribution",
                        y_title="Frequency", x_title="Daily Profit / Loss ($)")
    st.plotly_chart(fig_dist, width="stretch")

    st.markdown(f"#### Rolling {rolling_window}-Day Risk Time Series")
    fig_rolling = go.Figure()
    var_col = "rolling_var" if "rolling_var" in rolling_df.columns else rolling_df.columns[0]
    es_col = "rolling_es" if "rolling_es" in rolling_df.columns else rolling_df.columns[1]

    fig_rolling.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df[var_col],
        mode="lines", name=f"Rolling VaR ({conf_str})",
        line=dict(color=COLOR_NEGATIVE, width=1.2),
    ))
    fig_rolling.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df[es_col],
        mode="lines", name="Rolling Expected Shortfall",
        line=dict(color="#8b0000", width=1.2, dash="dash"),
    ))
    _apply_chart_layout(fig_rolling, title="Rolling VaR vs Expected Shortfall ($)",
                        y_title="Risk Measure ($)")
    st.plotly_chart(fig_rolling, width="stretch")

    st.markdown("#### GARCH(1,1) Conditional Volatility")
    fig_garch = go.Figure()
    fig_garch.add_trace(go.Scatter(
        x=garch_res.conditional_volatility.index,
        y=garch_res.conditional_volatility,
        mode="lines", name="Conditional sigma (daily)",
        line=dict(color=COLOR_GARCH, width=1),
    ))
    fig_garch.add_hline(
        y=garch_res.sigma_forecast,
        line_dash="dash", line_color=COLOR_NEGATIVE, line_width=1,
        annotation_text=f"1-Day Forecast: {garch_res.sigma_forecast:.2%}",
        annotation_position="top right",
        annotation_font=dict(size=10),
    )
    _apply_chart_layout(fig_garch, title="Conditional Volatility (GARCH(1,1))",
                        y_title="Volatility", yformat=".1%")
    st.plotly_chart(fig_garch, width="stretch")

    # GJR-GARCH
    st.markdown("#### GJR-GARCH Conditional Volatility")
    fig_gjr = go.Figure()
    fig_gjr.add_trace(go.Scatter(
        x=gjr_res.conditional_volatility.index,
        y=gjr_res.conditional_volatility,
        mode="lines", name="GJR sigma (daily)",
        line=dict(color=COLOR_GJR, width=1),
    ))
    fig_gjr.add_hline(
        y=gjr_res.sigma_forecast,
        line_dash="dash", line_color=COLOR_NEGATIVE, line_width=1,
        annotation_text=f"1-Day Forecast: {gjr_res.sigma_forecast:.2%}",
        annotation_position="top right",
        annotation_font=dict(size=10),
    )
    _apply_chart_layout(fig_gjr, title="Conditional Volatility (GJR-GARCH)",
                        y_title="Volatility", yformat=".1%")
    st.plotly_chart(fig_gjr, width="stretch")

    # Vol comparison
    st.markdown("#### GARCH vs GJR-GARCH Volatility Comparison")
    cmp = pd.DataFrame({
        "GJR-GARCH": gjr_res.conditional_volatility,
        "GARCH(1,1)": gjr_res.garch_conditional_volatility,
    }).dropna()
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=cmp.index, y=cmp["GJR-GARCH"],
        mode="lines", name="GJR-GARCH sigma",
        line=dict(color=COLOR_GJR, width=1),
    ))
    fig_cmp.add_trace(go.Scatter(
        x=cmp.index, y=cmp["GARCH(1,1)"],
        mode="lines", name="GARCH(1,1) sigma",
        line=dict(color=COLOR_GARCH, width=1, dash="dot"),
    ))
    _apply_chart_layout(fig_cmp, title="Conditional Volatility: GARCH vs GJR-GARCH",
                        y_title="Volatility", yformat=".1%")
    st.plotly_chart(fig_cmp, width="stretch")

    # Leverage
    st.markdown("#### Leverage Effect (Asymmetry)")
    ls = gjr_res.leverage_stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg sigma after negative day", f"{ls['avg_vol_after_negative']:.2%}")
    c2.metric("Avg sigma after positive day", f"{ls['avg_vol_after_positive']:.2%}")
    c3.metric("Neg / Pos sigma ratio", f"{ls['vol_ratio_neg_over_pos']:.3f}")

    lev_df = pd.DataFrame({
        "return": gjr_res.returns,
        "cond_vol_next": gjr_res.conditional_volatility.shift(-1),
        "sign": np.where(gjr_res.returns < 0, "Negative return", "Positive return"),
    }).dropna()
    fig_lev = px.scatter(
        lev_df, x="return", y="cond_vol_next", color="sign",
        color_discrete_map={"Negative return": COLOR_NEGATIVE, "Positive return": COLOR_POSITIVE},
        opacity=0.5,
        labels={"return": "Daily return", "cond_vol_next": "Next-day conditional sigma"},
    )
    _apply_chart_layout(fig_lev, title="Leverage Effect: Negative Returns -> Higher Next-Day Volatility",
                        yformat=".1%", y_title="Next-day conditional sigma")
    st.plotly_chart(fig_lev, width="stretch")

    # VaR comparison bar
    st.markdown(f"#### VaR Comparison ({conf_str})")
    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(
        x=["Historical VaR", "GARCH-Scaled VaR", "GJR-GARCH-Scaled VaR"],
        y=[risk_metrics["var_pct"], garch_res.var_pct, gjr_res.var_pct],
        marker_color=[COLOR_PRIMARY, COLOR_GARCH, COLOR_GJR],
        text=[
            f"{risk_metrics['var_pct']:.2%}",
            f"{garch_res.var_pct:.2%}",
            f"{gjr_res.var_pct:.2%}",
        ],
        textposition="auto",
        name="VaR (%)",
        marker_line_width=0,
    ))
    _apply_chart_layout(fig_var, title="Value-at-Risk Comparison (% of portfolio)",
                        y_title="VaR (%)", yformat=".1%")
    st.plotly_chart(fig_var, width="stretch")

    # Tail losses
    st.markdown("#### Tail Loss Events Exceeding VaR Threshold")
    tail_losses_series = result.portfolio_pnl[result.portfolio_pnl <= -risk_metrics["var_dollar"]]
    fig_tail = go.Figure(
        go.Bar(
            x=tail_losses_series.index,
            y=tail_losses_series,
            marker_color="#8b0000",
            name="Tail Losses",
            marker_line_width=0,
        )
    )
    _apply_chart_layout(fig_tail,
                        title=f"Filtered Tail Losses Below -${risk_metrics['var_dollar']:,.0f}",
                        y_title="Loss Amount ($)", yformat="$,.0f")
    fig_tail.update_layout(height=350)
    st.plotly_chart(fig_tail, width="stretch")

# ── Tab: Backtest ───────────────────────────────────────────────────────────
with tab_bt:
    st.markdown(f"#### VaR Backtesting - Rolling Out-of-Sample (est. window = {bt_window})")
    st.caption(
        "Each day in the backtest period, the VaR is re-estimated using only the "
        "preceding estimation window. An exception occurs when the realised loss "
        "exceeds the VaR. Models are assessed with the Kupiec POF, Christoffersen "
        "independence & conditional-coverage tests, and the Basel Traffic Light."
    )

    bt_rows = []
    for name, btr in bt_results.items():
        bt_rows.append({
            "Model": name,
            "Obs": btr.n_observations,
            "Exceptions": btr.n_exceptions,
            "Expected": round(btr.expected_exceptions, 1),
            "Ratio": f"{btr.exception_ratio:.2%}",
            "Kupiec p": f"{btr.kupiec_pof_pvalue:.3g}",
            "Indep p": f"{btr.christoffersen_independence_pvalue:.3g}",
            "CC p": f"{btr.christoffersen_cc_pvalue:.3g}",
            "Basel": btr.basel_zone,
            "Result": btr.pass_fail,
        })
    bt_df = pd.DataFrame(bt_rows).set_index("Model")
    st.dataframe(bt_df, width="stretch")

    bt_model = st.selectbox(
        "Model to visualise", options=list(bt_results.keys()), index=0,
    )
    btr = bt_results[bt_model]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exceptions", f"{btr.n_exceptions} / {btr.n_observations}")
    c2.metric("Expected Exceptions", f"{btr.expected_exceptions:.1f}")
    c3.metric("Exception Ratio", f"{btr.exception_ratio:.2%}")
    c4.metric("Basel Zone", btr.basel_zone, delta=btr.pass_fail)

    st.markdown(f"#### {bt_model}: VaR Forecast vs Realised Returns")
    plot_df = pd.DataFrame({
        "actual": btr.actual_returns,
        "VaR": btr.var_series,
    })
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["actual"], mode="lines",
        name="Actual return", line=dict(color=COLOR_PRIMARY, width=0.8),
    ))
    fig_bt.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["VaR"], mode="lines",
        name="VaR (1-day)", line=dict(color=COLOR_EWMA, width=1, dash="dot"),
    ))
    exc = plot_df[btr.exceptions.astype(bool)]
    fig_bt.add_trace(go.Scatter(
        x=exc.index, y=exc["actual"], mode="markers",
        name="Exception",
        marker=dict(color=COLOR_NEGATIVE, size=5, symbol="x"),
    ))
    _apply_chart_layout(fig_bt, title=f"{bt_model} - Realised Returns vs VaR",
                        y_title="Return", yformat=".1%")
    st.plotly_chart(fig_bt, width="stretch")

    st.markdown("#### Exceptions vs Expected")
    zone_color = {"Green": COLOR_POSITIVE, "Yellow": COLOR_EWMA, "Red": COLOR_NEGATIVE}
    fig_exc = go.Figure()
    fig_exc.add_trace(go.Bar(
        x=["Expected", "Observed"],
        y=[btr.expected_exceptions, btr.n_exceptions],
        marker_color=[COLOR_PRIMARY, zone_color[btr.basel_zone]],
        text=[f"{btr.expected_exceptions:.1f}", f"{btr.n_exceptions}"],
        textposition="auto",
        marker_line_width=0,
    ))
    _apply_chart_layout(fig_exc,
                        title=f"{bt_model} - Exceptions (Basel zone: {btr.basel_zone})",
                        y_title="Count")
    fig_exc.update_layout(height=360)
    st.plotly_chart(fig_exc, width="stretch")

    # ── Rolling exception rate + backtest p-value summary ────────────────────
    win = st.slider("Rolling exception window (days)", 20, 120, 60)
    roll_rate = rolling_exception_series(btr.exceptions, window=win)
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=roll_rate.index, y=roll_rate, mode="lines",
        name="Rolling exception rate", line=dict(color=COLOR_GARCH, width=1.5),
    ))
    fig_roll.add_hline(y=1 - confidence_level, line_dash="dash",
                       line_color=COLOR_SECONDARY,
                       annotation_text=f"Target (1 - {conf_str})",
                       annotation_font=dict(size=10))
    _apply_chart_layout(
        fig_roll,
        title=f"{bt_model} - Rolling {win}-day Exception Rate",
        y_title="Exception rate", yformat=".1%",
    )
    fig_roll.update_layout(height=360)
    st.plotly_chart(fig_roll, width="stretch")

    st.markdown("#### Backtest Hypothesis Test p-values")
    pv_rows = [
        {"Test": "Kupiec POF (unconditional coverage)",
         "p-value": f"{btr.kupiec_pof_pvalue:.4g}"},
        {"Test": "Christoffersen independence",
         "p-value": f"{btr.christoffersen_independence_pvalue:.4g}"},
        {"Test": "Christoffersen CC (joint)",
         "p-value": f"{btr.christoffersen_cc_pvalue:.4g}"},
        {"Test": "Pass / Fail", "p-value": btr.pass_fail},
    ]
    st.dataframe(
        pd.DataFrame(pv_rows).set_index("Test"),
        use_container_width=True,
        hide_index=False,
    )

# ── Tab: Model Comparison ──────────────────────────────────────────────────
with tab_cmp:
    st.markdown("#### VaR Model Comparison")
    st.caption(
        "Four VaR methodologies - Historical Simulation, EWMA (RiskMetrics), "
        "GARCH-scaled and GJR-GARCH-scaled Historical VaR - evaluated on the same "
        "rolling out-of-sample backtest."
    )

    sel_model = st.selectbox(
        "Focus model", options=list(cmp_results.keys()), index=1,
    )

    m = cmp_results[sel_model]
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Avg VaR", f"{m.average_var:.2%}")
    mc2.metric("VaR Volatility", f"{m.var_volatility:.2%}")
    mc3.metric("Worst Daily Loss", f"{m.worst_daily_loss:.2%}")
    mc4.metric("Expected Shortfall", f"{m.expected_shortfall:.2%}")
    mc5.metric("Exceptions", f"{m.backtest.n_exceptions} / {m.backtest.n_observations}")

    st.markdown("#### VaR Model Comparison (over backtest period)")
    fig_cmp = go.Figure()
    for name, cr in cmp_results.items():
        fig_cmp.add_trace(go.Scatter(
            x=cr.backtest.var_series.index,
            y=cr.backtest.var_series,
            mode="lines", name=name,
            line=dict(color=MODEL_COLORS.get(name, COLOR_SECONDARY), width=1),
        ))
    _apply_chart_layout(fig_cmp, title="1-Day VaR Forecasts by Model (negative = loss threshold)",
                        y_title="VaR", yformat=".1%")
    st.plotly_chart(fig_cmp, width="stretch")

    st.markdown(f"#### Actual P&L vs VaR - {sel_model}")
    pv = float(initial_value)
    plot_pnl = pd.DataFrame({
        "actual_pnl": m.backtest.actual_returns * pv,
        "var_pnl": m.backtest.var_series * pv,
    })
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=plot_pnl.index, y=plot_pnl["actual_pnl"], mode="lines",
        name="Actual P&L ($)", line=dict(color=COLOR_PRIMARY, width=0.8),
    ))
    fig_pnl.add_trace(go.Scatter(
        x=plot_pnl.index, y=plot_pnl["var_pnl"], mode="lines",
        name="VaR threshold ($)", line=dict(color=COLOR_NEGATIVE, width=1, dash="dot"),
    ))
    fig_pnl.add_hline(y=0, line_color="#dee2e6", line_width=0.8)
    _apply_chart_layout(fig_pnl, title=f"{sel_model} - Daily P&L vs VaR Threshold",
                        y_title="P&L ($)", yformat="$,.0f")
    st.plotly_chart(fig_pnl, width="stretch")

    st.markdown(f"#### Exceptions Overlay - {sel_model}")
    fig_ov = go.Figure()
    fig_ov.add_trace(go.Scatter(
        x=m.backtest.actual_returns.index, y=m.backtest.actual_returns,
        mode="lines", name="Actual return", line=dict(color=COLOR_PRIMARY, width=0.8),
    ))
    fig_ov.add_trace(go.Scatter(
        x=m.backtest.var_series.index, y=m.backtest.var_series,
        mode="lines", name="VaR", line=dict(color=COLOR_EWMA, width=1, dash="dot"),
    ))
    exc = m.backtest.actual_returns[m.backtest.exceptions.astype(bool)]
    fig_ov.add_trace(go.Scatter(
        x=exc.index, y=exc, mode="markers",
        name="Exception", marker=dict(color=COLOR_NEGATIVE, size=5, symbol="x"),
    ))
    _apply_chart_layout(fig_ov, title=f"{sel_model} - Exceptions (return < VaR)",
                        y_title="Return", yformat=".1%")
    st.plotly_chart(fig_ov, width="stretch")

    st.markdown("#### Backtesting Summary - Expected vs Observed Exceptions")
    zone_color = {"Green": COLOR_POSITIVE, "Yellow": COLOR_EWMA, "Red": COLOR_NEGATIVE}
    names = list(cmp_results.keys())
    expected = [cmp_results[n].backtest.expected_exceptions for n in names]
    observed = [cmp_results[n].backtest.n_exceptions for n in names]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=names, y=expected, name="Expected", marker_color=COLOR_PRIMARY,
        marker_line_width=0,
    ))
    fig_bar.add_trace(go.Bar(
        x=names, y=observed, name="Observed",
        marker_color=[zone_color[cmp_results[n].backtest.basel_zone] for n in names],
        marker_line_width=0,
    ))
    fig_bar.update_layout(barmode="group", yaxis_title="Exceptions")
    _apply_chart_layout(fig_bar, y_title="Exceptions")
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown("#### Model Ranking")
    rank_rows = []
    for name in MODEL_ORDER:
        if name not in cmp_results:
            continue
        cr = cmp_results[name]
        rank_rows.append({
            "Rank": cr.rank,
            "Model": name,
            "Zone": cr.backtest.basel_zone,
            "Exceptions": cr.backtest.n_exceptions,
            "Ratio": f"{cr.backtest.exception_ratio:.2%}",
            "Avg VaR": f"{cr.average_var:.2%}",
            "VaR Vol": f"{cr.var_volatility:.2%}",
            "Worst Loss": f"{cr.worst_daily_loss:.2%}",
            "ES": f"{cr.expected_shortfall:.2%}",
            "Kupiec p": f"{cr.backtest.kupiec_pof_pvalue:.3g}",
            "Result": cr.backtest.pass_fail,
        })
    rank_df = pd.DataFrame(rank_rows).sort_values("Rank").reset_index(drop=True)
    st.dataframe(rank_df, width="stretch")

# ── Tab: Stress Testing ────────────────────────────────────────────────────
with tab_stress:
    st.markdown("#### Stress Testing")
    st.caption(
        "Historical crises (actual portfolio loss when the window is in-sample, "
        "otherwise a representative equity drawdown) and hypothetical shocks "
        "(equity drops, volatility / correlation spikes, sector shock)."
    )

    zone_color = {"Green": COLOR_POSITIVE, "Yellow": COLOR_EWMA, "Red": COLOR_NEGATIVE}
    alpha = 1.0 - confidence_level

    s1, s2, s3 = st.columns(3)
    s1.metric("Worst-case scenario", stress.worst_case_name)
    s2.metric("Worst-case loss", f"${stress.scenarios[stress.worst_case_name].stressed_loss_value:,.0f}")
    s3.metric("Worst-case VaR", f"{stress.scenarios[stress.worst_case_name].stressed_var_pct:.2%}")

    st.markdown("#### Stress Scenario Results")
    srows = []
    for name in stress.ranking:
        sc = stress.scenarios[name]
        srows.append({
            "Scenario": name,
            "Type": sc.scenario_type,
            "Shock Return": f"{sc.shock_return_pct:.2%}",
            "Stressed VaR": f"{sc.stressed_var_pct:.2%}",
            "Stressed ES": f"{sc.stressed_es_pct:.2%}",
            "Stressed Loss ($)": f"{sc.stressed_loss_value:,.0f}",
            "Worst": "*" if sc.worst_case else "",
        })
    st.dataframe(pd.DataFrame(srows), width="stretch", height=300)

    st.markdown("#### Scenario Loss Bar Chart")
    loss_df = pd.DataFrame({
        "Scenario": list(stress.scenarios.keys()),
        "Stressed Loss ($)": [sc.stressed_loss_value for sc in stress.scenarios.values()],
    }).sort_values("Stressed Loss ($)")
    fig_loss = go.Figure(go.Bar(
        x=loss_df["Stressed Loss ($)"], y=loss_df["Scenario"], orientation="h",
        marker_color=COLOR_NEGATIVE,
        text=loss_df["Stressed Loss ($)"].map(lambda v: f"${v:,.0f}"),
        textposition="auto",
        marker_line_width=0,
    ))
    _apply_chart_layout(fig_loss, title="Stressed Portfolio Loss by Scenario",
                        x_title="Loss ($)", y_title="")
    st.plotly_chart(fig_loss, width="stretch")

    st.markdown("#### Stress VaR Comparison")
    var_df = pd.DataFrame({
        "Scenario": list(stress.scenarios.keys()),
        "Stressed VaR (%)": [sc.stressed_var_pct for sc in stress.scenarios.values()],
        "Baseline VaR (%)": [abs(sc.baseline_var_pct) for sc in stress.scenarios.values()],
    })
    fig_svar = go.Figure()
    fig_svar.add_trace(go.Bar(
        x=var_df["Scenario"], y=var_df["Stressed VaR (%)"], name="Stressed VaR",
        marker_color=COLOR_NEGATIVE, marker_line_width=0,
    ))
    fig_svar.add_trace(go.Bar(
        x=var_df["Scenario"], y=var_df["Baseline VaR (%)"], name="Baseline VaR",
        marker_color=COLOR_PRIMARY, marker_line_width=0,
    ))
    _apply_chart_layout(fig_svar, y_title="VaR (%)", yformat=".1%")
    st.plotly_chart(fig_svar, width="stretch")

    st.markdown("#### Worst Loss Ranking")
    rank_loss = pd.DataFrame({
        "Scenario": list(stress.scenarios.keys()),
        "Stressed VaR (%)": [sc.stressed_var_pct for sc in stress.scenarios.values()],
    }).sort_values("Stressed VaR (%)")
    fig_rank = go.Figure(go.Bar(
        x=rank_loss["Stressed VaR (%)"], y=rank_loss["Scenario"], orientation="h",
        marker_color=COLOR_GARCH,
        text=rank_loss["Stressed VaR (%)"].map(lambda v: f"{v:.1%}"),
        textposition="auto",
        marker_line_width=0,
    ))
    _apply_chart_layout(fig_rank, title="Worst Loss Ranking (by stressed VaR)",
                        x_title="Stressed VaR (%)", y_title="")
    fig_rank.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(fig_rank, width="stretch")

    st.markdown("#### Sector Shock Impact")
    sec = stress.scenarios.get("Sector Shock")
    if sec is not None and sec.sector_contrib is not None:
        contrib = sec.sector_contrib.sort_values(ascending=False).head(10)
        fig_sec = go.Figure(go.Bar(
            x=contrib.index, y=contrib.values,
            marker_color=[COLOR_NEGATIVE if v > 0 else COLOR_PRIMARY for v in contrib.values],
            text=[f"${v:,.0f}" for v in contrib.values], textposition="auto",
            marker_line_width=0,
        ))
        _apply_chart_layout(fig_sec, title="Sector Shock - Loss Contribution by Asset",
                            y_title="Loss ($)", yformat="$,.0f")
        st.plotly_chart(fig_sec, width="stretch")
        st.caption(f"Stressed loss from sector shock: ${sec.stressed_loss_value:,.0f}")
    else:
        st.info("Sector shock impact unavailable.")

    st.markdown("---")
    st.markdown("#### Return & P&L Analysis")

    dd = result.portfolio_value / result.portfolio_value.cummax() - 1.0
    fig_dd = go.Figure(go.Scatter(
        x=dd.index, y=dd, mode="lines", fill="tozeroy",
        line=dict(color=COLOR_NEGATIVE, width=0.8),
        fillcolor="rgba(220,53,69,0.1)",
    ))
    _apply_chart_layout(fig_dd, title="Drawdown (portfolio value vs running peak)",
                        y_title="Drawdown", yformat=".1%")
    fig_dd.update_layout(height=360)
    st.plotly_chart(fig_dd, width="stretch")

    fig_dist = go.Figure(go.Histogram(
        x=result.portfolio_returns, nbinsx=60,
        marker_color=COLOR_HIST, opacity=0.8,
        marker_line=dict(width=0.3, color="#dee2e6"),
    ))
    _apply_chart_layout(fig_dist, title="Daily Return Distribution",
                        x_title="Daily Return", y_title="Frequency", yformat=".1%")
    fig_dist.update_layout(height=360)
    st.plotly_chart(fig_dist, width="stretch")

    st.markdown("---")
    st.markdown("#### Volatility Analysis")

    roll_win = int(min(250, max(20, len(result.portfolio_returns) // 3)))
    roll_hist_var = result.portfolio_returns.rolling(roll_win).quantile(alpha)
    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(
        x=result.portfolio_returns.index, y=result.portfolio_returns,
        mode="lines", name="Actual return",
        line=dict(color=COLOR_PRIMARY, width=0.6),
    ))
    fig_rv.add_trace(go.Scatter(
        x=roll_hist_var.index, y=roll_hist_var,
        mode="lines", name=f"Rolling Historical VaR ({roll_win}d)",
        line=dict(color=COLOR_NEGATIVE, width=1, dash="dot"),
    ))
    _apply_chart_layout(fig_rv, title="Rolling Historical VaR vs Actual Returns",
                        y_title="Return", yformat=".1%")
    st.plotly_chart(fig_rv, width="stretch")

    st.markdown("#### VaR vs Actual Loss (all models)")
    fig_val = go.Figure()
    for name, cr in cmp_results.items():
        fig_val.add_trace(go.Scatter(
            x=cr.backtest.actual_returns.index, y=cr.backtest.var_series,
            mode="lines", name=f"{name} VaR",
            line=dict(color=MODEL_COLORS.get(name), width=0.8, dash="dot"),
        ))
    any_cr = next(iter(cmp_results.values()))
    fig_val.add_trace(go.Scatter(
        x=any_cr.backtest.actual_returns.index, y=any_cr.backtest.actual_returns,
        mode="lines", name="Actual return",
        line=dict(color=COLOR_SECONDARY, width=0.6),
    ))
    _apply_chart_layout(fig_val, title="1-Day VaR Forecasts vs Realised Returns (backtest period)",
                        y_title="Return", yformat=".1%")
    st.plotly_chart(fig_val, width="stretch")

    st.markdown("#### VaR vs Expected Shortfall")
    var_es = pd.DataFrame({
        "Model": ["Historical", "EWMA", "GARCH", "GJR-GARCH"],
        "VaR (%)": [
            abs(np.quantile(result.portfolio_returns.values, alpha)),
            abs(ewma_full.var_pct),
            abs(garch_res.var_pct),
            abs(gjr_res.var_pct),
        ],
        "ES (%)": [
            abs(np.mean(result.portfolio_returns.values[
                result.portfolio_returns.values <= np.quantile(result.portfolio_returns.values, alpha)])),
            abs(ewma_full.es_pct),
            abs(garch_res.es_pct),
            abs(gjr_res.es_pct),
        ],
    })
    fig_ves = go.Figure()
    fig_ves.add_trace(go.Bar(
        x=var_es["Model"], y=var_es["VaR (%)"], name="VaR",
        marker_color=COLOR_PRIMARY, marker_line_width=0,
    ))
    fig_ves.add_trace(go.Bar(
        x=var_es["Model"], y=var_es["ES (%)"], name="Expected Shortfall",
        marker_color=COLOR_NEGATIVE, marker_line_width=0,
    ))
    _apply_chart_layout(fig_ves, y_title="Loss (%)", yformat=".1%")
    st.plotly_chart(fig_ves, width="stretch")

    st.markdown("#### Volatility Regime Comparison")
    roll_std = result.portfolio_returns.rolling(21).std()
    vol_cmp = pd.DataFrame({
        "Rolling (21d)": roll_std,
        "EWMA": ewma_full.conditional_volatility,
        "GARCH": garch_res.conditional_volatility,
        "GJR-GARCH": gjr_res.conditional_volatility,
    }).dropna()
    fig_vol = go.Figure()
    vcolors = {"Rolling (21d)": COLOR_PRIMARY, "EWMA": COLOR_EWMA, "GARCH": COLOR_GARCH, "GJR-GARCH": COLOR_GJR}
    for col in vol_cmp.columns:
        fig_vol.add_trace(go.Scatter(
            x=vol_cmp.index, y=vol_cmp[col], mode="lines", name=col,
            line=dict(color=vcolors[col], width=0.8),
        ))
    _apply_chart_layout(fig_vol, title="Conditional / Rolling Volatility by Model",
                        y_title="Daily volatility", yformat=".1%")
    st.plotly_chart(fig_vol, width="stretch")

    st.markdown("---")
    st.markdown("#### Backtesting Summary")

    bl_names = list(cmp_results.keys())
    bl_zones = [cmp_results[n].backtest.basel_zone for n in bl_names]
    fig_bl = go.Figure(go.Bar(
        x=bl_names, y=[1] * len(bl_names),
        marker_color=[zone_color[z] for z in bl_zones],
        text=[f"{z} ({cmp_results[n].backtest.n_exceptions} exc)" for n, z in zip(bl_names, bl_zones)],
        textposition="inside",
        marker_line_width=0,
    ))
    fig_bl.update_layout(yaxis=dict(visible=False))
    _apply_chart_layout(fig_bl, title="Basel Traffic Light by Model")
    fig_bl.update_layout(height=220)
    st.plotly_chart(fig_bl, width="stretch")

    fig_exc = go.Figure(go.Bar(
        x=bl_names, y=[cmp_results[n].backtest.n_exceptions for n in bl_names],
        marker_color=[zone_color[cmp_results[n].backtest.basel_zone] for n in bl_names],
        text=[cmp_results[n].backtest.n_exceptions for n in bl_names], textposition="auto",
        marker_line_width=0,
    ))
    _apply_chart_layout(fig_exc, title="Observed Exceptions by Model", y_title="Exceptions")
    fig_exc.update_layout(height=340)
    st.plotly_chart(fig_exc, width="stretch")

    fig_ea = go.Figure()
    fig_ea.add_trace(go.Bar(
        x=bl_names, y=[cmp_results[n].backtest.expected_exceptions for n in bl_names],
        name="Expected", marker_color=COLOR_PRIMARY, marker_line_width=0,
    ))
    fig_ea.add_trace(go.Bar(
        x=bl_names, y=[cmp_results[n].backtest.n_exceptions for n in bl_names],
        name="Observed",
        marker_color=[zone_color[cmp_results[n].backtest.basel_zone] for n in bl_names],
        marker_line_width=0,
    ))
    _apply_chart_layout(fig_ea, title="Expected vs Actual Exceptions", y_title="Exceptions")
    st.plotly_chart(fig_ea, width="stretch")

# ── Tab: Returns ────────────────────────────────────────────────────────────
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
            line=dict(width=0.8, color=COLOR_EWMA),
        ),
        row=1, col=1,
    )
    cum_ret = (1 + result.portfolio_returns).cumprod() - 1
    fig_ret.add_trace(
        go.Scatter(
            x=cum_ret.index, y=cum_ret, mode="lines", name="Cumulative",
            line=dict(width=1.5, color=COLOR_PRIMARY), fill="tozeroy",
            fillcolor="rgba(13,110,253,0.08)",
        ),
        row=2, col=1,
    )
    fig_ret.update_layout(
        height=580,
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_BG,
        font=dict(size=CHART_FONT_SIZE, color=CHART_FONT, family="Inter"),
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    fig_ret.update_xaxes(gridcolor=CHART_GRID, gridwidth=0.5, linecolor="#dee2e6", linewidth=0.5)
    fig_ret.update_yaxes(gridcolor=CHART_GRID, gridwidth=0.5, linecolor="#dee2e6", linewidth=0.5, tickformat=".1%")
    st.plotly_chart(fig_ret, width="stretch")

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
        st.markdown("#### Return Statistics")
        st.table(pd.Series(stats, name="Value"))

# ── Tab: Analytics ──────────────────────────────────────────────────────────
with tab_ana:
    st.markdown("#### Diversification & Concentration")
    div_c1, div_c2, div_c3 = st.columns(3)
    div_c1.metric("Herfindahl Index (HHI)", f"{div_metrics.get('HHI', 0):.3f}")
    div_c2.metric("Effective Number of Assets", f"{div_metrics.get('Effective N', 0):.2f}")
    div_c3.metric("Largest Weight", f"{div_metrics.get('Largest weight', 0):.2%}")

    st.markdown("#### Correlation Heatmap (Asset Returns)")
    corr_tickers = list(corr_matrix.columns)
    fig_heat = go.Figure(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_tickers,
            y=corr_tickers,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
        )
    )
    _apply_chart_layout(fig_heat, title="Asset Return Correlation",
                        x_title="", y_title="")
    fig_heat.update_layout(
        height=440,
        xaxis=dict(tickangle=-45, gridcolor=CHART_GRID),
        yaxis=dict(gridcolor=CHART_GRID),
        coloraxis_colorbar=dict(title="Corr."),
    )
    st.plotly_chart(fig_heat, width="stretch")

    st.markdown("#### Multi-Confidence VaR & Expected Shortfall")
    mve_rows = []
    for _, r in multi_var_es.iterrows():
        mve_rows.append(
            {
                "Confidence": f"{r['conf_level']:.0%}",
                "VaR (%)": f"{r['var_pct']:.2%}",
                "VaR ($)": f"${r['var_dollar']:,.0f}",
                "ES (%)": f"{r['es_pct']:.2%}",
                "ES ($)": f"${r['es_dollar']:,.0f}",
            }
        )
    st.dataframe(
        pd.DataFrame(mve_rows).set_index("Confidence"),
        use_container_width=True,
        hide_index=False,
    )
    mve_chart = go.Figure()
    mve_chart.add_trace(go.Bar(
        x=multi_var_es["confidence"].astype(str),
        y=-multi_var_es["var_dollar"],
        name="VaR ($)",
        marker_color=COLOR_NEGATIVE,
        marker_line_width=0,
    ))
    mve_chart.add_trace(go.Bar(
        x=multi_var_es["confidence"].astype(str),
        y=-multi_var_es["es_dollar"],
        name="ES ($)",
        marker_color=COLOR_GARCH,
        marker_line_width=0,
    ))
    _apply_chart_layout(
        mve_chart,
        title="Tail-Risk by Confidence Level (loss, $)",
        y_title="Loss ($)", yformat="$,.0f",
    )
    mve_chart.update_layout(height=380, barmode="group")
    st.plotly_chart(mve_chart, width="stretch")

    st.markdown("#### Return Attribution by Asset")
    attr_df = attribution.sort_values("contribution_pct", ascending=False)
    attr_colors = [
        COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE
        for v in attr_df["contribution_pct"]
    ]
    fig_attr = go.Figure(go.Bar(
        x=attr_df["asset"], y=attr_df["contribution_pct"],
        marker_color=attr_colors, text=attr_df["contribution_share"],
        texttemplate="%{text:.1%}", textposition="auto",
        marker_line_width=0,
    ))
    _apply_chart_layout(
        fig_attr,
        title="Contribution to Portfolio Return by Asset",
        y_title="Contribution (%)",
    )
    fig_attr.update_layout(height=380, xaxis=dict(tickangle=-45))
    st.plotly_chart(fig_attr, width="stretch")
    st.dataframe(
        pd.DataFrame(
            {
                "Asset": attr_df["asset"],
                "Total Return": attr_df["total_return"].map(lambda v: f"{v:.2%}"),
                "Weight": attr_df["weight"].map(lambda v: f"{v:.2%}"),
                "Contribution (%)": attr_df["contribution_pct"].map(lambda v: f"{v:.2%}"),
                "Contribution Share": attr_df["contribution_share"].map(lambda v: f"{v:.2%}"),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# ── Tab: Weights ────────────────────────────────────────────────────────────
with tab_w:
    st.markdown("#### Target Weights")
    tw = result.target_weights.sort_values(ascending=False)
    fig_w = go.Figure(go.Bar(
        x=tw.index, y=tw.values,
        marker_color=[COLOR_PRIMARY if v >= 0 else COLOR_NEGATIVE for v in tw.values],
        text=[f"{v:.1%}" for v in tw.values],
        textposition="outside",
        marker_line_width=0,
    ))
    _apply_chart_layout(fig_w, title="Target Allocation",
                        x_title="Ticker", y_title="Weight", yformat=".0%")
    fig_w.update_layout(height=340)
    st.plotly_chart(fig_w, width="stretch")

    st.markdown("#### Weight Evolution")
    show_cols = list(result.weights_history.columns)[:8]
    fig_wh = go.Figure()
    for col in show_cols:
        fig_wh.add_trace(go.Scatter(
            x=result.weights_history.index, y=result.weights_history[col],
            mode="lines", name=col, line=dict(width=0.8),
        ))
    _apply_chart_layout(fig_wh, title="Daily Weights Over Time",
                        y_title="Weight", yformat=".0%")
    fig_wh.update_layout(height=380)
    st.plotly_chart(fig_wh, width="stretch")

    with st.expander("Holdings (shares) - last 10 days"):
        st.dataframe(result.holdings.tail(10).style.format("{:.2f}"), width="stretch")

# ── Tab: Data ───────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("#### Daily Series")
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
        width="stretch",
        height=400,
    )
    csv = df_out.to_csv().encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="portfolio_daily.csv", mime="text/csv")

    st.markdown("#### Cleaning Summary")
    ov = data_overview(clean_prices)
    st.text(ov.summary_text())

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Scheme: {scheme} | Rebalance: {rebalance_freq} | "
    f"Cost: {transaction_cost_bps} bps | Period: {start_date} to {end_date}"
)

# ── Export Report ───────────────────────────────────────────────────────────
st.markdown("#### Export Report")
st.caption(
    "Download a self-contained HTML report (print-to-PDF friendly) or a bundle "
    "of CSV exports for further analysis."
)

try:
    backtest_rows_for_report = [
        {
            "model": name,
            "zone": cr.backtest.basel_zone,
            "ratio": cr.backtest.exception_ratio,
            "kupiec_p": cr.backtest.kupiec_pof_pvalue,
        }
        for name, cr in cmp_results.items()
    ]
    html_report = build_html_report(
        meta=meta,
        scheme=scheme,
        rebalance_freq=rebalance_freq,
        transaction_cost_bps=float(transaction_cost_bps),
        start_date=start_date,
        end_date=end_date,
        conf_str=conf_str,
        target_weights=result.target_weights,
        risk_metrics=risk_metrics,
        garch_res=garch_res,
        gjr_res=gjr_res,
        ewma_res=ewma_full,
        cmp_results=cmp_results,
        stress=stress,
        correlation_df=corr_matrix,
        div_metrics=div_metrics,
        multi_var_es=multi_var_es,
        attribution=attribution,
        backtest_rows=backtest_rows_for_report,
    )
    ex_c1, ex_c2 = st.columns(2)
    ex_c1.download_button(
        "Download HTML Report",
        data=html_report.encode("utf-8"),
        file_name="portfolio_risk_report.html",
        mime="text/html",
    )
    ex_c2.download_button(
        "Download Portfolio CSV",
        data=pd.DataFrame(
            {
                "portfolio_value": result.portfolio_value,
                "portfolio_return": result.portfolio_returns,
            }
        ).to_csv().encode("utf-8"),
        file_name="portfolio_daily_full.csv",
        mime="text/csv",
    )
except Exception as _e:
    st.warning(f"Report export unavailable: {_e}")
