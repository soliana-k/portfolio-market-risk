"""
End-to-end pipeline integration test (Phase: production hardening).

Runs the *exact* compute pipeline the Streamlit app executes - download,
cleaning, portfolio construction, VaR model fitting, backtesting, model
comparison, stress testing and the analytics helpers - on synthetic data with a
mocked Yahoo Finance downloader.

This is the regression test that would have caught runtime-only failures such
as a NameError in the pipeline (code that passes ``import app`` but crashes on
execution) because it exercises the whole chain headlessly, without network.
"""

from __future__ import annotations

from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning.cleaning import run_data_cleaning_pipeline
from src.data_collection.download_portfolio import download_asset_prices_cached
from src.historical_simulation_var.historical_var import HistoricalSimulation
from src.portfolio_construction.portfolio import (
    construct_portfolio,
    resolve_target_weights,
)
from src.garch_var.garch import fit_garch
from src.gjr_garch.gjr import fit_gjr_garch
from src.ewma_var.ewma import fit_ewma
from src.var_backtesting.backtest import run_all_backtests, backtest_model
from src.model_comparison.compare import assemble_comparison
from src.stress_testing.stress import run_stress_tests
from src.analytics.portfolio_metrics import (
    asset_correlation_matrix,
    asset_return_attribution,
    concentration_metrics,
    multi_confidence_var_es,
    portfolio_variance,
)

TICKERS = ["AAPL", "MSFT", "GOOGL"]
BENCHMARK = "^GSPC"
START, END = "2023-01-01", "2026-01-01"


def _make_price_data() -> dict:
    """Deterministic synthetic price paths for tickers + benchmark."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(START, END)
    n = len(idx)
    base = pd.Series(100.0, index=idx)
    out = {}
    for sym in TICKERS + [BENCHMARK]:
        drift = rng.normal(0.0003, 0.0)
        rets = np.r_[0.0, rng.normal(drift, 0.015, n - 1)]
        out[sym] = base * np.cumprod(1 + rets)
    return out


@pytest.fixture
def pipeline_inputs():
    """Patch yfinance + isolate cache, return the raw download DataFrame."""
    prices_by_sym = _make_price_data()

    def fake_download(sym, **kwargs):
        return pd.DataFrame(
            {"Close": prices_by_sym[sym]}, index=prices_by_sym[sym].index
        )

    with (
        TemporaryDirectory() as tmp,
        patch("src.data_collection.download_portfolio.yf.download", side_effect=fake_download),
        patch("src.data_collection.cache.DEFAULT_CACHE_DIR", __import__("pathlib").Path(tmp)),
    ):
        prices = download_asset_prices_cached(
            tickers=TICKERS,
            start_date=START,
            end_date=END,
            market_benchmark=BENCHMARK,
            force_refresh=True,
        )
        yield prices


def _run_pipeline(prices: pd.DataFrame) -> dict:
    """Mirror the app's pipeline call sequence exactly."""
    confidence_level = 0.95
    initial_value = 100_000.0
    rebalance_freq = "monthly"
    transaction_cost_bps = 5.0
    volatility_window = 21
    rolling_window = 250

    target_w = resolve_target_weights(scheme="equal", tickers=TICKERS)
    clean = run_data_cleaning_pipeline(
        prices=prices[TICKERS],
        weights=target_w,
        align_dates=True,
        missing_method="ffill_bfill",
        stale_window=3,
        apply_winsorize=False,
        vol_window=volatility_window,
    )
    clean_prices = clean.clean_prices

    result = construct_portfolio(
        prices=clean_prices,
        target_weights=target_w,
        initial_value=initial_value,
        rebalance_freq=rebalance_freq,
        transaction_cost_bps=transaction_cost_bps,
    )

    risk_sim = HistoricalSimulation(
        portfolio_values=result.portfolio_value,
        portfolio_pnl=result.portfolio_pnl,
        confidence_level=confidence_level,
    )
    risk_metrics = risk_sim.pnl_calculation()
    rolling_df = risk_sim.rolling_var_series(window=rolling_window)

    garch_res = fit_garch(
        result.portfolio_returns,
        confidence_level=confidence_level,
        portfolio_value=initial_value,
    )
    gjr_res = fit_gjr_garch(
        result.portfolio_returns,
        confidence_level=confidence_level,
        portfolio_value=initial_value,
    )

    bt_window = int(max(250, min(750, len(result.portfolio_returns) - 250)))
    bt_results = run_all_backtests(
        result.portfolio_returns,
        confidence_level=confidence_level,
        estimation_window=bt_window,
        portfolio_value=initial_value,
    )
    ewma_bt = backtest_model(
        result.portfolio_returns,
        lambda tr, cl, pv: fit_ewma(tr, confidence_level=cl, portfolio_value=pv).var_pct,
        model_name="EWMA",
        confidence_level=confidence_level,
        estimation_window=bt_window,
        portfolio_value=initial_value,
    )
    cmp_results = assemble_comparison({**bt_results, "EWMA": ewma_bt}, confidence_level)

    asset_returns = clean_prices.pct_change().dropna()
    ewma_full = fit_ewma(
        result.portfolio_returns,
        confidence_level=confidence_level,
        portfolio_value=initial_value,
    )
    stress = run_stress_tests(
        portfolio_value=initial_value,
        weights=result.target_weights,
        portfolio_returns=result.portfolio_returns,
        asset_returns=asset_returns,
        baseline_var_pct=gjr_res.var_pct,
        baseline_es_pct=gjr_res.es_pct,
        confidence_level=confidence_level,
    )

    analytics = {
        "corr": asset_correlation_matrix(asset_returns),
        "div": concentration_metrics(result.target_weights),
        "mve": multi_confidence_var_es(
            result.portfolio_returns,
            result.portfolio_value,
            confidences=[0.95, 0.975, 0.99, 0.995],
        ),
        "attr": asset_return_attribution(asset_returns, result.target_weights),
        "pvar": portfolio_variance(result.target_weights, asset_returns.cov()),
    }

    return {
        "result": result,
        "risk_metrics": risk_metrics,
        "rolling_df": rolling_df,
        "garch_res": garch_res,
        "gjr_res": gjr_res,
        "ewma_full": ewma_full,
        "bt_results": bt_results,
        "cmp_results": cmp_results,
        "stress": stress,
        "analytics": analytics,
    }


def test_download_returns_assets_and_benchmark(pipeline_inputs):
    prices = pipeline_inputs
    for sym in TICKERS + [BENCHMARK]:
        assert sym in prices.columns, f"missing {sym} in downloaded prices"
    assert len(prices) > 500, "should be enough trading days for backtesting"


def test_full_pipeline_runs_without_error(pipeline_inputs):
    out = _run_pipeline(pipeline_inputs)

    assert out["result"].portfolio_value.iloc[-1] > 0
    assert "var_dollar" in out["risk_metrics"]
    assert out["garch_res"].sigma_forecast > 0
    assert out["gjr_res"].sigma_forecast > 0
    assert len(out["bt_results"]) >= 3  # Historical, GARCH, GJR-GARCH
    assert len(out["cmp_results"]) >= 4  # + EWMA


def test_stress_ranking_well_formed(pipeline_inputs):
    out = _run_pipeline(pipeline_inputs)
    assert len(out["stress"].ranking) > 0
    worst_name = out["stress"].ranking[0]
    worst = out["stress"].scenarios[worst_name]
    assert worst.stressed_loss_value > 0
    assert worst.worst_case


def test_analytics_well_formed(pipeline_inputs):
    out = _run_pipeline(pipeline_inputs)
    ana = out["analytics"]
    assert list(ana["corr"].columns) == TICKERS
    assert ana["div"]["Effective N"] <= len(TICKERS)
    assert len(ana["mve"]) == 4
    assert len(ana["attr"]) == len(TICKERS)
    assert ana["pvar"] > 0


def test_benchmark_comparison_computable(pipeline_inputs):
    prices = pipeline_inputs
    out = _run_pipeline(prices)
    bench = prices[BENCHMARK].reindex(out["result"].portfolio_value.index).ffill()
    bench_rets = bench.pct_change().dropna()
    port_norm = out["result"].portfolio_value / out["result"].portfolio_value.iloc[0]
    bench_norm = (1 + bench_rets).cumprod()
    assert (port_norm.reindex(bench_norm.index).ffill() > 0).all().all()