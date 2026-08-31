"""
Coverage for the VaR Model Comparison module (Phase 9):
- EWMA registered alongside Historical / GARCH / GJR-GARCH
- Rolling backtest over all four models
- Comparison metrics: average VaR, VaR volatility, worst daily loss, ES
- Model ranking (Green zone / exception-ratio deviation / Kupiec p-value)
- assemble_comparison reuses existing BacktestResults (incl. EWMA)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model_comparison.compare import (
    MODEL_ORDER,
    assemble_comparison,
    compare_models,
)
from src.var_backtesting.backtest import backtest_model


@pytest.fixture
def cmp_returns() -> pd.Series:
    rng = np.random.default_rng(77)
    n = 1000
    vol = 0.01 * np.exp(np.sin(np.arange(n) / 30.0))
    return pd.Series(
        0.0003 + rng.standard_normal(n) * vol,
        index=pd.bdate_range("2019-01-02", periods=n),
        name="portfolio_returns",
    )


def test_compare_models_four_models(cmp_returns):
    est = 750  # -> 250 out-of-sample days, fast enough
    results = compare_models(cmp_returns, confidence_level=0.95, estimation_window=est)
    present = set(results.keys())
    assert present == {"Historical", "EWMA", "GARCH", "GJR-GARCH"}
    for cr in results.values():
        assert cr.backtest.n_observations == len(cmp_returns) - est
        assert np.isfinite(cr.average_var)
        assert cr.var_volatility >= 0
        assert np.isfinite(cr.worst_daily_loss)
        assert np.isfinite(cr.expected_shortfall)


def test_compare_models_ranking_unique(cmp_returns):
    results = compare_models(cmp_returns, confidence_level=0.95, estimation_window=750)
    ranks = [cr.rank for cr in results.values()]
    assert sorted(ranks) == list(range(1, len(MODEL_ORDER) + 1))


def test_assemble_reuses_backtest_results(cmp_returns):
    est = 750
    hist_bt = backtest_model(
        cmp_returns, lambda tr, cl, pv: float(np.quantile(tr.values, 1.0 - cl)),
        model_name="Historical", confidence_level=0.95, estimation_window=est,
    )
    garch_bt = backtest_model(
        cmp_returns, lambda tr, cl, pv: -0.02,  # trivial stand-in, no fitting
        model_name="GARCH", confidence_level=0.95, estimation_window=est,
    )
    merged = assemble_comparison(
        {"Historical": hist_bt, "GARCH": garch_bt}, 0.95
    )
    assert set(merged.keys()) == {"Historical", "GARCH"}
    assert merged["Historical"].backtest is hist_bt


def test_ranking_prefers_green_zone(cmp_returns):
    # Build two synthetic BacktestResults: one Green, one Red.
    from src.var_backtesting.backtest import BacktestResult

    def _fake(name, zone, ratio):
        return BacktestResult(
            model_name=name, confidence_level=0.95, estimation_window=750,
            backtest_dates=cmp_returns.index[750:],
            actual_returns=cmp_returns.iloc[750:],
            var_series=cmp_returns.iloc[750:] * 0.0 - 0.02,
            exceptions=(cmp_returns.iloc[750:] < -0.02).astype(int),
            n_observations=250, n_exceptions=int(ratio * 250),
            expected_exceptions=12.5, exception_ratio=ratio,
            kupiec_pof_lr=0.0, kupiec_pof_pvalue=0.5,
            christoffersen_independence_lr=0.0, christoffersen_independence_pvalue=0.5,
            christoffersen_cc_lr=0.0, christoffersen_cc_pvalue=0.5,
            basel_zone=zone, pass_fail="Pass" if zone == "Green" else "Fail",
        )

    assembled = assemble_comparison(
        {"A": _fake("A", "Red", 0.10), "B": _fake("B", "Green", 0.05)},
        0.95,
    )
    assert assembled["B"].rank < assembled["A"].rank


def test_compare_models_bad_model(cmp_returns):
    with pytest.raises(ValueError):
        compare_models(cmp_returns, models=["NotAModel"])
