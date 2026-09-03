"""
Tests for Phase 3 – Portfolio Construction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio_construction.portfolio import (
    construct_portfolio,
    resolve_target_weights,
    make_equal_weights,
    make_user_weights,
    make_long_short_weights,
    _rebalance_mask,
    _normalize_weights,
    PortfolioResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """100 business days, 3 assets, deterministic random walk."""
    dates = pd.bdate_range("2023-01-02", periods=100)
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0004, 0.012, size=(100, 3))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(prices, index=dates, columns=["AAPL", "MSFT", "GOOGL"])


@pytest.fixture
def equal_weights() -> pd.Series:
    return make_equal_weights(["AAPL", "MSFT", "GOOGL"])


# ─────────────────────────────────────────────────────────────────────────────
# Weight helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeEqualWeights:
    def test_basic(self):
        w = make_equal_weights(["A", "B", "C"])
        assert len(w) == 3
        assert np.isclose(w.sum(), 1.0)
        np.testing.assert_allclose(w, 1.0 / 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No tickers"):
            make_equal_weights([])


class TestMakeUserWeights:
    def test_normalises(self):
        w = make_user_weights(["A", "B"], {"A": 0.3, "B": 0.7})
        assert np.isclose(w.sum(), 1.0)
        assert np.isclose(w["A"], 0.3)
        assert np.isclose(w["B"], 0.7)

    def test_renormalises_when_not_one(self):
        w = make_user_weights(["A", "B"], {"A": 2.0, "B": 2.0})
        assert np.isclose(w.sum(), 1.0)
        assert np.isclose(w["A"], 0.5)

    def test_missing_ticker_gets_zero_then_renorm(self):
        w = make_user_weights(["A", "B", "C"], {"A": 0.6, "B": 0.4})
        assert "C" in w.index
        assert np.isclose(w.sum(), 1.0)


class TestMakeLongShortWeights:
    def test_dollar_neutral(self):
        w = make_long_short_weights(
            tickers=["A", "B", "C", "D"],
            long_tickers=["A", "B"],
            short_tickers=["C", "D"],
            long_weight=0.5,
            short_weight=0.5,
        )
        assert np.isclose(w.abs().sum(), 1.0)
        assert np.isclose(w[w > 0].sum(), 0.5)
        assert np.isclose(w[w < 0].sum(), -0.5)

    def test_net_long(self):
        w = make_long_short_weights(
            tickers=["A", "B"],
            long_tickers=["A"],
            short_tickers=["B"],
            long_weight=0.7,
            short_weight=0.3,
        )
        assert np.isclose(w.abs().sum(), 1.0)
        assert w["A"] > 0
        assert w["B"] < 0

    def test_empty_legs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            make_long_short_weights(["A"], [], [], 0.5, 0.5)


class TestNormalizeWeights:
    def test_long_only(self):
        w = pd.Series({"A": 2.0, "B": 2.0})
        out = _normalize_weights(w, allow_negative=False)
        assert np.isclose(out.sum(), 1.0)

    def test_long_short(self):
        w = pd.Series({"A": 3.0, "B": -1.0})
        out = _normalize_weights(w, allow_negative=True)
        assert np.isclose(out.abs().sum(), 1.0)


class TestResolveTargetWeights:
    def test_equal(self):
        w = resolve_target_weights("equal", ["X", "Y"])
        assert np.isclose(w.sum(), 1.0)
        assert (w == 0.5).all()

    def test_user(self):
        w = resolve_target_weights(
            "user", ["X", "Y"], user_weights={"X": 0.6, "Y": 0.4}
        )
        assert np.isclose(w["X"], 0.6)

    def test_long_short(self):
        w = resolve_target_weights(
            "long_short",
            ["A", "B"],
            long_tickers=["A"],
            short_tickers=["B"],
            long_weight=0.6,
            short_weight=0.4,
        )
        assert w["A"] > 0
        assert w["B"] < 0

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown weight scheme"):
            resolve_target_weights("foo", ["A"])  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Rebalance mask
# ─────────────────────────────────────────────────────────────────────────────

class TestRebalanceMask:
    def test_none(self):
        dates = pd.bdate_range("2023-01-01", periods=20)
        mask = _rebalance_mask(dates, "none")
        assert mask.sum() == 1
        assert bool(mask.iloc[0])

    def test_daily(self):
        dates = pd.bdate_range("2023-01-01", periods=10)
        mask = _rebalance_mask(dates, "daily")
        assert mask.all()

    def test_monthly(self):
        dates = pd.bdate_range("2023-01-01", periods=100)
        mask = _rebalance_mask(dates, "monthly")
        # roughly 5 months → ~5 rebalances
        assert 4 <= mask.sum() <= 6

    def test_weekly(self):
        dates = pd.bdate_range("2023-01-01", periods=20)
        mask = _rebalance_mask(dates, "weekly")
        assert 3 <= mask.sum() <= 5

    def test_quarterly(self):
        dates = pd.bdate_range("2023-01-01", periods=200)
        mask = _rebalance_mask(dates, "quarterly")
        assert 3 <= mask.sum() <= 5

    def test_unknown_freq_raises(self):
        dates = pd.bdate_range("2023-01-01", periods=5)
        with pytest.raises(ValueError, match="Unknown rebalance frequency"):
            _rebalance_mask(dates, "yearly")  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# construct_portfolio – core behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestConstructPortfolio:
    def test_returns_portfolio_result(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="monthly"
        )
        assert isinstance(res, PortfolioResult)
        assert isinstance(res.portfolio_value, pd.Series)
        assert isinstance(res.portfolio_pnl, pd.Series)
        assert isinstance(res.portfolio_returns, pd.Series)
        assert isinstance(res.weights_history, pd.DataFrame)
        assert isinstance(res.holdings, pd.DataFrame)

    def test_initial_value(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=50_000, rebalance_freq="none"
        )
        assert np.isclose(res.portfolio_value.iloc[0], 50_000, rtol=1e-6)

    def test_lengths_match(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="monthly"
        )
        n = len(sample_prices)
        assert len(res.portfolio_value) == n
        assert len(res.portfolio_pnl) == n
        assert len(res.portfolio_returns) == n
        assert len(res.weights_history) == n
        assert len(res.holdings) == n

    def test_first_day_return_and_pnl_zero(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="none"
        )
        assert res.portfolio_returns.iloc[0] == 0.0
        assert res.portfolio_pnl.iloc[0] == 0.0

    def test_pnl_equals_value_diff(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="monthly"
        )
        expected_pnl = res.portfolio_value.diff().fillna(0)
        np.testing.assert_allclose(
            res.portfolio_pnl.values, expected_pnl.values, rtol=1e-8, atol=1e-6
        )

    def test_returns_consistent_with_pnl(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="none"
        )
        # r_t = pnl_t / value_{t-1}
        prev = res.portfolio_value.shift(1)
        expected = res.portfolio_pnl / prev
        expected.iloc[0] = 0.0
        np.testing.assert_allclose(
            res.portfolio_returns.values[1:], expected.values[1:], rtol=1e-8, atol=1e-8
        )

    def test_buy_and_hold_no_rebalance(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="none"
        )
        assert res.metadata["n_rebalances"] == 1  # only initial allocation
        # holdings should be constant after day 0
        assert (res.holdings.iloc[1:] == res.holdings.iloc[0]).all().all()

    def test_monthly_rebalances_count(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="monthly"
        )
        assert 4 <= res.metadata["n_rebalances"] <= 6

    def test_daily_rebalance(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="daily"
        )
        assert res.metadata["n_rebalances"] == len(sample_prices)

    def test_metadata_keys(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="monthly"
        )
        expected_keys = {
            "initial_value",
            "rebalance_freq",
            "transaction_cost_bps",
            "n_rebalances",
            "final_value",
            "total_return",
            "n_days",
        }
        assert expected_keys.issubset(res.metadata.keys())

    def test_total_return_formula(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="none"
        )
        expected = res.portfolio_value.iloc[-1] / 100_000 - 1.0
        assert np.isclose(res.metadata["total_return"], expected)

    def test_transaction_cost_reduces_value(self, sample_prices, equal_weights):
        res_no_cost = construct_portfolio(
            sample_prices,
            equal_weights,
            initial_value=100_000,
            rebalance_freq="monthly",
            transaction_cost_bps=0.0,
        )
        res_cost = construct_portfolio(
            sample_prices,
            equal_weights,
            initial_value=100_000,
            rebalance_freq="monthly",
            transaction_cost_bps=20.0,  # 20 bps
        )
        # with costs, final value should be lower (or equal if no turnover)
        assert res_cost.portfolio_value.iloc[-1] <= res_no_cost.portfolio_value.iloc[-1] + 1e-6

    def test_long_short_portfolio(self, sample_prices):
        w = make_long_short_weights(
            tickers=["AAPL", "MSFT", "GOOGL"],
            long_tickers=["AAPL", "MSFT"],
            short_tickers=["GOOGL"],
            long_weight=0.6,
            short_weight=0.4,
        )
        res = construct_portfolio(
            sample_prices, w, initial_value=100_000, rebalance_freq="monthly"
        )
        assert res.portfolio_value.iloc[0] > 0
        assert not res.portfolio_value.isna().any()

    def test_weights_sum_near_one_long_only(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="monthly"
        )
        # on rebalance days weights should be very close to target
        for d in res.rebalance_dates:
            day_w = res.weights_history.loc[d]
            assert np.isclose(day_w.sum(), 1.0, atol=1e-5)

    def test_empty_prices_raises(self, equal_weights):
        empty = pd.DataFrame(columns=["AAPL", "MSFT", "GOOGL"])
        with pytest.raises(ValueError, match="empty"):
            construct_portfolio(empty, equal_weights)

    def test_non_positive_initial_value_raises(self, sample_prices, equal_weights):
        with pytest.raises(ValueError, match="positive"):
            construct_portfolio(sample_prices, equal_weights, initial_value=0)

    def test_missing_weight_raises(self, sample_prices):
        bad_w = pd.Series({"AAPL": 0.5, "MSFT": 0.5})  # missing GOOGL
        with pytest.raises(ValueError, match="Missing target weights"):
            construct_portfolio(sample_prices, bad_w)


# ─────────────────────────────────────────────────────────────────────────────
# Edge / numerical checks
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalSanity:
    def test_no_nans_in_outputs(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="weekly"
        )
        assert not res.portfolio_value.isna().any()
        assert not res.portfolio_pnl.isna().any()
        assert not res.portfolio_returns.isna().any()

    def test_value_stays_positive(self, sample_prices, equal_weights):
        res = construct_portfolio(
            sample_prices, equal_weights, initial_value=100_000, rebalance_freq="monthly"
        )
        assert (res.portfolio_value > 0).all()

    def test_deterministic_with_seed(self):
        """Same inputs → same outputs."""
        dates = pd.bdate_range("2023-01-02", periods=50)
        rng = np.random.default_rng(0)
        px = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0, 0.01, (50, 2)), axis=0),
            index=dates,
            columns=["X", "Y"],
        )
        w = make_equal_weights(["X", "Y"])
        r1 = construct_portfolio(px, w, 10_000, "monthly")
        r2 = construct_portfolio(px, w, 10_000, "monthly")
        np.testing.assert_array_equal(r1.portfolio_value.values, r2.portfolio_value.values)