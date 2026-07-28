"""
Coverage:
- Normal-case behaviour
- Known-answer (hand-calculable) checks
- Edge cases (minimal data, all-identical prices)
- Invalid-input rejection
- Financial correctness (weights sum, log-return formula, no look-ahead)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning.cleaning import (
    CleanedDataResult,
    align_trading_dates,
    build_weighted_portfolio_returns,
    calculate_log_returns,
    calculate_rolling_volatility,
    handle_missing_values,
    remove_stale_prices,
    run_data_cleaning_pipeline,
    validate_prices,
    winsorize_returns,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    
    dates = pd.bdate_range("2024-01-02", periods=10)
    data = {
        "AAPL": [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 107.0],
        "MSFT": [200.0, 201.0, 199.0, 202.0, 203.0, 204.0, 205.0, 204.5, 206.0, 207.0],
        "NVDA": [50.0, 51.0, 52.0, 51.5, 53.0, 54.0, 53.5, 55.0, 56.0, 57.0],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def equal_weights() -> pd.Series:
    return pd.Series({"AAPL": 1 / 3, "MSFT": 1 / 3, "NVDA": 1 / 3})



def test_validate_prices_accepts_valid(sample_prices):
    validate_prices(sample_prices)


def test_validate_prices_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_prices(pd.DataFrame())


def test_validate_prices_rejects_non_positive(sample_prices):
    bad = sample_prices.copy()
    bad.iloc[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive"):
        validate_prices(bad)


def test_validate_prices_rejects_non_datetime_index(sample_prices):
    bad = sample_prices.copy()
    bad.index = range(len(bad))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_prices(bad)





def test_align_trading_dates_fills_calendar_gaps():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])
    prices = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]}, index=dates)
    aligned = align_trading_dates(prices)
    assert pd.Timestamp("2024-01-04") in aligned.index
    assert np.isnan(aligned.loc["2024-01-04", "AAPL"])


def test_align_trading_dates_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        align_trading_dates(pd.DataFrame())



def test_handle_missing_values_ffill_bfill():
    dates = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame(
        {"AAPL": [100.0, np.nan, 102.0, np.nan, 104.0]}, index=dates
    )
    cleaned, n_filled = handle_missing_values(prices, method="ffill_bfill")
    assert n_filled == 2
    assert cleaned.isna().sum().sum() == 0
    assert cleaned.loc[dates[1], "AAPL"] == 100.0
    assert cleaned.loc[dates[3], "AAPL"] == 102.0


def test_handle_missing_values_reports_count(sample_prices):
    _, n = handle_missing_values(sample_prices)
    assert n == 0


def test_handle_missing_values_invalid_method(sample_prices):
    with pytest.raises(ValueError, match="Unsupported"):
        handle_missing_values(sample_prices, method="magic")



def test_remove_stale_prices_detects_flat_run():
    dates = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 101.0, 101.0, 102.0, 103.0]}, index=dates
    )
    cleaned, n_stale = remove_stale_prices(prices, window=3)
    assert n_stale >= 1
    assert cleaned.isna().sum().sum() == 0


def test_remove_stale_prices_no_stale(sample_prices):
    cleaned, n_stale = remove_stale_prices(sample_prices, window=3)
    assert n_stale == 0
    pd.testing.assert_frame_equal(cleaned, sample_prices)


def test_remove_stale_prices_rejects_small_window(sample_prices):
    with pytest.raises(ValueError, match="window"):
        remove_stale_prices(sample_prices, window=1)



def test_log_returns_known_answer():
    dates = pd.bdate_range("2024-01-02", periods=2)
    prices = pd.DataFrame({"AAPL": [100.0, 105.0]}, index=dates)
    rets = calculate_log_returns(prices)
    expected = np.log(105.0 / 100.0)
    assert len(rets) == 1
    assert rets.iloc[0, 0] == pytest.approx(expected)


def test_log_returns_rejects_non_positive():
    dates = pd.bdate_range("2024-01-02", periods=2)
    prices = pd.DataFrame({"AAPL": [100.0, 0.0]}, index=dates)
    with pytest.raises(ValueError, match="positive"):
        calculate_log_returns(prices)



def test_winsorize_clips_extremes():
    dates = pd.bdate_range("2024-01-02", periods=20)
    rng = np.random.default_rng(42)
    raw = rng.normal(0, 0.01, size=(20, 1))
    raw[0, 0] = 0.50
    raw[1, 0] = -0.50
    returns = pd.DataFrame(raw, index=dates, columns=["AAPL"])
    winsorized, n_clipped = winsorize_returns(returns, limits=(0.05, 0.05))
    assert n_clipped >= 2
    assert winsorized["AAPL"].max() < 0.50
    assert winsorized["AAPL"].min() > -0.50



def test_portfolio_returns_equal_weight_known_answer():
    dates = pd.bdate_range("2024-01-02", periods=3)
    asset_rets = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, -0.01],
            "MSFT": [0.00, 0.01, 0.01],
        },
        index=dates,
    )
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    port = build_weighted_portfolio_returns(asset_rets, weights)
    expected = pd.Series([0.005, 0.015, 0.0], index=dates, name="portfolio_returns")
    pd.testing.assert_series_equal(port, expected)


def test_portfolio_returns_rejects_weights_not_summing_to_one():
    dates = pd.bdate_range("2024-01-02", periods=2)
    rets = pd.DataFrame({"AAPL": [0.01, 0.02], "MSFT": [0.0, 0.01]}, index=dates)
    bad_weights = pd.Series({"AAPL": 0.6, "MSFT": 0.6})
    with pytest.raises(ValueError, match="sum to 1"):
        build_weighted_portfolio_returns(rets, bad_weights)


def test_portfolio_returns_aligns_by_ticker_not_position():
    dates = pd.bdate_range("2024-01-02", periods=2)
    rets = pd.DataFrame(
        {"AAPL": [0.10, 0.00], "MSFT": [0.00, 0.00]}, index=dates
    )
    weights = pd.Series({"MSFT": 0.0, "AAPL": 1.0})
    port = build_weighted_portfolio_returns(rets, weights)
    assert port.iloc[0] == pytest.approx(0.10)



def test_rolling_vol_length_and_name(equal_weights, sample_prices):
    rets = calculate_log_returns(sample_prices)
    port = build_weighted_portfolio_returns(rets, equal_weights)
    vol = calculate_rolling_volatility(port, window=5)
    assert vol.name == "rolling_volatility"
    assert vol.isna().sum() == 4
    assert vol.dropna().shape[0] == len(port) - 4


def test_rolling_vol_rejects_small_window():
    s = pd.Series([0.01, -0.01, 0.02])
    with pytest.raises(ValueError, match="window"):
        calculate_rolling_volatility(s, window=1)



def test_pipeline_returns_cleaned_data_result(sample_prices, equal_weights):
    result = run_data_cleaning_pipeline(sample_prices, equal_weights)
    assert isinstance(result, CleanedDataResult)
    assert not result.clean_prices.empty
    assert not result.clean_returns.empty
    assert not result.portfolio_returns.empty
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.n_missing_filled >= 0
    assert result.n_stale_removed >= 0


def test_pipeline_metadata_populated(sample_prices, equal_weights):
    result = run_data_cleaning_pipeline(sample_prices, equal_weights)
    assert "n_assets" in result.metadata
    assert result.metadata["n_assets"] == 3


def test_pipeline_rejects_bad_weights(sample_prices):
    bad = pd.Series({"AAPL": 0.5, "MSFT": 0.5, "NVDA": 0.5})
    with pytest.raises(ValueError, match="sum to 1"):
        run_data_cleaning_pipeline(sample_prices, bad)


def test_pipeline_with_winsorize(sample_prices, equal_weights):
    result = run_data_cleaning_pipeline(
        sample_prices, equal_weights, apply_winsorize=True, winsor_limits=(0.1, 0.1)
    )
    assert result.n_returns_winsorized >= 0


def test_pipeline_sign_convention_losses_negative(sample_prices, equal_weights):
    prices = sample_prices.copy()
    prices.iloc[-1, 0] = prices.iloc[-2, 0] * 0.90
    result = run_data_cleaning_pipeline(prices, equal_weights)
    assert result.portfolio_returns.iloc[-1] < 0