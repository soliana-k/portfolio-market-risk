"""

Public interface
----------------
align_trading_dates
handle_missing_values
remove_stale_prices
validate_prices
calculate_log_returns
winsorize_returns
build_weighted_portfolio_returns
calculate_rolling_volatility
run_data_cleaning_pipeline
CleanedDataResult
"""

from src.data_cleaning.cleaning import (
    align_trading_dates,
    handle_missing_values,
    remove_stale_prices,
    validate_prices,
    calculate_log_returns,
    winsorize_returns,
    build_weighted_portfolio_returns,
    calculate_rolling_volatility,
    run_data_cleaning_pipeline,
    CleanedDataResult,
)

__all__ = [
    "align_trading_dates",
    "handle_missing_values",
    "remove_stale_prices",
    "validate_prices",
    "calculate_log_returns",
    "winsorize_returns",
    "build_weighted_portfolio_returns",
    "calculate_rolling_volatility",
    "run_data_cleaning_pipeline",
    "CleanedDataResult",
]