import numpy as np
import pandas as pd
from typing import Dict
from src.data_collection.download_portfolio import download_asset_prices
from src.data_collection.returns import calculate_daily_returns

def build_portfolio_phase_one(
    portfolio_weights: Dict[str, float], 
    start_date: str, 
    end_date: str, 
    market_benchmark: str = "^GSPC",
    output_filepath: str = None
) -> Dict[str, pd.Series | pd.DataFrame]:
    """
    Validate weights, download prices, calculate returns, and assemble Phase 1 dataset.

    Parameters:
        portfolio_weights : Dict[str, float]
            Portfolio weights mapped by ticker symbol, summing to 1.0.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
        market_benchmark : str, optional
            Market index ticker, by default '^GSPC'.
        output_filepath : str, optional
            Path to export CSV of portfolio prices, by default None.

    Returns:
        Dict[str, pd.Series | pd.DataFrame]
            A structured data contract containing prices, returns, weights, and benchmarks.

    Raises:
        ValueError
            If portfolio weights do not sum strictly to 1.0 within tolerance.
    """
    total_weights = sum(portfolio_weights.values())
    if not np.isclose(total_weights, 1.0, atol=1e-2):
        raise ValueError(
            f"Portfolio weights must sum to 1.0. Current sum: {total_weights:.4f}"
        )

    tickers = list(portfolio_weights.keys())
    
    
    prices_df = download_asset_prices(
        tickers=tickers, 
        start_date=start_date, 
        end_date=end_date, 
        market_benchmark=market_benchmark
    )

    portfolio_prices = prices_df[tickers]
    market_prices = prices_df[[market_benchmark]] if market_benchmark in prices_df.columns else pd.DataFrame()

    portfolio_returns = calculate_daily_returns(portfolio_prices)
    market_returns = calculate_daily_returns(market_prices) if not market_prices.empty else pd.DataFrame()

    weights_series = pd.Series(portfolio_weights)
    aligned_weights = weights_series.reindex(portfolio_prices.columns)
    if aligned_weights.isna().any():
        raise ValueError("Missing weights for one or more portfolio assets.")

    
    if output_filepath:
        portfolio_prices.to_csv(output_filepath)

    return {
        "prices": portfolio_prices,
        "returns": portfolio_returns,
        "weights": aligned_weights,
        "market_prices": market_prices,
        "market_returns": market_returns
    }