import numpy as np
import pandas as pd

def calculate_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily percentage returns in decimal form from asset prices.

    Parameters:
        prices_df : pd.DataFrame
            DataFrame of positive adjusted-close prices indexed by trading date.

    Returns:
        pd.DataFrame
            Daily percentage returns in decimal form.

    Raises:
        ValueError
            If prices_df is empty or contains non-numeric data.
    """
    if prices_df.empty:
        raise ValueError("prices_df cannot be empty for return calculation.")
    
    if not pd.api.types.is_numeric_dtype(prices_df.iloc[0]):
        raise ValueError("prices_df must contain numeric price columns.")

    returns_df = prices_df.pct_change().dropna()
    return returns_df