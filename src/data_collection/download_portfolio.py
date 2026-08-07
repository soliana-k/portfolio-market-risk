import pandas as pd
from typing import List
import numpy as np
import yfinance as yf
import time

def download_asset_prices(
    tickers: List[str], 
    start_date: str, 
    end_date: str, 
    interval: str ='1d',
    market_benchmark: str = "^GSPC"
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for portfolio assets and market benchmark.

    Parameters:

        tickers : List[str]
            List of equity ticker symbols.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
        interval : str, optional
            Data frequency interval (e.g., '1d', '1wk', '1mo'), by default '1d'.
        market_benchmark : str, optional
            Ticker symbol for the market benchmark, by default '^GSPC'.

    Returns:

        pd.DataFrame
            DataFrame of adjusted closing prices indexed by date.

    Raises:

        ValueError
            If tickers list is empty or downloaded data is empty.
        RuntimeError
            If network or API execution fails.
    """
    if not tickers:
        raise ValueError("tickers list cannot be empty.")
    

    all_symbols = tickers + [market_benchmark]
    data_frames = {}

    for ticker in all_symbols:
        try:
            df = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                interval=interval,
                auto_adjust=True,
                progress=False,
                multi_level_index=False
            )
            if not df.empty and 'Close' in df.columns:
                data_frames[ticker] = df['Close']
            time.sleep(1.0)
        except Exception as e:
            raise RuntimeError(f"Failed to download data for {ticker}: {e}")

    prices_df = pd.DataFrame(data_frames).dropna(how="all").ffill()
    
    if prices_df.empty:
        raise ValueError("Downloaded price data is empty. Check dates or network connection.")
        
    return prices_df

def download_market_caps(tickers: List[str]) -> pd.Series:
    """Fetch latest market capitalisation for each ticker (USD)."""
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mc = info.get("marketCap") or info.get("totalAssets")
            if mc is None or mc <= 0:
                shares = info.get("sharesOutstanding")
                price = info.get("currentPrice") or info.get("regularMarketPrice")
                if shares and price:
                    mc = shares * price
            caps[t] = float(mc) if mc else np.nan
        except Exception:
            caps[t] = np.nan
        time.sleep(0.3)
    s = pd.Series(caps, dtype=float)
    if s.isna().all():
        raise ValueError("Could not retrieve market caps for any ticker.")
    return s