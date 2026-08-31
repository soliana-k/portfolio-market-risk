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

def _fetch_market_cap(ticker: str, retries: int = 3, sleep_sec: float = 1.0) -> float:
    """Robustly fetch a single ticker's market capitalisation (USD).

    The Yahoo Finance ``info`` endpoint is frequently rate-limited, which can
    return an empty or partially-populated dict. We therefore retry a few times
    and prefer explicit fields (marketCap, shares*price) over the fund-only
    ``totalAssets`` field, which is ``None`` for ordinary equities.
    """
    info: dict = {}
    for attempt in range(retries):
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}

        mc = info.get("marketCap")
        valid_mc = isinstance(mc, (int, float)) and mc > 0
        if not valid_mc:
            shares = info.get("sharesOutstanding")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if isinstance(shares, (int, float)) and shares > 0 and isinstance(price, (int, float)) and price > 0:
                mc = shares * price
                valid_mc = isinstance(mc, (int, float)) and mc > 0

        if valid_mc:
            return float(mc)

        if attempt < retries - 1:
            time.sleep(sleep_sec)

    # Fall back to fund/ETF total assets if still unavailable.
    total = info.get("totalAssets")
    if isinstance(total, (int, float)) and total > 0:
        return float(total)

    return float("nan")


def download_market_caps(tickers: List[str]) -> pd.Series:
    """Fetch latest market capitalisation for each ticker (USD).

    Raises:
        ValueError: If no market cap could be retrieved for any ticker.
    """
    caps = {}
    for t in tickers:
        caps[t] = _fetch_market_cap(t)
        time.sleep(0.5)
    s = pd.Series(caps, dtype=float)
    if s.isna().all():
        raise ValueError("Could not retrieve market caps for any ticker.")
    return s