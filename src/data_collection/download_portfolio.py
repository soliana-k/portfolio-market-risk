import pandas as pd
from typing import List
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