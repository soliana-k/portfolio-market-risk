import numpy as np
import pandas as pd
from requests_ratelimiter import LimiterSession
from typing import Dict
import yfinance as yf
import time

class CollectPortfolio:
    def __init__(self, portfolio: Dict[str, float], output_filepath: str):
        self.portfolio = portfolio
        self.rate_limiter = LimiterSession(per_second=0.5)
        self.tickers = list(portfolio.keys())
        
        total_weights = sum(self.portfolio.values())
        self.output_filepath = output_filepath

        if total_weights > 1.0:
            print(f'Error please check the portfolio weights, its currently {total_weights} and exceeds 100%')

    def collect_portfolio_data(self, start_date: str, end_date: str, interval: str):
        data = pd.DataFrame()
        try:
            data_frames = {}
            for ticker in self.tickers:
                df = yf.download(
                    ticker, 
                    # session=self.rate_limiter, 
                    start=start_date, 
                    end=end_date, 
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    multi_level_index=False
                )
                if not df.empty:
                    data_frames[ticker] = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
                time.sleep(3)  
            
            if data_frames:
                data = pd.DataFrame(data_frames).dropna(how="all").ffill()
            
            print(data)
        except Exception as e:
            print(f'error {e}')

        return data