import pytest
import pandas as pd
from unittest.mock import patch
from src.data_collection.download_portfolio import download_asset_prices

def test_download_empty_tickers_raises_error():
    """
    Test that an empty ticker list triggers a ValueError early.
    
    """
    with pytest.raises(ValueError, match="tickers list cannot be empty"):
        download_asset_prices([], "2024-01-01", "2024-01-10")

@patch('src.data_collection.download_portfolio.yf.download')
def test_download_asset_prices_success(mock_yf_download):
    """
    Test normal-case execution with mocked data returns expected schema.
    
    """
    
    mock_df = pd.DataFrame({
        'Close': [100.0, 101.0, 102.0]
    }, index=pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04']))
    
    mock_yf_download.return_value = mock_df

    tickers = ['AAPL']
    result = download_asset_prices(tickers, "2024-01-01", "2024-01-05", market_benchmark="SPY")

    assert isinstance(result, pd.DataFrame)
    assert 'AAPL' in result.columns
    assert 'SPY' in result.columns
    assert not result.empty