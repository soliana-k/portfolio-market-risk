from __future__ import annotations
import numpy as np
import pandas as pd

class HistoricalSimulation:
    def __init__(self, portfolio_values: pd.Series, portfolio_pnl: pd.Series, confidence_level: float = 0.95):
        """
        :param portfolio_values: Series of daily portfolio values ($)
        :param portfolio_pnl: Series of daily dollar PnL ($)
        :param confidence_level: Confidence level for VaR (e.g., 0.95, 0.99)
        """
        self.portfolio_values = portfolio_values
        self.portfolio_pnl = portfolio_pnl.dropna()
        self.confidence_level = confidence_level

    def pnl_calculation(self) -> dict:
        """
        Computes historical VaR, Expected Shortfall, and returns distribution metrics.
        """
       
        sorted_pnl = np.sort(self.portfolio_pnl)
        
        alpha = 1.0 - self.confidence_level
        var_index = int(np.floor(alpha * len(sorted_pnl)))
        historical_var_dollar = sorted_pnl[var_index]
        
        current_val = self.portfolio_values.iloc[-1]
        historical_var_pct = historical_var_dollar / current_val

        tail_losses = sorted_pnl[sorted_pnl <= historical_var_dollar]
        expected_shortfall_dollar = tail_losses.mean() if len(tail_losses) > 0 else historical_var_dollar
        expected_shortfall_pct = expected_shortfall_dollar / current_val

        return {
            "var_dollar": abs(historical_var_dollar),
            "var_pct": abs(historical_var_pct),
            "es_dollar": abs(expected_shortfall_dollar),
            "es_pct": abs(expected_shortfall_pct),
            "sorted_pnl": sorted_pnl,
        }