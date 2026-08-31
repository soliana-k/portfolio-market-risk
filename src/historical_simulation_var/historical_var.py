from __future__ import annotations
import numpy as np
import pandas as pd

class HistoricalSimulation:
    def __init__(self, portfolio_values: pd.Series, portfolio_pnl: pd.Series, confidence_level: float = 0.95):
        """
        :param portfolio_values: Series of daily portfolio values ($)
        :param portfolio_pnl: Series of daily dollar PnL ($)
        :param confidence_level: Confidence level for VaR (e.g., 0.95, 0.99, 0.995)
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
        historical_var_dollar = sorted_pnl[var_index] if len(sorted_pnl) > 0 else 0.0
        
        current_val = self.portfolio_values.iloc[-1] if len(self.portfolio_values) > 0 else 1.0
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

    def rolling_var_series(self, window: int) -> pd.DataFrame:
        """
        Computes rolling historical VaR and Expected Shortfall over a specified lookback window.
        
        """
        alpha = 1.0 - self.confidence_level
        
        rolling_var = []
        rolling_es = []
        dates = []

        pnl_vals = self.portfolio_pnl.values
        val_vals = self.portfolio_values.reindex(self.portfolio_pnl.index).values
        idx = self.portfolio_pnl.index

        for i in range(window, len(pnl_vals) + 1):
            window_pnl = pnl_vals[i - window : i]
            current_val = val_vals[i - 1]
            
            sorted_window = np.sort(window_pnl)
            var_idx = int(np.floor(alpha * window))
            var_dollar = sorted_window[max(0, var_idx)]
            
            tail_losses = sorted_window[sorted_window <= var_dollar]
            es_dollar = tail_losses.mean() if len(tail_losses) > 0 else var_dollar
            
            rolling_var.append(abs(var_dollar))
            rolling_es.append(abs(es_dollar))
            dates.append(idx[i - 1])

        return pd.DataFrame({
            "rolling_var": rolling_var,
            "rolling_es": rolling_es
        }, index=pd.Index(dates, name="Date"))