import pandas as pd
import numpy as np

from typing import Any


class SecurityRating(object):
    def __init__(self, price_data: pd.DataFrame):
        """
        Initialize with price data.

        Args:
            price_data (pd.DataFrame): DataFrame containing datetime index and float values representing prices.
        """
        self.price_data = price_data
        self.returns = self.price_data.pct_change().dropna()

    @property
    def mean_return(self) -> float:
        """Mean return over the period."""
        return self.returns.mean()

    @property
    def volatility(self) -> float:
        """Volatility (standard deviation of returns)."""
        return self.returns.std()

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown."""
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    @property
    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Sharpe ratio.

        Args:
            risk_free_rate (float): Risk-free rate, default is 2%.

        Returns:
            float: Sharpe ratio.
        """
        excess_return = self.mean_return - risk_free_rate / len(self.returns)
        return excess_return / self.volatility

    @property
    def sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Sortino ratio.

        Args:
            risk_free_rate (float): Risk-free rate, default is 2%.

        Returns:
            float: Sortino ratio.
        """
        downside = self.returns[self.returns < 0]
        downside_std = downside.std()
        excess_return = self.mean_return - risk_free_rate / len(self.returns)
        return excess_return / downside_std if downside_std != 0 else np.nan

    @property
    def trend_slope(self) -> np.ndarray[Any, Any]:
        """Trend slope using linear regression on log prices."""
        log_prices = np.log(self.price_data).values.flatten()
        x = np.arange(len(log_prices))
        slope = np.polyfit(x, log_prices, 1)[0]
        return slope

    def summary(self) -> pd.Series:
        """Provide a summary of all metrics."""
        return pd.Series(
            {
                "Mean Return": self.mean_return,
                "Volatility": self.volatility,
                "Max Drawdown": self.max_drawdown,
                "Sharpe Ratio": self.sharpe_ratio,
                "Sortino Ratio": self.sortino_ratio,
                "Trend Slope": self.trend_slope,
            }
        )

