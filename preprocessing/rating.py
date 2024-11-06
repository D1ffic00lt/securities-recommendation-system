import pandas as pd
import numpy as np

from datetime import datetime
from typing import Any, Iterable
from pandas import Series


class SecurityPriceRating(object):
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


class NormalizedData(dict):
    def __init__(self, data: pd.DataFrame, keys: Iterable[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in keys:
            match key:
                case "ipo_date":
                    self[key] = self.normalize(
                        (datetime.now() - self.to_datetime(data[key])).apply(
                            lambda x: x.days
                        )
                    )
                case "maturity_date":
                    self[key] = self.normalize(
                        (self.to_datetime(data[key]) - datetime.now()).apply(lambda x: x.days)
                    )
                case _:
                    self[key] = self.normalize(data[key])

    @staticmethod
    def to_datetime(value: pd.Series) -> Series:
        return pd.to_datetime(value, format="%d/%m/%Y")

    @staticmethod
    def normalize(series: pd.Series) -> pd.Series:
        return 10 * (series - series.min()) / (series.max() - series.min())


class SecurityRating(object):
    def __init__(
        self, shares: pd.DataFrame, bonds: pd.DataFrame, etfs: pd.DataFrame
    ) -> None:
        shares = shares.copy()
        bonds = bonds.copy()
        etfs = etfs.copy()

        if isinstance(shares["ipo_date"].dtype.type, np.object_):
            shares["ipo_date"] = pd.to_datetime(shares["ipo_date"], format="%d/%m/%Y")
        if isinstance(bonds["maturity_date"].dtype.type, np.object_):
            bonds["maturity_date"] = pd.to_datetime(
                bonds["maturity_date"], format="%d/%m/%Y"
            )

        self._normalized_shares = NormalizedData(shares, ("issue_size", "ipo_date"))
        self._normalized_etfs = NormalizedData(etfs, ("fixed_commission", "num_shares"))
        self._normalized_bonds = NormalizedData(
            bonds,
            ("coupon_quantity_per_year", "maturity_date", "issue_size"),
        )

    @staticmethod
    def _country_risk_score(country: str) -> int:
        match country:
            case "RU":
                return 10
            case "US":
                return 8
            case x if x in ["DE", "FR", "EU"]:
                return 6
            case "CN":
                return 4
            case "KZ":
                return 5
            case _:
                return 2

    @staticmethod
    def _rebalancing_freq_score(freq: str) -> int:
        match freq:
            case "daily":
                return 10
            case "quarterly":
                return 8
            case "semi_annual":
                return 5
            case "annual":
                return 3
            case _:
                return 0

    @staticmethod
    def _focus_type_score(focus_type: str) -> int:
        match focus_type:
            case "equity":
                return 10
            case "fixed_income":
                return 8
            case "mixed_allocation":
                return 6
            case "commodity":
                return 4
            case "alternative_investment":
                return 2
            case _:
                return 0

    def calculate_shares_rating(self, row: pd.Series) -> float:
        liquidity = 10 if row["liquidity_flag"] else 0
        country_risk = self._country_risk_score(row["country_of_risk"])
        issue_size = self._normalized_shares["issue_size"][row.name]
        ipo_days = 10 - self._normalized_shares["ipo_date"][row.name]

        return 0.3 * liquidity + 0.2 * country_risk + 0.3 * issue_size + 0.2 * ipo_days

    def calculate_etfs_rating(self, row: pd.Series) -> float:
        rebalancing_freq = self._rebalancing_freq_score(row["rebalancing_freq"])
        fixed_commission = 10 - self._normalized_etfs["fixed_commission"][row.name]
        focus_type = self._focus_type_score(row["focus_type"])
        num_shares = self._normalized_etfs["num_shares"][row.name]

        return (
            0.25 * rebalancing_freq
            + 0.25 * fixed_commission
            + 0.3 * focus_type
            + 0.2 * num_shares
        )

    def calculate_bonds_rating(self, row: pd.Series) -> float:
        coupon_quantity = self._normalized_bonds["coupon_quantity_per_year"][row.name]
        maturity_date = 10 - self._normalized_bonds["maturity_date"][row.name]
        country_risk = self._country_risk_score(row["country_of_risk"])
        issue_size = self._normalized_bonds["issue_size"][row.name]

        return (
            0.25 * coupon_quantity
            + 0.2 * maturity_date
            + 0.25 * country_risk
            + 0.2 * issue_size
        )
