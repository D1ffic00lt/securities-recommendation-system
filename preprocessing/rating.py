import pandas as pd
import numpy as np

from datetime import datetime
from typing import Any, Iterable
from matplotlib.cbook import boxplot_stats

from preprocessing.transformers import make_empty_values_filler_pipeline


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
    def __init__(
        self,
        data: pd.DataFrame,
        keys: Iterable[str],
        *args,
        process_outliers: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if data.shape[0] == 0:
            return

        for key in keys:
            match key:
                case "ipo_date":
                    dict.__setitem__(
                        self,
                        key,
                        self.normalize(
                            (datetime.now() - self.to_datetime(data[key])).apply(
                                lambda x: x.days
                            )
                        ),
                    )
                case "maturity_date":
                    dict.__setitem__(
                        self,
                        key,
                        self.normalize(
                            (self.to_datetime(data[key]) - datetime.now()).apply(
                                lambda x: x.days
                            )
                        ),
                    )
                case _:
                    if not process_outliers:
                        dict.__setitem__(self, key, self.normalize(data[key]))
                    outliers_indexes = self._get_outliers_indexes(data[key])
                    column = data[key].copy()
                    column[outliers_indexes] = 10
                    column[~outliers_indexes] = self.normalize(data[key])
                    dict.__setitem__(self, key, column)

    def __setitem__(self, key, value):
        raise KeyError("You can't change the attributes.")

    @staticmethod
    def to_datetime(value: pd.Series) -> pd.Series:
        return pd.to_datetime(value, format="%d/%m/%Y")

    @staticmethod
    def normalize(series: pd.Series) -> pd.Series:
        return 10 * (series - series.min()) / (series.max() - series.min())

    @staticmethod
    def _get_outliers_indexes(data: pd.Series) -> pd.Series:
        outliers = boxplot_stats(data).pop(0)["fliers"]
        return data.isin(outliers)


class SecurityRating(object):
    def __init__(
        self,
        shares: pd.DataFrame = None,
        bonds: pd.DataFrame = None,
        etfs: pd.DataFrame = None,
        *,
        remove_nan_values: bool = True,
        process_outliers: bool = True,
        num_nan_value: Any = 0,
        cat_nan_value: str = "unknown",
    ) -> None:
        if all([shares is None, bonds is None, etfs is None]):
            raise ValueError("At least one DataFrame must be non-empty.")

        shares = shares.copy() if shares is not None else pd.DataFrame()
        bonds = bonds.copy() if bonds is not None else pd.DataFrame()
        etfs = etfs.copy() if etfs is not None else pd.DataFrame()

        shares_cat_columns = shares.dtypes[shares.dtypes == "object"].index.tolist()
        shares_num_columns = shares.dtypes[shares.dtypes != "object"].index.tolist()

        etfs_cat_columns = etfs.dtypes[etfs.dtypes == "object"].index.tolist()
        etfs_num_columns = etfs.dtypes[etfs.dtypes != "object"].index.tolist()

        bonds_cat_columns = bonds.dtypes[bonds.dtypes == "object"].index.tolist()
        bonds_num_columns = bonds.dtypes[bonds.dtypes != "object"].index.tolist()

        self._shares_pipeline = make_empty_values_filler_pipeline(
            num_columns=shares_num_columns,
            cat_columns=shares_cat_columns,
            num_nan_value=num_nan_value,
            cat_nan_value=cat_nan_value,
        )
        self._bonds_pipeline = make_empty_values_filler_pipeline(
            num_columns=bonds_num_columns,
            cat_columns=bonds_cat_columns,
            num_nan_value=num_nan_value,
            cat_nan_value=cat_nan_value,
        )
        self._etfs_pipeline = make_empty_values_filler_pipeline(
            num_columns=etfs_num_columns,
            cat_columns=etfs_cat_columns,
            num_nan_value=num_nan_value,
            cat_nan_value=cat_nan_value,
        )

        if remove_nan_values:
            shares = pd.DataFrame(
                self._shares_pipeline.fit_transform(shares),
                columns=(shares_num_columns + shares_cat_columns),
            )
            bonds = pd.DataFrame(
                self._bonds_pipeline.fit_transform(bonds),
                columns=(bonds_num_columns + bonds_cat_columns),
            )
            etfs = pd.DataFrame(
                self._etfs_pipeline.fit_transform(etfs),
                columns=(etfs_num_columns + etfs_cat_columns),
            )

        if shares.shape[0] != 0 and shares["ipo_date"].dtype.kind == "O":
            shares["ipo_date"] = pd.to_datetime(shares["ipo_date"], format="%d/%m/%Y")
        if bonds.shape[0] != 0 and bonds["maturity_date"].dtype.kind == "O":
            bonds["maturity_date"] = pd.to_datetime(
                bonds["maturity_date"], format="%d/%m/%Y"
            )
        self._normalized_shares = NormalizedData(
            data=shares,
            keys=("issue_size", "ipo_date"),
            process_outliers=process_outliers,
        )
        self._normalized_etfs = NormalizedData(
            data=etfs,
            keys=("fixed_commission", "num_shares"),
            process_outliers=process_outliers,
        )
        self._normalized_bonds = NormalizedData(
            data=bonds,
            keys=("coupon_quantity_per_year", "maturity_date", "issue_size"),
            process_outliers=process_outliers,
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
        if len(self._normalized_shares.keys()) == 0:
            raise ValueError("No shares were calculated.")

        liquidity = 10 if row["liquidity_flag"] else 0
        country_risk = self._country_risk_score(row["country_of_risk"])
        issue_size = self._normalized_shares["issue_size"][row.name]
        ipo_days = 10 - self._normalized_shares["ipo_date"][row.name]

        return 0.3 * liquidity + 0.2 * country_risk + 0.3 * issue_size + 0.2 * ipo_days

    def calculate_etfs_rating(self, row: pd.Series) -> float:
        if len(self._normalized_etfs.keys()) == 0:
            raise ValueError("No etfs were calculated.")

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
        if len(self._normalized_bonds.keys()) == 0:
            raise ValueError("No bonds were calculated.")

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
