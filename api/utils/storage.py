import os
import pandas as pd

from datetime import datetime, timedelta
from typing import Union
from tinkoff.invest.utils import now

from parser.utils.api import APIParser
from preprocessing.rating import SecurityRating


class SecurityVault(object):
    def __init__(
        self,
        token: str = os.environ.get("TINKOFF_TOKEN", None),
        *,
        cache_path: str = "./.cache",
    ):
        """
        Initialize the SecurityVault class.

        Args:
            token (Optional[str]): Authentication token for accessing Tinkoff API. Defaults to value from environment variable "TINKOFF_TOKEN".
        """
        self.bonds: Union[None, pd.DataFrame] = None
        self.shares: Union[None, pd.DataFrame] = None
        self.etfs: Union[None, pd.DataFrame] = None
        self.currencies: Union[None, pd.DataFrame] = None

        self._cache_path = cache_path
        self._token = token
        self._parser = APIParser(token)
        self._evaluator = None

        self._parser.get_currencies_exchange_rates(use_tqdm=True)

        checks = self.cache_check()
        if not checks["folder"]:
            os.mkdir(self._cache_path)

        for i in ["shares", "etfs", "bonds", "currencies"]:
            if os.path.exists(os.path.join(self._cache_path, f"{i}.csv")):
                self._load(i)


    def build_evaluator(self, **kwargs) -> SecurityRating:
        """
        Build and initialize the SecurityRating evaluator.

        Args:
            **kwargs: Additional keyword arguments to configure the SecurityRating instance.

        Returns:
            SecurityRating: The initialized SecurityRating evaluator.
        """
        self._evaluator = SecurityRating(
            shares=self.shares,
            bonds=self.bonds,
            etfs=self.etfs,
            **kwargs,
        )
        return self._evaluator

    def build_bonds(self):
        """
        Build bonds DataFrame by parsing and saving bonds data to cache.
        """
        bonds = self._parser.parse_bonds()
        self.bonds = self._parser.write(
            bonds,
            os.path.join(self._cache_path, "bonds.csv"),
            include_price=True,
            use_tqdm=True,
            convert_to_rubles=True,
            skip_unknown=True,
        )
        self.build_evaluator()
        self.bonds["rating"] = self.bonds.apply(
            self._evaluator.calculate_bonds_rating, axis=1
        )
        self.bonds.to_csv(os.path.join(self._cache_path, "bonds.csv"), index=False)

    def build_shares(self):
        """
        Build shares DataFrame by parsing and saving shares data to cache.
        """
        shares = self._parser.parse_shares()
        self.shares = self._parser.write(
            shares,
            os.path.join(self._cache_path, "shares.csv"),
            include_price=True,
            use_tqdm=True,
            convert_to_rubles=True,
            skip_unknown=True,
        )
        self.build_evaluator()
        self.shares["rating"] = self.shares.apply(
            self._evaluator.calculate_shares_rating, axis=1
        )
        self.shares.to_csv(os.path.join(self._cache_path, "shares.csv"), index=False)

    def build_etfs(self):
        """
        Build ETFs DataFrame by parsing and saving ETFs data to cache.
        """
        etfs = self._parser.parse_etfs()
        self.etfs = self._parser.write(
            etfs,
            os.path.join(self._cache_path, "etfs.csv"),
            include_price=True,
            use_tqdm=True,
            convert_to_rubles=True,
            skip_unknown=True,
        )
        self.build_evaluator()
        self.etfs["rating"] = self.etfs.apply(
            self._evaluator.calculate_etfs_rating, axis=1
        )
        self.etfs.to_csv(os.path.join(self._cache_path, "etfs.csv"), index=False)

    def build_currencies(self):
        """
        Build currencies DataFrame by parsing and saving currencies data to cache.
        """
        currencies = self._parser.parse_currencies()
        self._parser.write(
            currencies,
            os.path.join(self._cache_path, "currencies.csv"),
            include_price=True,
            use_tqdm=True,
            convert_to_rubles=True,
            skip_unknown=True,
            to_csv=True,
        )

    def price_ranking(
        self,
        *,
        use_tqdm: bool = True,
        retry_if_limit: bool = True,
        from_date: datetime = now() - timedelta(days=365),
        to_date: datetime = now(),
    ):
        """
        Calculate and update price ranking for bonds, shares, and ETFs.

        Args:
            use_tqdm (bool): Whether to use tqdm progress bar. Defaults to True.
            retry_if_limit (bool): Whether to retry if the API rate limit is reached. Defaults to True.
            from_date (datetime): From date to calculate price ranking.
            to_date (datetime): To date to calculate price ranking.
        """
        rating_columns = {"rating_x": "company_rating", "rating_y": "price_rating"}

        etfs_price_ratings = self._evaluator.calculate_price_rating(
            self._parser,
            self.etfs.figi,
            use_tqdm=use_tqdm,
            retry_if_limit=retry_if_limit,
            from_date=from_date,
            to_date=to_date,
        )
        self.etfs = self.etfs.merge(etfs_price_ratings, on="figi", how="inner")
        self.etfs = self.etfs.rename(rating_columns, axis=1)
        self.etfs = self.etfs.loc[self.etfs.rub_price > 0]
        self.etfs["ratings"] = (
            self.etfs.price_rating + self.etfs.company_rating
        ).apply(lambda x: int(x * 100))
        self.etfs["candle_price"] = self.etfs.figi.apply(
            lambda x: self._parser.figis_last_candle_prices[x]
        )
        self.etfs.to_csv(os.path.join(self._cache_path, "etfs.csv"), index=False)

        bonds_price_ratings = self._evaluator.calculate_price_rating(
            self._parser,
            self.bonds.figi,
            use_tqdm=use_tqdm,
            retry_if_limit=retry_if_limit,
            from_date=from_date,
            to_date=to_date,
        )
        self.bonds = self.bonds.merge(bonds_price_ratings, on="figi", how="inner")
        self.bonds = self.bonds.rename(rating_columns, axis=1)
        self.bonds = self.bonds.loc[self.bonds.rub_price > 0]
        self.bonds["ratings"] = (
            self.bonds.price_rating + self.bonds.company_rating
        ).apply(lambda x: int(x * 100))
        self.bonds["candle_price"] = self.bonds.figi.apply(
            lambda x: self._parser.figis_last_candle_prices[x]
        )
        self.bonds.to_csv(os.path.join(self._cache_path, "bonds.csv"), index=False)

        shares_price_ratings = self._evaluator.calculate_price_rating(
            self._parser,
            self.shares.figi,
            use_tqdm=use_tqdm,
            retry_if_limit=retry_if_limit,
            from_date=from_date,
            to_date=to_date,
        )
        self.shares = self.shares.merge(shares_price_ratings, on="figi", how="inner")
        self.shares = self.shares.rename(rating_columns, axis=1)
        self.shares = self.shares[self.shares.rub_price > 0]
        self.shares["ratings"] = (
            self.shares.price_rating + self.shares.company_rating
        ).apply(lambda x: int(x * 100))
        self.shares["candle_price"] = self.shares.figi.apply(
            lambda x: self._parser.figis_last_candle_prices[x]
        )
        self.shares.to_csv(os.path.join(self._cache_path, "shares.csv"), index=False)

    def build(
        self,
        use_tqdm: bool = True,
        retry_if_limit: bool = True,
        from_date: datetime = now() - timedelta(days=365),
        to_date: datetime = now(),
    ):
        """
        Build all securities (currencies, bonds, shares, and ETFs) and calculate price ranking.

        Args:
            use_tqdm (bool): Whether to use tqdm progress bar. Defaults to True.
            retry_if_limit (bool): Whether to retry if the API rate limit is reached. Defaults to True.
            from_date (datetime): From date to calculate price ranking.
            to_date (datetime): To date to calculate price ranking.
        """
        with self._parser:
            self.build_currencies()
            self.build_bonds()
            self.build_shares()
            self.build_etfs()
            self.price_ranking(
                use_tqdm=use_tqdm,
                retry_if_limit=retry_if_limit,
                from_date=from_date,
                to_date=to_date,
            )

    def cache_check(self):
        """
        Check if cache files for securities exist.

        Returns:
            Dict[str, bool]: A dictionary indicating the existence of cache files for each security type.
        """
        return {
            "folder": os.path.exists(self._cache_path),
            "bonds": os.path.exists(os.path.join(self._cache_path, "bonds.csv")),
            "etfs": os.path.exists(os.path.join(self._cache_path, "etfs.csv")),
            "currencies": os.path.exists(
                os.path.join(self._cache_path, "currencies.csv")
            ),
            "shares": os.path.exists(os.path.join(self._cache_path, "shares.csv")),
        }

    def get_price(self, figis: list[str], by: str = "rub_price") -> int:
        current_price = 0
        bonds_figis = self.bonds.figi.unique()
        shares_figis = self.shares.figi.unique()
        etfs_figis = self.etfs.figi.unique()

        for figi in figis:
            if figi in bonds_figis:
                current_price += self.bonds.loc[self.bonds.figi == figi][by].values[0]
            elif figi in shares_figis:
                current_price += self.shares.loc[self.shares.figi == figi][by].values[0]
            elif figi in etfs_figis:
                current_price += self.etfs.loc[self.etfs.figi == figi][by].values[0]
        return current_price

    def _load(self, name):
        """
        Load cached data for a given security type.

        Args:
            name (str): The name of the security type to load (e.g., "shares", "etfs", "bonds", "currencies").
        """
        match name:
            case "shares":
                self.shares = pd.read_csv(f"./{self._cache_path}/{name}.csv")
            case "etfs":
                self.etfs = pd.read_csv(f"./{self._cache_path}/{name}.csv")
            case "bonds":
                self.bonds = pd.read_csv(f"./{self._cache_path}/{name}.csv")
            case "currencies":
                self.currencies = pd.read_csv(f"./{self._cache_path}/{name}.csv")
            case _:
                raise ValueError(f"Invalid name: {name}")
