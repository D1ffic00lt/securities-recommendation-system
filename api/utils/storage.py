import os
import pandas as pd

from typing import Union

from parser.utils.api import APIParser
from preprocessing.rating import SecurityRating


class SecurityVault(object):
    def __init__(self, token: str = os.environ.get("TINKOFF_TOKEN", None)):
        """
        Initialize the SecurityVault class.

        Args:
            token (Optional[str]): Authentication token for accessing Tinkoff API. Defaults to value from environment variable "TINKOFF_TOKEN".
        """
        self.bonds: Union[None, pd.DataFrame] = None
        self.shares: Union[None, pd.DataFrame] = None
        self.etfs: Union[None, pd.DataFrame] = None
        self.currencies: Union[None, pd.DataFrame] = None

        self._token = token
        self._parser = APIParser(token)
        self._evaluator = None

        self._parser.get_currencies_exchange_rates(use_tqdm=True)

        checks = self.cache_check()
        if not checks["folder"]:
            os.mkdir("./.cache")

        for i in ["shares", "etfs", "bonds", "currencies"]:
            if os.path.exists(f"./.cache/{i}.csv"):
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
            "./.cache/bonds.csv",
            include_price=True,
            use_tqdm=True,
            convert_to_rubles=True,
            skip_unknown=True,
        )
        self.build_evaluator()
        self.bonds["rating"] = self.bonds.apply(
            self._evaluator.calculate_bonds_rating, axis=1
        )
        self.bonds.to_csv("./.cache/bonds.csv", index=False)

    def build_shares(self):
        """
        Build shares DataFrame by parsing and saving shares data to cache.
        """
        shares = self._parser.parse_shares()
        self.shares = self._parser.write(
            shares,
            "./.cache/shares.csv",
            include_price=True,
            use_tqdm=True,
            convert_to_rubles=True,
            skip_unknown=True,
        )
        self.build_evaluator()
        self.shares["rating"] = self.shares.apply(
            self._evaluator.calculate_shares_rating, axis=1
        )
        self.shares.to_csv("./.cache/shares.csv", index=False)

    def build_etfs(self):
        """
        Build ETFs DataFrame by parsing and saving ETFs data to cache.
        """
        etfs = self._parser.parse_etfs()
        self.etfs = self._parser.write(
            etfs,
            "./.cache/etfs.csv",
            include_price=True,
            use_tqdm=True,
            convert_to_rubles=True,
            skip_unknown=True,
        )
        self.build_evaluator()
        self.etfs["rating"] = self.etfs.apply(
            self._evaluator.calculate_etfs_rating, axis=1
        )
        self.etfs.to_csv("./.cache/etfs.csv", index=False)

    def build_currencies(self):
        """
        Build currencies DataFrame by parsing and saving currencies data to cache.
        """
        currencies = self._parser.parse_currencies()
        self._parser.write(
            currencies,
            "./.cache/currencies.csv",
            include_price=True,
            use_tqdm=True,
            convert_to_rubles=True,
            skip_unknown=True,
            to_csv=True,
        )

    def price_ranking(self, *, use_tqdm: bool = True, retry_if_limit: bool = True):
        """
        Calculate and update price ranking for bonds, shares, and ETFs.

        Args:
            use_tqdm (bool): Whether to use tqdm progress bar. Defaults to True.
            retry_if_limit (bool): Whether to retry if the API rate limit is reached. Defaults to True.
        """
        rating_columns = {"rating_x": "company_rating", "rating_y": "price_rating"}
        bonds_price_ratings = self._evaluator.calculate_price_rating(
            self._parser,
            self.bonds.figi,
            use_tqdm=use_tqdm,
            retry_if_limit=retry_if_limit,
        )
        self.bonds = self.bonds.merge(bonds_price_ratings, on="figi", how="inner")
        self.bonds = self.bonds.rename(rating_columns, axis=1)
        self.bonds = self.bonds.loc[self.bonds.rub_price > 0]

        shares_price_ratings = self._evaluator.calculate_price_rating(
            self._parser,
            self.shares.figi,
            use_tqdm=use_tqdm,
            retry_if_limit=retry_if_limit,
        )
        self.shares = self.shares.merge(shares_price_ratings, on="figi", how="inner")
        self.shares = self.shares.rename(rating_columns, axis=1)
        self.shares = self.shares[self.shares.rub_price > 0]

        etfs_price_ratings = self._evaluator.calculate_price_rating(
            self._parser,
            self.etfs.figi,
            use_tqdm=use_tqdm,
            retry_if_limit=retry_if_limit,
        )
        self.etfs = self.etfs.merge(etfs_price_ratings, on="figi", how="inner")
        self.etfs = self.etfs.rename(rating_columns, axis=1)
        self.etfs = self.etfs.loc[self.etfs.rub_price > 0]

        self.bonds.to_csv("./.cache/bonds.csv", index=False)
        self.shares.to_csv("./.cache/shares.csv", index=False)
        self.etfs.to_csv("./.cache/etfs.csv", index=False)

    def build(self, use_tqdm: bool = True, retry_if_limit: bool = True):
        """
        Build all securities (currencies, bonds, shares, and ETFs) and calculate price ranking.

        Args:
            use_tqdm (bool): Whether to use tqdm progress bar. Defaults to True.
            retry_if_limit (bool): Whether to retry if the API rate limit is reached. Defaults to True.
        """
        with self._parser:
            self.build_currencies()
            self.build_bonds()
            self.build_shares()
            self.build_etfs()
            self.price_ranking(use_tqdm=use_tqdm, retry_if_limit=retry_if_limit)

    @staticmethod
    def cache_check():
        """
        Check if cache files for securities exist.

        Returns:
            Dict[str, bool]: A dictionary indicating the existence of cache files for each security type.
        """
        return {
            "folder": os.path.exists("./.cache"),
            "bonds": os.path.exists("./.cache/bonds.csv"),
            "etfs": os.path.exists("./.cache/etfs.csv"),
            "currencies": os.path.exists("./.cache/currencies.csv"),
            "shares": os.path.exists("./.cache/shares.csv"),
        }

    def _load(self, name):
        """
        Load cached data for a given security type.

        Args:
            name (str): The name of the security type to load (e.g., "shares", "etfs", "bonds", "currencies").
        """
        match name:
            case "shares":
                self.shares = pd.read_csv(f"./.cache/{name}.csv")
            case "etfs":
                self.etfs = pd.read_csv(f"./.cache/{name}.csv")
            case "bonds":
                self.bonds = pd.read_csv(f"./.cache/{name}.csv")
            case "currencies":
                self.currencies = pd.read_csv(f"./.cache/{name}.csv")
            case _:
                raise ValueError(f"Invalid name: {name}")
