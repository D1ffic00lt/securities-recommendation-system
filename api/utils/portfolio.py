import numpy as np
import pandas as pd

from ortools.algorithms.python.knapsack_solver import (
    KnapsackSolver,
    KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
)

from .storage import SecurityVault


class Security(object):
    def __init__(
        self,
        figi: str,
        name: str,
        price: int | float,
        price_rating: float,
        company_rating: float,
        final_rating: float,
        sector: str,
        lot: int,
        type_: str,
    ):
        """
        Initialize the Security class.

        Args:
            figi (str): Financial Instrument Global Identifier.
            name (str): Name of the Security.
            price (int | float): Price of the security.
            price_rating (float): Price rating of the security.
            company_rating (float): Company rating of the security.
            sector (str): Sector to which the security belongs.
            lot (int): Number of units in a lot for the security.
        """
        self.figi = figi
        self.price = price
        self.price_rating = price_rating
        self.company_rating = company_rating
        self.sector = sector
        self.final_rating = final_rating
        self.lot = lot
        self.type = type_
        self.name = name

    @property
    def json(self):
        return {
            "figi": str(self.figi),
            "price": float(self.price),
            "sector": str(self.sector),
            "lot": int(self.lot),
            "price_rating": float(self.price_rating),
            "company_rating": float(self.company_rating),
            "name": str(self.name),
            "final_rating": int(self.final_rating),
            "type": str(self.type),
        }

    def __repr__(self):
        """
        Return a string representation of the Security object.

        Returns:
            str: A string containing the FIGI and price of the security.
        """
        return f"<{self.figi}, {self.price:.2f}>"


class Portfolio(list):
    def __init__(self, user_id: int, securities: list[Security] = None):
        """
        Initialize the Portfolio class.

        Args:
            user_id (int): ID of the user who owns the portfolio.
            securities (list[Security]): List of Security objects in the portfolio.
        """
        super().__init__()
        self.user_id = user_id
        if securities is not None:
            self.extend(securities)

    @property
    def json(self):
        return [security.json for security in self]


class RecommendationSystem(object):
    def __init__(self, storage: SecurityVault):
        """
        Initialize the RecommendationSystem class.

        Args:
            storage (SecurityVault): An instance of SecurityVault to access securities data.
        """
        self.storage = storage

        number_of_securities = (
            self.storage.etfs.shape[0]
            + self.storage.bonds.shape[0]
            + self.storage.shares.shape[0]
        )

        self.etfs_capacity_coefficient = (
            self.storage.etfs.shape[0] + self.storage.bonds.shape[0] // 4
        ) / number_of_securities
        self.bonds_capacity_coefficient = (
            self.storage.bonds.shape[0] // 2
        ) / number_of_securities
        self.shares_capacity_coefficient = (
            self.storage.shares.shape[0] + self.storage.bonds.shape[0] // 4
        ) / number_of_securities

    def recommend(self, portfolio: Portfolio, capacity: int):
        """
        Recommend securities to add to the portfolio based on available capacity.

        Args:
            portfolio (Portfolio): The user's portfolio to which securities are to be added.
            capacity (int): The total capacity to be used for adding new securities.
        """
        previous_capacity = 0
        while 0 < capacity != previous_capacity:
            bonds_capacity = np.floor(capacity * self.bonds_capacity_coefficient)
            shares_capacity = np.floor(capacity * self.shares_capacity_coefficient)
            etfs_capacity = np.floor(capacity * self.etfs_capacity_coefficient)

            bonds_sectors_weights = self.storage.bonds.sector.value_counts(
                normalize=True
            )
            bonds_sectors_weights *= bonds_capacity
            bonds_sectors_weights = bonds_sectors_weights.apply(np.ceil)

            shares_sectors_weights = self.storage.shares.sector.value_counts(
                normalize=True
            )
            shares_sectors_weights *= shares_capacity
            shares_sectors_weights = shares_sectors_weights.apply(np.ceil)

            etfs_sectors_weights = self.storage.etfs.sector.value_counts(normalize=True)
            etfs_sectors_weights *= etfs_capacity
            etfs_sectors_weights = etfs_sectors_weights.apply(np.ceil)

            bonds_solvers = self._build_recommendation_systems(
                self.storage.bonds, bonds_sectors_weights
            )
            shares_solvers = self._build_recommendation_systems(
                self.storage.shares, shares_sectors_weights
            )
            etfs_solvers = self._build_recommendation_systems(
                self.storage.etfs, etfs_sectors_weights
            )

            money_spent_for_bonds = self._validate_solvers(
                self.storage.bonds, portfolio, bonds_solvers, type_="bonds"
            )
            money_spent_for_shares = self._validate_solvers(
                self.storage.shares, portfolio, shares_solvers, type_="shares"
            )
            money_spent_for_etfs = self._validate_solvers(
                self.storage.etfs, portfolio, etfs_solvers, type_="etfs"
            )
            previous_capacity = capacity
            capacity -= (
                money_spent_for_etfs + money_spent_for_shares + money_spent_for_bonds
            )

        infinity_portfolio = self.storage.etfs.loc[
            self.storage.etfs.figi == "BBG000000001"
        ].iloc[0]
        infinity_portfolio.candle_price = np.ceil(infinity_portfolio.candle_price)

        free_capacity = capacity // int(infinity_portfolio.candle_price)

        for _ in range(int(free_capacity)):
            portfolio.append(
                Security(
                    figi=infinity_portfolio.figi,
                    price=infinity_portfolio.candle_price,
                    price_rating=infinity_portfolio.price_rating,
                    company_rating=infinity_portfolio.company_rating,
                    sector=infinity_portfolio.sector,
                    lot=infinity_portfolio.lot,
                    name=infinity_portfolio["name"],
                    final_rating=infinity_portfolio.ratings,
                    type_="etfs",
                )
            )

    @staticmethod
    def _validate_solvers(
        data: pd.DataFrame,
        portfolio: Portfolio,
        solvers: dict[str, KnapsackSolver],
        type_: str,
    ) -> int:
        """
        Validate and apply solutions from knapsack solvers to add securities to the portfolio.

        Args:
            data (pd.DataFrame): DataFrame containing securities data.
            portfolio (Portfolio): The user's portfolio to which securities are to be added.
            solvers (dict[str, KnapsackSolver]): A dictionary of knapsack solvers for each sector.

        Returns:
            int: The total amount of money spent for the selected securities.
        """
        valid_columns = [
            "figi",
            "sector",
            "price_rating",
            "company_rating",
            "lot",
            "candle_price",
            "name",
            "ratings",
        ]
        original_data = data.copy()
        original_data = original_data[valid_columns].dropna()
        original_data = original_data.loc[original_data.candle_price > 0]
        original_data.candle_price = original_data.candle_price.apply(np.ceil).astype(
            int
        )
        original_data.ratings = original_data.ratings.apply(np.ceil).astype(int)

        money_spent = 0
        for sector, solver in solvers.items():
            solver.solve()
            sector_data = original_data.loc[original_data.sector == sector]
            for i in range(sector_data.shape[0]):
                if solver.best_solution_contains(i):
                    value = sector_data.iloc[i]
                    money_spent += int(value.candle_price)
                    portfolio.append(
                        Security(
                            figi=value.figi,
                            price=value.candle_price,
                            price_rating=value.price_rating,
                            company_rating=value.company_rating,
                            sector=value.sector,
                            lot=value.lot,
                            name=value["name"],
                            final_rating=value.ratings,
                            type_=type_,
                        )
                    )
        return money_spent

    @staticmethod
    def _build_recommendation_systems(
        data: pd.DataFrame, sectors_weights: pd.Series
    ) -> dict[str, KnapsackSolver]:
        """
        Build knapsack solvers for each sector based on sector weights.

        Args:
            data (pd.DataFrame): DataFrame containing securities data.
            sectors_weights (pd.Series): Series containing sector weights for allocating securities.

        Returns:
            dict[str, KnapsackSolver]: A dictionary of knapsack solvers for each sector.
        """
        valid_columns = [
            "figi",
            "sector",
            "price_rating",
            "company_rating",
            "lot",
            "candle_price",
            "name",
            "ratings",
        ]
        solvers = {}

        original_data = data.copy()
        original_data = original_data[valid_columns].dropna()
        original_data = original_data.loc[original_data.candle_price > 0]
        original_data.candle_price = original_data.candle_price.apply(np.ceil).astype(
            int
        )
        original_data.ratings = original_data.ratings.apply(np.ceil).astype(int)

        for sector, weight in sectors_weights.items():
            solver = KnapsackSolver(
                KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
                f"Knapsack{sector}{np.random.randint(0, 1000)}",
            )
            sector_data = original_data.loc[original_data.sector == sector]
            solver.init(
                sector_data.ratings.tolist(),
                [sector_data.candle_price.tolist()],
                [int(weight)],
            )
            solvers[str(sector)] = solver
        return solvers
