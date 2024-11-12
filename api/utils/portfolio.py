import numpy as np
import pandas as pd

from ortools.algorithms.python import knapsack_solver
from ortools.algorithms.python.knapsack_solver import KnapsackSolver

from .storage import SecurityVault


class Security(object):
    def __init__(
        self,
        figi: str,
        price: int | float,
        price_rating: float,
        company_rating: float,
        sector: str,
        lot: int,
    ):
        """
        Initialize the Security class.

        Args:
            figi (str): Financial Instrument Global Identifier.
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
        self.lot = lot

    @property
    def json(self):
        return {
            "figi": str(self.figi),
            "price": float(self.price),
            "sector": str(self.sector),
            "lot": int(self.lot),
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

    def recommend(self, portfolio: Portfolio, capacity: int):
        """
        Recommend securities to add to the portfolio based on available capacity.

        Args:
            portfolio (Portfolio): The user's portfolio to which securities are to be added.
            capacity (int): The total capacity to be used for adding new securities.
        """
        free_capacity = capacity % 3
        etfs_capacity = capacity // 3
        bonds_capacity = capacity // 3
        shares_capacity = capacity // 3

        bonds_sectors_weights = self.storage.bonds.sector.value_counts(normalize=True)
        bonds_sectors_weights *= bonds_capacity
        bonds_sectors_weights = bonds_sectors_weights.apply(np.ceil)

        free_capacity += bonds_capacity - bonds_sectors_weights.sum()

        shares_sectors_weights = self.storage.shares.sector.value_counts(normalize=True)
        shares_sectors_weights *= shares_capacity
        shares_sectors_weights = shares_sectors_weights.apply(np.ceil)

        free_capacity += shares_capacity - shares_sectors_weights.sum()

        etfs_sectors_weights = self.storage.etfs.sector.value_counts(normalize=True)

        etfs_sectors_weights *= etfs_capacity
        etfs_sectors_weights = etfs_sectors_weights.apply(np.ceil)

        free_capacity += etfs_capacity - etfs_sectors_weights.sum()

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
            self.storage.bonds, portfolio, bonds_solvers
        )
        money_spent_for_shares = self._validate_solvers(
            self.storage.shares, portfolio, shares_solvers
        )
        money_spent_for_etfs = self._validate_solvers(
            self.storage.etfs, portfolio, etfs_solvers
        )

        free_capacity += (
            (bonds_capacity - money_spent_for_bonds)
            + (shares_capacity - money_spent_for_shares)
            + (etfs_capacity - money_spent_for_etfs)
        )

        infinity_portfolio = self.storage.etfs.loc[
            self.storage.etfs.figi == "BBG000000001"
        ].iloc[0]
        infinity_portfolio.rub_price = np.ceil(infinity_portfolio.rub_price)

        number_of_etfs = free_capacity // int(infinity_portfolio.rub_price)
        infinity_portfolio.lot = infinity_portfolio.lot * number_of_etfs
        infinity_portfolio.rub_price = infinity_portfolio.rub_price * number_of_etfs

        portfolio.append(
            Security(
                figi=infinity_portfolio.figi,
                price=infinity_portfolio.rub_price,
                price_rating=infinity_portfolio.price_rating,
                company_rating=infinity_portfolio.company_rating,
                sector=infinity_portfolio.sector,
                lot=infinity_portfolio.lot,
            )
        )

    @staticmethod
    def _validate_solvers(
        data: pd.DataFrame, portfolio: Portfolio, solvers: dict[str, KnapsackSolver]
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
            "rub_price",
        ]
        original_data = data.copy()
        original_data = original_data[valid_columns].dropna()
        original_data.rub_price = original_data.rub_price.apply(np.ceil).astype(int)
        original_data["ratings"] = (
            original_data.price_rating + original_data.company_rating
        ).apply(lambda x: int(x * 10))
        original_data.ratings = original_data.ratings.apply(np.ceil).astype(int)

        money_spent = 0
        for sector, solver in solvers.items():
            solver.solve()
            sector_data = original_data.loc[original_data.sector == sector]
            for i in range(sector_data.shape[0]):
                if solver.best_solution_contains(i):
                    value = sector_data.iloc[i]
                    money_spent += int(value.rub_price)
                    portfolio.append(
                        Security(
                            figi=value.figi,
                            price=value.rub_price,
                            price_rating=value.price_rating,
                            company_rating=value.company_rating,
                            sector=value.sector,
                            lot=value.lot,
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
            "rub_price",
        ]
        solvers = {}

        original_data = data.copy()
        # breakpoint()
        original_data = original_data[valid_columns].dropna()
        original_data.rub_price = original_data.rub_price.apply(np.ceil).astype(int)
        original_data["ratings"] = (
            original_data.price_rating + original_data.company_rating
        ).apply(lambda x: int(x * 10))
        original_data.ratings = original_data.ratings.apply(np.ceil).astype(int)

        for sector, weight in sectors_weights.items():
            solver = knapsack_solver.KnapsackSolver(
                knapsack_solver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
                f"Knapsack{sector}{np.random.randint(0, 1000)}",
            )
            sector_data = original_data.loc[original_data.sector == sector]
            solver.init(
                sector_data.ratings.tolist(),
                [sector_data.rub_price.tolist()],
                [int(weight)],
            )
            solvers[str(sector)] = solver
        return solvers
