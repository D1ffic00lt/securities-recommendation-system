from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_dataframe(
    data: np.ndarray, columns: list[str] = None
) -> pd.DataFrame | np.ndarray:
    if columns is None:
        return pd.DataFrame(data)
    return pd.DataFrame(data, columns=columns)


def convert_price(
    prices: pd.Series, units: pd.Series, currencies: pd.DataFrame
) -> pd.Series:
    if set(units.unique().tolist()) == {"rub"}:
        return prices

    currencies_prices = {
        row.iso_currency_name: (row.price / row.nominal)
        for _, row in currencies.iterrows()
    }
    units_diff = set(units.unique().tolist()) - set(currencies_prices.keys())
    if len(units_diff) != 0:
        raise ValueError(f"Some currency is not supported. ({units_diff})")
    return prices * units.map(lambda x: currencies_prices[x])


def make_empty_values_filler_pipeline(
    num_columns: list[str],
    cat_columns: list[str],
    *,
    num_nan_value: Any = 0,
    cat_nan_value: str = "unknown",
) -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(fill_value=num_nan_value), num_columns),
            (
                "cat",
                SimpleImputer(fill_value=cat_nan_value, strategy="most_frequent"),
                cat_columns,
            ),
        ]
    )
    return preprocessor


def make_pipeline(
    num_columns: list[str],
    cat_columns: list[str],
    model: Callable,
    *,
    num_nan_value: Any = 0,
    cat_nan_value: str = "unknown",
) -> Pipeline:
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(fill_value=num_nan_value)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(fill_value=cat_nan_value, strategy="most_frequent"),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, num_columns),
            ("cat", categorical_transformer, cat_columns),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("cluster", model)])
    return pipeline
