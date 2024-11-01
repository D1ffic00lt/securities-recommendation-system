import os
import warnings
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Any, Union
from tqdm.auto import tqdm
from tinkoff.invest import Client, GetLastPricesResponse, InstrumentStatus
from tinkoff.invest.schemas import (
    BondsResponse,
    CurrenciesResponse,
    EtfsResponse,
    RiskLevel,
    SharesResponse,
)
from tinkoff.invest.services import Services
from functools import singledispatchmethod, wraps
from collections import defaultdict

from ._types import *
from ._properties import *

__all__ = ("APIParser",)


class APIParser(object):
    def __init__(self, token: str = os.environ.get("TINKOFF_TOKEN", None)):
        """
        Initializes the APIParser with a Tinkoff API token.

        Args:
            token (str): The API token. If not provided, it is taken from the 'TINKOFF_TOKEN' environment variable.

        Raises:
            ValueError: If no token is provided.
        """
        if token is None:
            raise ValueError("Tinkoff token must be provided")
        self._token: str = token
        self._client: Client | None = None
        self._channel: Services | None = None
        self.figis_prices = defaultdict(int)
        self.currencies_prices = {"rub": 1.0}

    def __enter__(self) -> "APIParser":
        """
        Enters the APIParser context, initializing the Tinkoff client services channel.

        Returns:
            APIParser: The instance of the APIParser with an active client connection.
        """
        self._client = Client(self._token)
        # it's interesting that in Tinkoff library there is no possibility
        # to open 2 connections (not parallel) from one client,
        # I don't know if it's a bug, but I spent at least 2 days to solve this problem.
        # --------------------------------
        # client = Client(...)
        # with client as channel:
        #     ...  # some channel work
        # with client as channel:
        #     ...  # some channel work
        # --------------------------------
        # just try running this, and it will give ValueError: Cannot invoke RPC on closed channel!
        self._channel = self._client.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exits the APIParser context, closing the Tinkoff client connection and releasing resources.
        """
        self._client.__exit__(exc_type, exc_val, exc_tb)
        self._channel = None

    @staticmethod
    def connection(func: Any) -> Any:
        """
        Decorator to ensure that a connection to the Tinkoff API is open before calling the wrapped method.
        If the connection is not open, the context manager opens it temporarily.

        Args:
            func: The method to wrap.

        Returns:
            Any: The wrapped function result.
        """

        @wraps(func)
        def wrapper(self: "APIParser", *args: Any, **kwargs: Any) -> Any:
            if getattr(self, "_channel", None) is not None:
                return func(self, *args, **kwargs)
            with self:
                return func(self, *args, **kwargs)

        return wrapper

    @connection
    def parse_shares(self) -> SharesResponse:
        """
        Fetches a list of shares from the Tinkoff API.

        Returns:
            SharesResponse: A response containing information about available shares.
        """
        return self._channel.instruments.shares(
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
        )

    @connection
    def parse_bonds(self) -> BondsResponse:
        """
        Fetches a list of bonds from the Tinkoff API.

        Returns:
            BondsResponse: A response containing information about available bonds.
        """
        return self._channel.instruments.bonds(
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
        )

    @connection
    def parse_etfs(self) -> EtfsResponse:
        """
        Fetches a list of ETFs from the Tinkoff API.

        Returns:
            EtfsResponse: A response containing information about available ETFs.
        """
        return self._channel.instruments.etfs(
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
        )

    @connection
    def parse_currencies(self) -> CurrenciesResponse:
        """
        Fetches a list of currencies from the Tinkoff API.

        Returns:
            CurrenciesResponse: A response containing information about available currencies.
        """
        return self._channel.instruments.currencies(
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
        )

    @connection
    def exchange_rate_parsing(self, figis: list[str]) -> list[float]:
        """
        Retrieves the latest exchange rates for a list of instrument FIGIs.

        Args:
            figis (list[str]): A list of FIGI strings representing financial instruments.

        Returns:
            list[float]: The latest exchange rates for a list of instrument FIGIs.
        """
        response: GetLastPricesResponse = self._channel.market_data.get_last_prices(
            figi=figis
        )

        prices = list(
            map(lambda x: self._validate_value(x.price), response.last_prices)
        )
        figis = dict(zip(figis, prices))

        self.figis_prices.update(figis)

        return prices

    @singledispatchmethod
    def write(
        self,
        data: Any,
        filename: str,
        *,
        use_tqdm: bool = False,
        include_price: bool = False,
        convert_to_rubles: bool = False,
        skip_unknown: bool = False,
        unknown_value: Any = np.nan,
    ) -> None:
        """
        Writes data from a Tinkoff API response to a CSV file. The method is dispatched based on data type.

        Args:
            data: The Tinkoff API response data.
            filename (str): The name of the output CSV file.
            use_tqdm (bool): Whether to display a progress bar using tqdm.
            include_price (bool): Whether to include the price information in the CSV file.
            convert_to_rubles (bool): Whether to convert the price information to rubles.
            skip_unknown (bool): Whether to skip unknown currencies.
            unknown_value (Any): The value to use for unknown currencies.
        Raises:
            ValueError: If the data type is invalid.
        """
        raise ValueError("Invalid data type. (%s)" % data.__class__.__name__)

    @write.register
    def _(
        self,
        data: SharesResponse,
        filename: str,
        *,
        use_tqdm: bool = False,
        include_price: bool = False,
        convert_to_rubles: bool = False,
        skip_unknown: bool = False,
        unknown_value: Any = np.nan,
    ) -> None:
        """
        Writes shares data to a CSV file.

        Args:
            data (SharesResponse): The response data for shares.
            filename (str): The name of the CSV file to write.
            use_tqdm (bool): Whether to display a progress bar.
            include_price (bool): Whether to include the price information in the CSV file.
            skip_unknown (bool): Whether to skip unknown currencies.
            unknown_value (Any): The value to use for unknown currencies.
        """
        self._generate_csv(
            columns=ResponseColumns.SHARES.value,
            data=data,
            filename=filename,
            use_tqdm=use_tqdm,
            include_price=include_price,
            convert_to_rubles=convert_to_rubles,
            skip_unknown=skip_unknown,
            unknown_value=unknown_value,
        )

    @write.register
    def _(
        self,
        data: BondsResponse,
        filename: str,
        *,
        use_tqdm: bool = False,
        include_price: bool = False,
        convert_to_rubles: bool = False,
        skip_unknown: bool = False,
        unknown_value: Any = np.nan,
    ) -> None:
        """
        Writes bonds data to a CSV file.

        Args:
            data (BondsResponse): The response data for bonds.
            filename (str): The name of the CSV file to write.
            use_tqdm (bool): Whether to display a progress bar.
            include_price (bool): Whether to include the price information in the CSV file.
            skip_unknown (bool): Whether to skip unknown currencies.
            unknown_value (Any): The value to use for unknown currencies.
        """
        self._generate_csv(
            columns=ResponseColumns.BONDS.value,
            data=data,
            filename=filename,
            use_tqdm=use_tqdm,
            include_price=include_price,
            convert_to_rubles=convert_to_rubles,
            skip_unknown=skip_unknown,
            unknown_value=unknown_value,
        )

    @write.register
    def _(
        self,
        data: EtfsResponse,
        filename: str,
        *,
        use_tqdm: bool = False,
        include_price: bool = False,
        convert_to_rubles: bool = False,
        skip_unknown: bool = False,
        unknown_value: Any = np.nan,
    ) -> None:
        """
        Writes ETF data to a CSV file.

        Args:
            data (EtfsResponse): The response data for ETFs.
            filename (str): The name of the CSV file to write.
            use_tqdm (bool): Whether to display a progress bar.
            include_price (bool): Whether to include the price information in the CSV file.
            skip_unknown (bool): Whether to skip unknown currencies.
            unknown_value (Any): The value to use for unknown currencies.
        """
        self._generate_csv(
            columns=ResponseColumns.ETFS.value,
            data=data,
            filename=filename,
            use_tqdm=use_tqdm,
            include_price=include_price,
            convert_to_rubles=convert_to_rubles,
            skip_unknown=skip_unknown,
            unknown_value=unknown_value,
        )

    @write.register
    def _(
        self,
        data: CurrenciesResponse,
        filename: str,
        *,
        use_tqdm: bool = False,
        include_price: bool = False,
        convert_to_rubles: bool = False,
        skip_unknown: bool = False,
        unknown_value: Any = np.nan,
    ) -> None:
        """
        Writes currencies data to a CSV file.

        Args:
            data (CurrenciesResponse): The response data for currencies.
            filename (str): The name of the CSV file to write.
            use_tqdm (bool): Whether to display a progress bar.
            skip_unknown (bool): Whether to skip unknown currencies.
            unknown_value (Any): The value to use for unknown currencies.
        """
        self._generate_csv(
            columns=ResponseColumns.CURRENCIES.value,
            data=data,
            filename=filename,
            use_tqdm=use_tqdm,
            include_price=include_price,
            convert_to_rubles=convert_to_rubles,
            skip_unknown=skip_unknown,
            unknown_value=unknown_value,
        )

    def _generate_csv(
        self,
        columns: list[str],
        data: _response_types,
        filename: str,
        use_tqdm: bool = False,
        include_price: bool = False,
        convert_to_rubles: bool = False,
        skip_unknown: bool = False,
        unknown_value: Any = np.nan
    ) -> None:
        """
        Generates a CSV file from the Tinkoff API data response.

        Args:
            columns (list[str]): List of columns to include in the output.
            data (_response_types): The Tinkoff API data to write.
            filename (str): The output file name.
            use_tqdm (bool): Whether to use a progress bar during the data generation.
            include_price (bool): Whether to include the price in the output dataframe.
            skip_unknown (bool): Whether to skip unknown currencies.
            unknown_value (Any): The value to use for unknown currencies.
        """
        iterator = tqdm(data.instruments) if use_tqdm else data.instruments
        data = [
            {attr: self._validate_value(getattr(row, attr)) for attr in columns}
            for row in iterator
        ]
        dataframe = pd.DataFrame(data, columns=columns)
        if include_price:
            dataframe["price"] = self.exchange_rate_parsing(
                figis=dataframe["figi"].tolist()
            )
            if convert_to_rubles:
                if len(self.currencies_prices) == 1:
                    warnings.warn(
                        "Currencies data is only available for one currency. "
                        "Use get_currencies_exchange_rates before calling write"
                    )
                units_diff = set(dataframe.currency.unique().tolist()) - set(
                    self.currencies_prices.keys()
                )
                if len(units_diff) != 0 and not skip_unknown:
                    raise ValueError(f"Some currency is not supported. ({units_diff})")
                currencies_prices = self.currencies_prices.copy()
                if skip_unknown:
                    currencies_prices = defaultdict(lambda: unknown_value, currencies_prices)
                dataframe["rub_price"] = dataframe.price * dataframe.currency.map(
                    lambda x: currencies_prices[x]
                )
        dataframe.to_csv(filename, index=False)

    @connection
    def get_currencies_exchange_rates(
        self, *, use_tqdm: bool = False
    ) -> dict[str, float]:
        currencies = self.parse_currencies()
        iterator = tqdm(currencies.instruments) if use_tqdm else currencies.instruments
        data = [
            {
                attr: self._validate_value(getattr(row, attr))
                for attr in ResponseColumns.CURRENCIES.value
            }
            for row in iterator
        ]
        currencies = pd.DataFrame(data, columns=ResponseColumns.CURRENCIES.value)
        currencies["price"] = self.exchange_rate_parsing(
            figis=currencies["figi"].tolist()
        )
        self.currencies_prices = {
            row.iso_currency_name: (row.price / row.nominal)
            for _, row in currencies.iterrows()
        }
        self.currencies_prices["rub"] = 1
        return self.currencies_prices

    @staticmethod
    def _validate_value(value: Any) -> Union[float, str, int]:
        """
        Converts Tinkoff API values to standard data types.

        Args:
            value (Any): The value to validate and convert.

        Returns:
            Union[float, str, int]: The validated value, converted if needed.
        """
        match value:
            case x if isinstance(x, _money_types):
                return value.units + (value.nano / 1e9)
            case datetime():
                return value.strftime("%d/%m/%Y")
            case RiskLevel():
                return value.value
            case _:
                return value

    def __repr__(self):
        return "APIParser({})".format(getattr(self, "_channel", None) is not None)
