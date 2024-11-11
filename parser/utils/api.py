import os
import warnings
import numpy as np
import pandas as pd
import time

from datetime import datetime, timedelta
from tqdm.auto import tqdm
from typing import Any, Generator, Union
from tinkoff.invest.utils import now
from tinkoff.invest import Client, GetLastPricesResponse, InstrumentStatus, RequestError
from tinkoff.invest.schemas import (
    BondsResponse,
    CandleInterval,
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
        self._client: Union[Client, None] = None
        self._channel: Union[Services, None] = None
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
        if self._client:
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
        to_csv: bool = False,
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
            filename (str): The name of the CSV file to write.
            to_csv (bool, optional): Whether to write the data to a CSV file. Defaults to False.
            use_tqdm (bool, optional): Whether to display a progress bar.
            include_price (bool, optional): Whether to include the price information in the CSV file.
            convert_to_rubles (bool, optional): Whether to convert the price information to rubles.
            skip_unknown (bool, optional): Whether to skip unknown currencies.
            unknown_value (Any, optional): The value to use for unknown currencies.
        Raises:
            ValueError: If the data type is invalid.
        """
        raise ValueError("Invalid data type. (%s)" % data.__class__.__name__)

    @write.register
    def _(self, data: SharesResponse, filename: str, **kwargs) -> pd.DataFrame | None:
        """Writes shares data to a CSV file."""
        return self._generate_csv(
            columns=ResponseColumns.SHARES.value, data=data, filename=filename, **kwargs
        )

    @write.register
    def _(self, data: BondsResponse, filename: str, **kwargs) -> pd.DataFrame | None:
        """Writes bonds data to a CSV file."""
        return self._generate_csv(
            columns=ResponseColumns.BONDS.value, data=data, filename=filename, **kwargs
        )

    @write.register
    def _(self, data: EtfsResponse, filename: str, **kwargs) -> pd.DataFrame | None:
        """Writes ETF data to a CSV file."""
        return self._generate_csv(
            columns=ResponseColumns.ETFS.value, data=data, filename=filename, **kwargs
        )

    @write.register
    def _(
        self, data: CurrenciesResponse, filename: str, **kwargs
    ) -> pd.DataFrame | None:
        """Writes currencies data to a CSV file."""
        return self._generate_csv(
            columns=ResponseColumns.CURRENCIES.value,
            data=data,
            filename=filename,
            **kwargs,
        )

    def _generate_csv(
        self,
        columns: list[str],
        data: _response_types,
        filename: str,
        to_csv: bool = False,
        use_tqdm: bool = False,
        include_price: bool = False,
        convert_to_rubles: bool = False,
        skip_unknown: bool = False,
        unknown_value: Any = np.nan,
    ) -> pd.DataFrame | None:
        """
        Generates a CSV file from the Tinkoff API data response.

        Args:
            columns (list[str]): List of columns to include in the output.
            data (_response_types): The Tinkoff API data to write.
            filename (str): The output file name.
            to_csv (bool, optional): Whether to write the data to a CSV file. Defaults to False.
            use_tqdm (bool, optional): Whether to use a progress bar during the data generation.
            include_price (bool, optional): Whether to include the price in the output dataframe.
            convert_to_rubles (bool, optional): Whether to convert the price information to rubles.
            skip_unknown (bool, optional): Whether to skip unknown currencies.
            unknown_value (Any, optional): The value to use for unknown currencies.
        """
        iterator = (
            tqdm(data.instruments, desc="Creation of a csv")
            if use_tqdm
            else data.instruments
        )
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
                    currencies_prices = defaultdict(
                        lambda: unknown_value, currencies_prices
                    )
                dataframe["rub_price"] = dataframe.price * dataframe.currency.map(
                    lambda x: currencies_prices[x]
                )
        if to_csv:
            dataframe.to_csv(filename, index=False)
            return
        return dataframe

    @connection
    def get_currencies_exchange_rates(
        self, *, use_tqdm: bool = False
    ) -> dict[str, float]:
        """
        Retrieves and calculates exchange rates for available currencies against RUB.

        Args:
            use_tqdm (bool, optional): If True, displays a progress bar for currency parsing. Defaults to False.

        Returns:
            dict[str, float]: A dictionary mapping each currency's ISO code to its exchange rate relative to RUB.
                              The RUB currency is set to an exchange rate of 1.
        """
        currencies = self.parse_currencies()
        iterator = (
            tqdm(currencies.instruments, desc="Obtaining exchange rates")
            if use_tqdm
            else currencies.instruments
        )
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

    @connection
    def parse_price_history(
        self,
        figis: Union[str, list[str]],
        from_date: datetime = now() - timedelta(days=365),
        to_date: datetime = now(),
        interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_DAY,
        use_tqdm: bool = False,
        generator: bool = False,
        retry_if_limit: bool = False,
    ) -> Union[_price_history_type, Generator[pd.DataFrame, None, None]]:
        """
        Retrieves historical price data for a single FIGI or multiple FIGIs over a specified date range,
        optionally filtered up to a certain date.

        Args:
            figis (Union[str, list[str]]): A single FIGI string or a list of FIGIs for the instruments.
            from_date (datetime): Start date for the price history.
            to_date (datetime): End date for the price history.
            interval (CandleInterval, optional): The interval for the candle data. Defaults to daily.
            use_tqdm (bool, optional): If True, displays a progress bar for price history evaluating.
            Defaults to False.
            generator (bool, optional): If True, yields price history as a generator. Defaults to False.
            retry_if_limit (bool, optional): If True, try to retry the request at the limit (RequestError). Defaults to False.
        Returns:
            Union[dict[str, pd.DataFrame], Generator[pd.Series, None, dict[str, pd.DataFrame]]]: A dict where each key is a FIGI, and the value is a pd.DataFrame with
            historical price data. Or the price history generator if needed.
        """
        gen = self._parse_price_history(
            figis=figis,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            use_tqdm=use_tqdm and generator,
            retry_if_limit=retry_if_limit,
        )

        if generator:
            return gen

        price_history = {}

        iterator = (
            tqdm(zip(figis, gen), desc="Obtaining price history")
            if use_tqdm
            else zip(figis, gen)
        )
        for figi, price in iterator:
            price_history[figi] = price

        return price_history

    def _parse_price_history(
        self,
        figis: Union[str, list[str]],
        from_date: datetime = now() - timedelta(days=365),
        to_date: datetime = now(),
        interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_DAY,
        use_tqdm: bool = False,
        retry_if_limit: bool = False,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Retrieves historical price data for a single FIGI or multiple FIGIs over a specified date range.

        Args:
            figis (Union[str, list[str]]): A single FIGI string or a list of FIGIs for the instruments.
            from_date (datetime): Start date for the price history.
            to_date (datetime): End date for the price history.
            interval (CandleInterval, optional): The interval for the candle data. Defaults to daily.
            use_tqdm (bool, optional): If True, displays a progress bar for price history evaluating.
            retry_if_limit (bool, optional): Whether to retry fetching data if rate limit is hit. Defaults to False.

        Yields:
            pd.DataFrame: A DataFrame containing historical price data for each FIGI.
        """

        def validate_request():
            response = self._channel.market_data.get_candles(
                figi=figi, from_=from_date, to=to_date, interval=interval
            )

            candles = [
                {
                    "time": candle.time,
                    "open": candle.open.units + (candle.open.nano / 1e9),
                    "close": candle.close.units + (candle.close.nano / 1e9),
                    "high": candle.high.units + (candle.high.nano / 1e9),
                    "low": candle.low.units + (candle.low.nano / 1e9),
                    "volume": candle.volume,
                }
                for candle in response.candles
            ]
            return pd.DataFrame(candles)

        if isinstance(figis, str):
            figis = [figis]

        iterator = tqdm(figis, desc="Obtaining price history") if use_tqdm else figis

        for figi in iterator:
            while True:
                try:
                    yield validate_request()
                    break
                except RequestError as error:
                    if retry_if_limit:
                        warnings.warn(
                            "Request has been stopped via the API (limit). Retry in 10 seconds.",
                            category=RuntimeWarning,
                        )
                        time.sleep(10)
                        continue
                    raise error

    def __repr__(self):
        return "APIParser({})".format(getattr(self, "_channel", None) is not None)
