import os
import pandas as pd

from datetime import datetime
from typing import Any, Union
from tqdm.auto import tqdm
from tinkoff.invest import Client, InstrumentStatus
from tinkoff.invest.schemas import (
    BondsResponse, CurrenciesResponse, EtfsResponse, RiskLevel,
    SharesResponse
)
from tinkoff.invest.services import Services
from functools import singledispatchmethod, wraps

from ._types import *
from ._properties import *


__all__ = ("APIParser", )

class APIParser:
    def __init__(self, token: str = os.environ.get('TINKOFF_TOKEN', None)):
        if token is None:
            raise ValueError('Tinkoff token must be provided')
        self._token = token
        self._client = Client(self._token)
        self._channel: Services

    def __enter__(self):
        self._channel = self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.__exit__(exc_type, exc_val, exc_tb)
        self._channel = None

    @staticmethod
    def connection(func):
        @wraps(func)
        def wrapper(self: "APIParser", *args, **kwargs):
            if getattr(self, '_channel', None) is not None:
                return func(self, *args, **kwargs)
            with self as self:
                return func(self, *args, **kwargs)
        return wrapper

    @connection
    def parse_shares(self) -> SharesResponse:
        return self._channel.instruments.shares(
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
        )

    @connection
    def parse_bonds(self) -> BondsResponse:
        return self._channel.instruments.bonds(
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
        )

    @connection
    def parse_etfs(self) -> EtfsResponse:
        return self._channel.instruments.etfs(
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
        )

    @connection
    def parse_currencies(self) -> CurrenciesResponse:
        return self._channel.instruments.currencies(
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
        )

    @singledispatchmethod
    def write(
        self, data, filename: str, *, use_tqdm: bool = False
    ) -> None:
        raise ValueError("Invalid data type. (%s)" % data.__class__.__name__)

    @write.register
    def _(
        self, data: SharesResponse, filename: str, use_tqdm: bool = False
    ) -> None:
        self._generate_csv(
            columns=ResponseColumns.SHARES.value,
            data=data, filename=filename, use_tqdm=use_tqdm
        )

    @write.register
    def _(
        self, data: BondsResponse, filename: str, *, use_tqdm: bool = False
    ) -> None:
        self._generate_csv(
            columns=ResponseColumns.BONDS.value,
            data=data, filename=filename, use_tqdm=use_tqdm
        )

    @write.register
    def _(
        self, data: EtfsResponse, filename: str, *, use_tqdm: bool = False
    ) -> None:
        self._generate_csv(
            columns=ResponseColumns.ETFS.value,
            data=data, filename=filename, use_tqdm=use_tqdm
        )

    @write.register
    def _(
        self, data: CurrenciesResponse,
        filename: str, *, use_tqdm: bool = False
    ) -> None:
        self._generate_csv(
            columns=ResponseColumns.CURRENCIES.value,
            data=data, filename=filename, use_tqdm=use_tqdm
        )

    def _generate_csv(
        self, columns: list[str], data: _response_types,
        filename: str, use_tqdm: bool = False
    ) -> None:
        iterator = tqdm(data.instruments) if use_tqdm else data.instruments
        data = [
            {
                attr: self._validate_value(getattr(row, attr))
                for attr in columns
            }
            for row in iterator
        ]

        dataframe = pd.DataFrame(data, columns=columns)
        dataframe.to_csv(filename, index=False)

    @staticmethod
    def _validate_value(value: Any) -> Union[float, str, int]:
        if isinstance(value, _money_types):
            return value.units + (value.nano / 1e9)
        elif isinstance(value, datetime):
            return value.strftime("%d/%m/%Y")
        elif isinstance(value, RiskLevel):
            return value.value
        return value

if __name__ == '__main__':
    with open("../../secrets/tinkoff_token.txt", "r") as f:
        api_token = f.read().strip()

    parser = APIParser(api_token)
    shares = parser.parse_shares()
