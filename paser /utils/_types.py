from typing import Union

from tinkoff.invest import (
    BondsResponse, CurrenciesResponse, EtfsResponse,
    MoneyValue, Quotation, SharesResponse
)

__all__ = ("_response_types", "_money_types")

_response_types = Union[
    SharesResponse, BondsResponse,
    EtfsResponse, CurrenciesResponse
]
_money_types = Union[MoneyValue, Quotation]
