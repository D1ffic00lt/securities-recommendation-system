from enum import Enum

__all__ = ("ResponseColumns", )

class ResponseColumns(Enum):
    SHARES = [
        "figi", "ticker", "class_code", "isin",
        "currency", "name", "exchange", "ipo_date",
        "issue_size", "country_of_risk", "country_of_risk_name",
        "sector", "issue_size_plan", "liquidity_flag", "lot"
    ]
    BONDS = [
        "figi", "ticker", "class_code", "isin", "currency",
        "name", "exchange", "coupon_quantity_per_year",
        "maturity_date", "nominal", "initial_nominal",
        "placement_price", "aci_value", "country_of_risk",
        "country_of_risk_name", "sector", "issue_kind", "issue_size",
        "issue_size_plan", "liquidity_flag", "risk_level",
        "min_price_increment", "lot"
    ]
    ETFS = [
        "figi", "ticker", "class_code", "isin", "currency", "name",
        "exchange", "fixed_commission", "focus_type",
        "released_date", "num_shares", "country_of_risk",
        "country_of_risk_name", "sector", "rebalancing_freq",
        "min_price_increment", "lot"
    ]
    CURRENCIES = [
        "figi", "ticker", "class_code", "isin", "lot", "currency",
        "name", "exchange", "nominal", "country_of_risk",
        "country_of_risk_name", "iso_currency_name", "min_price_increment"
    ]
