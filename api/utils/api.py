import warnings
from datetime import timedelta

import uvicorn

from fastapi import FastAPI, Request
from tinkoff.invest.utils import now

from .portfolio import RecommendationSystem, Portfolio
from .storage import SecurityVault

app = FastAPI()
actual_storage = SecurityVault(cache_path="./.cache_actual")
storage = SecurityVault()

with warnings.catch_warnings(action="ignore"):
    if not all(storage.cache_check().values()):
        storage.build(to_date=now() - timedelta(days=30))
    if not all(actual_storage.cache_check().values()):
        actual_storage.build()

recommendation_system = RecommendationSystem(storage)


@app.post("/get-portfolio/{user_id}/{capacity}")
async def read_item(user_id: int, capacity: int):
    portfolio = Portfolio(user_id=user_id)
    recommendation_system.recommend(portfolio, capacity)
    return portfolio.json


@app.post("/get-portfolio-price/{where}/{by}")
async def get_portfolio_price(
    request: Request, where: str = "actual", by: str = "candle_price"
):
    figis = await request.json()
    if where == "actual":
        prices = actual_storage.get_price(figis, by=by)
    else:
        prices = storage.get_price(figis, by=by)
    return prices


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
