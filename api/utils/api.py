import uvicorn

from fastapi import FastAPI

from .portfolio import RecommendationSystem, Portfolio
from .storage import SecurityVault

app = FastAPI()
storage = SecurityVault()

if not all(storage.cache_check().values()):
    storage.build()

recommendation_system = RecommendationSystem(storage)

@app.post("/get-portfolio/{user_id}/{capacity}")
async def read_item(user_id: int, capacity: int):
    portfolio = Portfolio(user_id=user_id)
    recommendation_system.recommend(portfolio, capacity)
    return portfolio.json


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
