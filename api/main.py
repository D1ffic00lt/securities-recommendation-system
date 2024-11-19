import os
import uvicorn

try:
    from utils.api import app
except ValueError:
    os.environ["TINKOFF_TOKEN"] = (
        open(os.environ.get("TOKEN"), "r").read().strip()
    )
    from utils.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="api", port=8002)
