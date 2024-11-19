import requests


def get_portfolio(user_id, capacity):
    url = f"http://api:8002/get-portfolio/{user_id}/{capacity}"
    try:
        response = requests.post(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching portfolio: {e}")
        return None


def get_price(figis: list[str], how: str = "actual", by: str = "candle_price"):
    url = f"http://api:8002/get-portfolio-price/{how}/{by}"
    try:
        response = requests.post(url, json=figis)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching price: {e}")
        return None
