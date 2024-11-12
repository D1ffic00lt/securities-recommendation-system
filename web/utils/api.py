import requests

def get_portfolio(user_id, capacity):
    url = f'http://localhost:8002/get-portfolio/{user_id}/{capacity}'
    try:
        response = requests.post(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching portfolio: {e}")
        return None
