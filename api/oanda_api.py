# oanda_api.py
from config import API_KEY

import requests

class OandaAPI:
    def __init__(self, account_id):
        self.account_id = account_id
        self.api_base_url = 'https://api-fxpractice.oanda.com/v3'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }

    def get_account_details(self):
        try:
            url = f'{self.api_base_url}/accounts/{self.account_id}'
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def get_pricing(self, instruments):
        url = f'{self.api_base_url}/accounts/{self.account_id}/pricing'
        params = {'instruments': instruments}
        response = requests.get(url, headers=self.headers, params=params)
        return response.json()

    def create_market_order(self, instrument, units):
        url = f"{self.api_base_url}/accounts/{self.account_id}/orders"
        data = {
            "order": {
                "instrument": instrument,
                "units": units,
                "type": "MARKET"
            }
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

    def create_limit_order(self, instrument, units, price):
        url = f"{self.api_base_url}/accounts/{self.account_id}/orders"
        data = {
            "order": {
                "instrument": instrument,
                "units": units,
                "price": str(price),
                "type": "LIMIT"
            }
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

    def create_stop_order(self, instrument, units, price):
        url = f"{self.api_base_url}/accounts/{self.account_id}/orders"
        data = {
            "order": {
                "instrument": instrument,
                "units": units,
                "price": str(price),
                "type": "STOP"
            }
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

    def close_trade(self, trade_id):
        url = f"{self.api_base_url}/accounts/{self.account_id}/trades/{trade_id}/close"
        response = requests.put(url, headers=self.headers)
        return response.json()
