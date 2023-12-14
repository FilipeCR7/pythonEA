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
