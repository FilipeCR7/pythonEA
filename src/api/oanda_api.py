from src.config import API_KEY
import requests


class OandaAPI:
    def __init__(self, account_id):
        """
        Initialize the Oanda API client with the account ID and necessary headers.

        :param account_id: Oanda account ID for making requests.
        """
        self.account_id = account_id
        self.api_base_url = 'https://api-fxpractice.oanda.com/v3'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }

    def get_account_details(self):
        """
        Fetch account details for the specified account ID.

        :return: JSON response with account details or None in case of an error.
        """
        try:
            url = f'{self.api_base_url}/accounts/{self.account_id}'
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def get_pricing(self, instruments):
        """
        Fetch real-time pricing data for the specified instruments.

        :param instruments: Comma-separated string of instrument names (e.g., 'EUR_USD,USD_CAD').
        :return: JSON response with pricing data.
        """
        url = f'{self.api_base_url}/accounts/{self.account_id}/pricing'
        params = {'instruments': instruments}
        response = requests.get(url, headers=self.headers, params=params)
        return response.json()

    def create_market_order(self, instrument, units):
        """
        Create a market order for the specified instrument and units.

        :param instrument: Trading instrument (e.g., 'USD_CAD').
        :param units: Number of units to buy (positive) or sell (negative).
        :return: JSON response with the order details.
        """
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
        """
        Create a limit order for the specified instrument, units, and price.

        :param instrument: Trading instrument (e.g., 'USD_CAD').
        :param units: Number of units to buy (positive) or sell (negative).
        :param price: Limit price for the order.
        :return: JSON response with the order details.
        """
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
        """
        Create a stop order for the specified instrument, units, and stop price.

        :param instrument: Trading instrument (e.g., 'USD_CAD').
        :param units: Number of units to buy (positive) or sell (negative).
        :param price: Stop price for the order.
        :return: JSON response with the order details.
        """
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
        """
        Close an open trade by trade ID.

        :param trade_id: The ID of the trade to close.
        :return: JSON response with the result of the trade closure.
        """
        url = f"{self.api_base_url}/accounts/{self.account_id}/trades/{trade_id}/close"
        response = requests.put(url, headers=self.headers)
        return response.json()

    def get_historical_data(self, start_date, end_date):
        """
        Fetch historical candlestick data for a given time period and instrument.

        :param start_date: Start date for the data in ISO 8601 format.
        :param end_date: End date for the data in ISO 8601 format.
        :return: JSON response with historical candlestick data.
        """
        url = f"{self.api_base_url}/instruments/USD_CAD/candles"
        params = {
            "from": start_date,
            "to": end_date,
            "granularity": "M1",  # Minute-level data
            "price": "M"  # Midpoint price
        }
        response = requests.get(url, headers=self.headers, params=params)
        return response.json()

# Example usage:
# oanda_api = OandaAPI(account_id)
# historical_data = oanda_api.get_historical_data("2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")

