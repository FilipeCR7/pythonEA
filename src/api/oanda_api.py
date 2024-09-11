import requests
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta


class OandaAPI:
    def __init__(self, account_id):
        """
        Initialize the Oanda API client with the account ID and necessary headers.
        """
        self.account_id = account_id
        self.api_base_url = 'https://api-fxpractice.oanda.com/v3'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer bafecd1d1dd3075320ca677069c53a04-1876cffb0dfc669fcd5ce6a54794d1ad'
            # Your OANDA API key with 'Bearer'
        }
        # MySQL Database Connection
        self.database_uri = 'mysql+mysqlconnector://root:root@localhost/python_ea'
        self.engine = create_engine(self.database_uri)

    def get_historical_data(self, instrument="USD_CAD", start_date="2023-01-01T00:00:00Z",
                            end_date="2023-01-01T23:59:00Z"):
        """
        Fetch historical candlestick data for the full day (minute-level granularity).
        It fetches data in hourly batches to ensure all data is collected.
        """
        url = f"{self.api_base_url}/instruments/{instrument}/candles"
        fetched_data = []

        # Convert strings to datetime objects
        current_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")

        # Fetch data hour by hour
        while current_date < end_datetime:
            next_hour = current_date + timedelta(hours=1)
            params = {
                "from": current_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": next_hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "granularity": "M1",  # Minute-level data
                "price": "M"  # Midpoint price
            }
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                candles = response.json().get('candles', [])
                fetched_data.extend(candles)
                print(f"Fetched {len(candles)} candles from {current_date} to {next_hour}.")
            else:
                print(f"Error fetching data: {response.status_code} - {response.text}")

            # Move to the next hour
            current_date = next_hour

        print(f"Total fetched candles: {len(fetched_data)}")
        return fetched_data

    def insert_historical_data_into_db(self, historical_data):
        """
        Insert historical data fetched from OANDA into the MySQL database.
        """
        if not historical_data:
            print("No candles data available to insert.")
            return

        data = []
        for candle in historical_data:
            timestamp = candle['time']
            open_price = candle['mid']['o']
            high_price = candle['mid']['h']
            low_price = candle['mid']['l']
            close_price = candle['mid']['c']
            volume = candle['volume']

            data.append({
                "timestamp": timestamp,
                "open_price": float(open_price),
                "high_price": float(high_price),
                "low_price": float(low_price),
                "close_price": float(close_price),
                "volume": int(volume)
            })

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.to_sql('historical_data', con=self.engine, if_exists='append', index=False)
        print(f"Inserted {len(df)} rows of historical data into the database.")


# Example usage
if __name__ == "__main__":
    oanda_api = OandaAPI(account_id="101-004-27721570-001")
    # Fetch one day worth of historical data in minute candles from a known active trading day (e.g., January 9, 2023)
    historical_data = oanda_api.get_historical_data(start_date="2023-01-09T00:00:00Z", end_date="2023-01-09T23:59:00Z")
    # Insert the data into the MySQL database
    oanda_api.insert_historical_data_into_db(historical_data)

