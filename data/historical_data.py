from oanda_api import OandaAPI
from db_connection import insert_data_into_db  # Assuming you have this function in db_connection.py
import datetime

def process_historical_data(json_data):
    # Parse the JSON response and extract relevant data
    processed_data = []
    for candle in json_data['candles']:
        time = candle['time']
        open_price = candle['mid']['o']
        high_price = candle['mid']['h']
        low_price = candle['mid']['l']
        close_price = candle['mid']['c']
        volume = candle['volume']
        processed_data.append((time, open_price, high_price, low_price, close_price, volume))
    return processed_data

def store_historical_data(processed_data):
    # Store the processed data in the database
    for data in processed_data:
        insert_data_into_db(data)

# Example usage
if __name__ == "__main__":
    oanda_api = OandaAPI()  # Initialize with necessary parameters
    end_datetime = datetime.datetime.utcnow().isoformat() + 'Z'  # Current UTC time in ISO 8601 format
    start_datetime = "1990-01-01T00:00:00Z"  # Start date as far back as data is available
    json_data = oanda_api.get_historical_data(start_datetime, end_datetime)
    processed_data = process_historical_data(json_data)
    store_historical_data(processed_data)
