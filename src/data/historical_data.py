from src.api.oanda_api import OandaAPI
from src.data.db_connection import insert_data_into_db
from datetime import timezone
from datetime import datetime

def process_historical_data(json_data):
    if 'candles' not in json_data:
        print("Candles key not in response. Response:", json_data)
        return []
    processed_data = []
    for candle in json_data['candles']:
        # Remove nanoseconds and replace 'Z' with '+00:00'
        time_str = candle['time'].split('.')[0]
        if time_str.endswith('Z'):
            time_str = time_str.replace('Z', '+00:00')
        time = datetime.fromisoformat(time_str).strftime('%Y-%m-%d %H:%M:%S')
        open_price = candle['mid']['o']
        high_price = candle['mid']['h']
        low_price = candle['mid']['l']
        close_price = candle['mid']['c']
        volume = candle['volume']
        processed_data.append((time, open_price, high_price, low_price, close_price, volume))
    return processed_data


def store_historical_data(processed_data):
    # Store the processed data in the database
    if processed_data:
        insert_data_into_db(processed_data)
# Example usage
if __name__ == "__main__":
    oanda_api = OandaAPI('101-004-27721570-001')
    end_datetime = datetime.now(timezone.utc).isoformat()
    start_datetime = "2023-12-17T00:00:00Z"
    json_data = oanda_api.get_historical_data(start_datetime, end_datetime)
    processed_data = process_historical_data(json_data)

    print("Processed Data:", processed_data)

    store_historical_data(processed_data)

