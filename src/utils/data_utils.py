# data_utils.py
import pandas as pd
import mysql.connector

def prepare_time_series_data():
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'root',
        'database': 'python_ea'
    }
    connection = mysql.connector.connect(**db_config)
    query = """SELECT timestamp, open_price, high_price, low_price, close_price, volume
               FROM historical_data
               WHERE currency_pair = 'USD/CAD'
               ORDER BY timestamp ASC;"""
    df = pd.read_sql(query, con=connection, parse_dates=['timestamp'], index_col='timestamp')
    connection.close()
    return df
