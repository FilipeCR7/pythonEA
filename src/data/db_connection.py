# File: db_connection.py

from sqlalchemy import create_engine
import pandas as pd

def create_db_connection():
    try:
        engine = create_engine('mysql+mysqlconnector://root:root@localhost/python_ea')
        connection = engine.connect()
    except Exception as e:
        print(f"Error: {e}")
        connection = None
    return connection

def close_db_connection(connection):
    if connection:
        connection.close()

def insert_data_into_db(data):
    connection = create_db_connection()
    if connection is None:
        return

    query = """
    INSERT INTO historical_data (timestamp, open_price, high_price, low_price, close_price, volume)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        connection.execute(query, data)
        connection.commit()
    except Exception as e:
        print(f"Error: {e}")
        connection.rollback()
    finally:
        close_db_connection(connection)

def fetch_historical_data():
    """
    Fetch historical data between specified timestamps.
    """
    connection = create_db_connection()
    if connection is None:
        return None

    query = """
    SELECT timestamp, close_price AS close
    FROM historical_data
    WHERE timestamp BETWEEN '2023-01-09 00:00:00' AND '2023-01-09 23:59:59'
    """
    df = pd.read_sql(query, connection, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    close_db_connection(connection)
    return df
