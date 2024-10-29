# src/data/db_connection.py

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_db_connection():
    # Fetch database credentials from environment variables
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST', 'localhost')  # Default to 'localhost' if not set
    db_name = os.getenv('DB_NAME')

    if not all([db_user, db_password, db_host, db_name]):
        print("Error: One or more database credentials are missing in the environment variables.")
        return None

    try:
        engine_url = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"
        engine = create_engine(engine_url)
        connection = engine.connect()
        print("Database connection established successfully.")
        return connection
    except SQLAlchemyError as e:
        print(f"Error connecting to the database: {e}")
        return None

def close_db_connection(connection):
    connection.close()

def fetch_historical_data(): # test
    connection = create_db_connection()
    if connection is None:
        return pd.DataFrame()  # Return empty DataFrame if connection failed

    try:
        # Adjusted the SELECT statement to use aliases matching your table schema
        query = """
        SELECT `timestamp`,
               open_price AS `open`,
               high_price AS `high`,
               low_price AS `low`,
               close_price AS `close`,
               volume
        FROM historical_data
        ORDER BY `timestamp` ASC
        """
        df = pd.read_sql(query, connection)
        close_db_connection(connection)
        return df
    except SQLAlchemyError as e:
        print(f"Error fetching data: {e}")
        close_db_connection(connection)
        return pd.DataFrame()  # Return an empty DataFrame on error
