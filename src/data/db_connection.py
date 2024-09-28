# src/data/db_connection.py

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

def create_db_connection():
    # Replace with your actual database connection details
    db_user = 'root'          # Your MySQL username
    db_password = 'root'      # Your MySQL password
    db_host = 'localhost'     # Update if necessary
    db_name = 'python_ea'     # Your database name

    engine_url = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"
    engine = create_engine(engine_url)
    connection = engine.connect()
    return connection

def close_db_connection(connection):
    connection.close()

def fetch_historical_data():
    connection = create_db_connection()
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
