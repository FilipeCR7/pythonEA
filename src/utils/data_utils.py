import pandas as pd
from sqlalchemy import create_engine


def prepare_time_series_data():
    """
    Fetch time series data from the MySQL database for a specific date range
    and return it as a pandas DataFrame.

    This function retrieves historical data, sets the 'timestamp' as the index,
    and parses it as datetime for time series analysis.

    :return: DataFrame containing 'timestamp' and 'close_price' as 'close'.
    """
    db_config = {
        'user': 'root',
        'password': 'root',
        'host': 'localhost',
        'database': 'python_ea'
    }

    # Construct the database connection URI
    DATABASE_URI = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
    engine = create_engine(DATABASE_URI)

    # SQL query to fetch the relevant historical data
    query = """SELECT timestamp, close_price AS close
               FROM historical_data
               WHERE timestamp BETWEEN '2023-12-17 22:03:00' AND '2023-12-18 14:52:00'
               ORDER BY timestamp ASC;"""

    # Execute the query and parse 'timestamp' as datetime
    df = pd.read_sql(query, con=engine, parse_dates=['timestamp'])

    # Set 'timestamp' as the DataFrame index
    df.set_index('timestamp', inplace=True)

    return df
