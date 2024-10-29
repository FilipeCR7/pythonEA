# your_project/scripts/preprocess_and_load.py

import pandas as pd
from sqlalchemy import create_engine
import os
import sys

# Adjust the path to ensure modules can be imported correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.db_connection import create_db_connection


def preprocess_csv(csv_path):
    """
    Preprocess the CSV data to match the database schema.

    :param csv_path: Path to the CSV file.
    :return: Preprocessed pandas DataFrame.
    """
    try:
        df = pd.read_csv(csv_path)
        print("CSV loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the CSV: {e}")
        return None

    # Inspect the DataFrame columns
    print("Original Columns:", df.columns.tolist())

    # Define column mapping based on your CSV structure
    column_mapping = {
        'Local time': 'timestamp',
        'Open': 'open_price',
        'High': 'high_price',
        'Low': 'low_price',
        'Close': 'close_price',
        'Volume': 'volume'
    }

    # Rename columns
    df.rename(columns=column_mapping, inplace=True)
    print("Columns after renaming:", df.columns.tolist())

    # Since 'Local time' is already a timestamp, no need to combine 'Date' and 'Time'
    # Remove any 'Date' and 'Time' columns if they exist
    if 'Date' in df.columns:
        df.drop(columns=['Date'], inplace=True)
        print("Dropped 'Date' column.")
    if 'Time' in df.columns:
        df.drop(columns=['Time'], inplace=True)
        print("Dropped 'Time' column.")

    # Adjust Time Zones if necessary
    # Uncomment and modify the following lines if your data is in EET and needs to be converted to UTC
    # try:
    #     df['timestamp'] = pd.to_datetime(df['timestamp'])
    #     df['timestamp'] = df['timestamp'].dt.tz_localize('EET').dt.tz_convert('UTC').dt.tz_localize(None)
    #     print("Timestamp converted from EET to UTC.")
    # except Exception as e:
    #     print(f"Error converting time zones: {e}")
    #     return None

    # Ensure correct data types
    price_columns = ['open_price', 'high_price', 'low_price', 'close_price']
    for col in price_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)

    print("Data types after conversion:")
    print(df.dtypes)

    # Reorder columns to match the database schema
    try:
        df = df[['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']]
        print("Columns reordered to match the database schema.")
    except KeyError as e:
        print(f"Error: Missing expected columns after renaming: {e}")
        return None

    # Handle duplicates
    initial_count = len(df)
    df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
    duplicates_removed = initial_count - len(df)
    print(f"Duplicates removed: {duplicates_removed}")

    # Handle missing values
    initial_count = len(df)
    df.dropna(subset=['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'], inplace=True)
    missing_removed = initial_count - len(df)
    print(f"Rows with missing values removed: {missing_removed}")

    return df


def insert_into_database(df, engine):
    """
    Insert the DataFrame into the historical_data table.

    :param df: Preprocessed pandas DataFrame.
    :param engine: SQLAlchemy engine connected to the database.
    """
    if df is None or df.empty:
        print("No data to insert.")
        return

    try:
        # Reset index if 'timestamp' is set as index
        if df.index.name == 'timestamp':
            df.reset_index(inplace=True)

        # Insert data
        df.to_sql('historical_data', con=engine, if_exists='append', index=False, chunksize=10000, method='multi')
        print("Data inserted into the 'historical_data' table successfully.")
    except Exception as e:
        print(f"Error inserting data into the database: {e}")


def main():
    # Path to your CSV file
    csv_file_path = 'filipe/code/dataEA/USD_CAD_data.csv'  # Updated to your actual path

    # Preprocess the CSV
    df = preprocess_csv(csv_file_path)

    if df is not None:
        # Create a database connection
        engine = create_db_connection()

        if engine:
            # Insert data into the database
            insert_into_database(df, engine)

            # Verify insertion
            try:
                sample_df = pd.read_sql('SELECT * FROM historical_data ORDER BY timestamp DESC LIMIT 5', con=engine)
                print("Sample data from 'historical_data' table:")
                print(sample_df)

                # Check total record count
                result = engine.execute('SELECT COUNT(*) FROM historical_data')
                row_count = result.fetchone()[0]
                print(f"Total records in 'historical_data': {row_count}")
            except Exception as e:
                print(f"Error verifying data in the database: {e}")
        else:
            print("Failed to create a database connection.")
    else:
        print("Data preprocessing failed. Exiting.")


if __name__ == "__main__":
    main()
