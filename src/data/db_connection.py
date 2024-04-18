import mysql.connector

def create_db_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',  # or your host
            user='root',
            password='root',
            database='python_ea'
        )
    except mysql.connector.Error as e:
        print(f"Error: {e}")
    return connection

def close_db_connection(connection):
    if connection:
        connection.close()

def insert_data_into_db(data):
    connection = create_db_connection()
    if connection is None:
        return

    cursor = connection.cursor()
    query = """
    INSERT INTO historical_data (timestamp, open_price, high_price, low_price, close_price, volume)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        cursor.executemany(query, data)  # Change back to executemany - test
        connection.commit()
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        connection.rollback()
    finally:
        close_db_connection(connection)

