import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.data_utils import prepare_time_series_data
from statsmodels.tsa.stattools import adfuller


def plot_time_series(df, column, title):
    """
    Plot the time series data for a specific column.

    :param df: Pandas DataFrame containing the data.
    :param column: The column of the DataFrame to plot.
    :param title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[column])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.show()


def check_stationarity(df, column):
    """
    Perform the Augmented Dickey-Fuller test to check for stationarity of the data.

    :param df: Pandas DataFrame containing the data.
    :param column: The column to perform the stationarity test on.
    """
    result = adfuller(df[column].dropna())  # Drop NA values for test
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')


def main():
    """
    Main function to load data, plot time series, and check for stationarity.
    """
    # Load time series data
    df = prepare_time_series_data()

    # Plot the closing prices and returns
    plot_time_series(df, 'close_price', 'USD/CAD Closing Prices')
    plot_time_series(df, 'returns', 'USD/CAD Returns')

    # Check for stationarity in the returns
    check_stationarity(df, 'returns')


if __name__ == "__main__":
    main()
