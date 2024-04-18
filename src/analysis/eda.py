import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.data_utils import prepare_time_series_data
from statsmodels.tsa.stattools import adfuller


def plot_time_series(df, column, title):
    """
    Plot the time series data.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[column])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.show()


def check_stationarity(df, column):
    """
    Perform the Augmented Dickey-Fuller test for stationarity.
    """
    result = adfuller(df[column].dropna())  # Ensure to drop NA values
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')


def main():
    # Prepare data
    df = prepare_time_series_data()

    # Plotting the data
    plot_time_series(df, 'close_price', 'USD/CAD Closing Prices')
    plot_time_series(df, 'returns', 'USD/CAD Returns')

    # Check stationarity
    check_stationarity(df, 'returns')


if __name__ == "__main__":
    main()
