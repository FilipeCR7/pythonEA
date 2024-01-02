import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from your_project.historical_data import prepare_time_series_data  # Adjust the import path as needed


def plot_time_series(df, column, title):
    """
    Plot the time series data.

    Parameters:
    df (DataFrame): Pandas DataFrame containing the time series data.
    column (str): Column name to plot.
    title (str): Title of the plot.
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

    Parameters:
    df (DataFrame): Pandas DataFrame containing the time series data.
    column (str): Column name to test for stationarity.
    """
    result = adfuller(df[column])
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')


def main():
    # Example currency pair
    currency_pair = 'USD/CAD'

    # Prepare data
    df = prepare_time_series_data(currency_pair)

    # Plotting the data
    plot_time_series(df, 'close', f'{currency_pair} Closing Prices')
    plot_time_series(df, 'returns', f'{currency_pair} Returns')

    # Check stationarity
    check_stationarity(df, 'returns')


if __name__ == "__main__":
    main()
