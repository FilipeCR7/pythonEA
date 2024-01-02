import numpy as np
import pandas as pd


class MeanReversionStrategy:
    def __init__(self, df, window=20, std_dev_multiplier=2):
        """
        Initialize the strategy with a DataFrame containing historical data and technical indicators.

        :param df: Pandas DataFrame containing historical price data and technical indicators.
        :param window: The window size for calculating the moving average.
        :param std_dev_multiplier: Multiplier for standard deviation to identify significant deviations.
        """
        self.df = df
        self.window = window
        self.std_dev_multiplier = std_dev_multiplier

    def generate_signals(self):
        """
        Generate buy and sell signals based on mean reversion logic using the moving average and standard deviation.
        """
        # Use the moving average and standard deviation from the DataFrame
        self.df['lower_bound'] = self.df['sma_' + str(self.window)] - (
                    self.std_dev_multiplier * self.df['sma_' + str(self.window)].std())
        self.df['upper_bound'] = self.df['sma_' + str(self.window)] + (
                    self.std_dev_multiplier * self.df['sma_' + str(self.window)].std())

        # Assuming the latest data is at the end of the DataFrame
        current_price = self.df['close'].iloc[-1]
        lower_bound = self.df['lower_bound'].iloc[-1]
        upper_bound = self.df['upper_bound'].iloc[-1]

        if current_price <= lower_bound:
            return "BUY"  # Price is significantly lower than the mean
        elif current_price >= upper_bound:
            return "SELL"  # Price is significantly higher than the mean
        else:
            return "HOLD"  # Price is within the normal range

# Example usage
# Assuming you have a DataFrame 'enriched_df' from your historical_data.py
# strategy = MeanReversionStrategy(enriched_df)
# signal = strategy.generate_signals()
# print(signal)
