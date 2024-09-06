import numpy as np
import pandas as pd

class MeanReversionStrategy:
    def __init__(self, df, window=20, std_dev_multiplier=2):
        """
        Initializes the mean reversion strategy.

        :param df: DataFrame with historical price data.
        :param window: Rolling window size for the moving average.
        :param std_dev_multiplier: Multiplier for standard deviation.
        """
        self.df = df
        self.window = window
        self.std_dev_multiplier = std_dev_multiplier

    def generate_signals(self):
        """
        Generates buy/sell signals based on deviation from the moving average.
        """
        # Calculate upper and lower bounds using SMA and standard deviation
        self.df['lower_bound'] = self.df['sma_' + str(self.window)] - (
            self.std_dev_multiplier * self.df['sma_' + str(self.window)].std())
        self.df['upper_bound'] = self.df['sma_' + str(self.window)] + (
            self.std_dev_multiplier * self.df['sma_' + str(self.window)].std())

        # Get the latest price and bounds
        current_price = self.df['close'].iloc[-1]
        lower_bound = self.df['lower_bound'].iloc[-1]
        upper_bound = self.df['upper_bound'].iloc[-1]

        # Generate signal based on price position relative to bounds
        if current_price <= lower_bound:
            return "BUY"
        elif current_price >= upper_bound:
            return "SELL"
        else:
            return "HOLD"
