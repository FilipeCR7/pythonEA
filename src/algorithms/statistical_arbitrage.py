import numpy as np
import pandas as pd


class StatisticalArbitrageStrategy:
    def __init__(self, df_pair1, df_pair2, threshold=0.02):
        """
        Initialize the strategy with DataFrames for two currency pairs.

        :param df_pair1: Pandas DataFrame for the first currency pair (e.g., USD/CAD).
        :param df_pair2: Pandas DataFrame for the second currency pair.
        :param threshold: Threshold for identifying significant spread deviations.
        """
        self.df_pair1 = df_pair1
        self.df_pair2 = df_pair2
        self.threshold = threshold

    def calculate_spread(self):
        """
        Calculate the spread between the two currency pairs.
        """
        spread = self.df_pair1['close'] - self.df_pair2['close']
        return spread

    def generate_signals(self):
        """
        Generate trading signals based on the spread's deviation from its mean.
        """
        spread = self.calculate_spread()
        mean_spread = spread.mean()
        std_spread = spread.std()

        current_spread = spread.iloc[-1]

        if current_spread > mean_spread + self.threshold * std_spread:
            return "SELL Pair 1 (USD/CAD), BUY Pair 2"
        elif current_spread < mean_spread - self.threshold * std_spread:
            return "BUY Pair 1 (USD/CAD), SELL Pair 2"
        else:
            return "HOLD"

# Example usage
# Assuming you have DataFrames 'df_usdcad' and 'df_eurusd' from your historical_data.py
# strategy = StatisticalArbitrageStrategy(df_usdcad, df_eurusd)
# signal = strategy.generate_signals()
# print(signal)
