import numpy as np
import pandas as pd


class StatisticalArbitrageStrategy:
    def __init__(self, df_pair1, df_pair2, threshold=0.02):
        """
        Initializes the strategy with two DataFrames representing currency pairs.

        :param df_pair1: DataFrame for the first currency pair (e.g., USD/CAD).
        :param df_pair2: DataFrame for the second currency pair.
        :param threshold: Threshold for spread deviation to trigger signals.
        """
        self.df_pair1 = df_pair1
        self.df_pair2 = df_pair2
        self.threshold = threshold

    def calculate_spread(self):
        """
        Calculates the spread between the closing prices of the two currency pairs.

        :return: Series representing the spread between the two pairs.
        """
        return self.df_pair1['close'] - self.df_pair2['close']

    def generate_signals(self):
        """
        Generates buy/sell signals based on the spread's deviation from its mean.

        :return: Trading signal based on the spread's current position.
        """
        spread = self.calculate_spread()
        mean_spread = spread.mean()
        std_spread = spread.std()
        current_spread = spread.iloc[-1]

        # Trading signal logic based on spread deviation
        if current_spread > mean_spread + self.threshold * std_spread:
            return "SELL Pair 1 (USD/CAD), BUY Pair 2"
        elif current_spread < mean_spread - self.threshold * std_spread:
            return "BUY Pair 1 (USD/CAD), SELL Pair 2"
        else:
            return "HOLD"

# Example usage
# strategy = StatisticalArbitrageStrategy(df_usdcad, df_eurusd)
# signal = strategy.generate_signals()
# print(signal)
