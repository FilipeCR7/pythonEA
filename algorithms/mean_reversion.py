import numpy as np

class MeanReversionStrategy:
    def __init__(self, historical_data, window=20, std_dev_multiplier=2):
        """
        :param historical_data: List of tuples containing historical price data
        :param window: The window size for calculating the moving average
        :param std_dev_multiplier: Multiplier for standard deviation to identify significant deviations
        """
        self.historical_data = historical_data
        self.window = window
        self.std_dev_multiplier = std_dev_multiplier

    def calculate_moving_average(self):
        """Calculate moving average and standard deviation."""
        prices = [data[4] for data in self.historical_data]  # Assuming close prices are at index 4
        prices = np.array(prices, dtype=np.float64)
        moving_avg = np.convolve(prices, np.ones(self.window), 'valid') / self.window
        std_dev = np.std(prices[-len(moving_avg):])
        return moving_avg, std_dev

    def generate_signals(self):
        """Generate buy and sell signals based on mean reversion logic."""
        moving_avg, std_dev = self.calculate_moving_average()
        current_price = self.historical_data[-1][4]  # Latest close price

        lower_bound = moving_avg[-1] - (self.std_dev_multiplier * std_dev)
        upper_bound = moving_avg[-1] + (self.std_dev_multiplier * std_dev)

        if current_price <= lower_bound:
            return "BUY"  # Price is significantly lower than the mean
        elif current_price >= upper_bound:
            return "SELL"  # Price is significantly higher than the mean
        else:
            return "HOLD"  # Price is within the normal range

# Example usage:
# historical_data = fetch_historical_data_from_db()  # Function to fetch data from your DB
# strategy = MeanReversionStrategy(historical_data)
# signal = strategy.generate_signals()
# print(signal)