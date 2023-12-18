# strategies/statistical_arbitrage.py
import numpy as np

class StatisticalArbitrageStrategy:
    def __init__(self, historical_data_pair1, historical_data_pair2, threshold=0.02):
        self.data1 = historical_data_pair1
        self.data2 = historical_data_pair2
        self.threshold = threshold

    def calculate_spread(self):
        prices1 = [float(data[4]) for data in self.data1]  # Close prices are at index 4
        prices2 = [float(data[4]) for data in self.data2]

        return np.array(prices1) - np.array(prices2)

    def generate_signals(self):
        spread = self.calculate_spread()
        mean_spread = np.mean(spread)
        std_spread = np.std(spread)

        current_spread = spread[-1]

        if current_spread > mean_spread + self.threshold * std_spread:
            return "SELL Pair 1 (USD/CAD), BUY Pair 2"
        elif current_spread < mean_spread - self.threshold * std_spread:
            return "BUY Pair 1 (USD/CAD), SELL Pair 2"
        else:
            return "HOLD"

# Example usage in your main script:
# main.py or another appropriate script in your project
from strategies.statistical_arbitrage import StatisticalArbitrageStrategy
# Assuming you have this function to fetch data from your database
from your_data_fetching_module import fetch_historical_data_from_db

def main():
    # Replace 'EUR_USD' with another currency pair that you want to compare with USD/CAD
    historical_data_usdcad = fetch_historical_data_from_db('USD_CAD')
    historical_data_pair2 = fetch_historical_data_from_db('EUR_USD')

    strategy = StatisticalArbitrageStrategy(historical_data_usdcad, historical_data_pair2)
    signal = strategy.generate_signals()
    print(f"Statistical Arbitrage Signal: {signal}")

if __name__ == "__main__":
    main()
