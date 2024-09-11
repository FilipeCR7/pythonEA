from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from data.db_connection import create_db_connection, close_db_connection


class MeanReversionStrategy:
    def __init__(self, df, window=20, std_dev_multiplier=2):
        self.df = df
        self.window = window
        self.std_dev_multiplier = std_dev_multiplier

    def generate_signals(self):
        # Calculate the moving average and upper/lower bands
        self.df[f'sma_{self.window}'] = self.df['close'].rolling(window=self.window).mean()
        self.df['lower_bound'] = self.df[f'sma_{self.window}'] - (
            self.std_dev_multiplier * self.df['close'].rolling(window=self.window).std())
        self.df['upper_bound'] = self.df[f'sma_{self.window}'] + (
            self.std_dev_multiplier * self.df['close'].rolling(window=self.window).std())

        # Get current price and bounds
        current_price = self.df['close'].iloc[-1]
        lower_bound = self.df['lower_bound'].iloc[-1]
        upper_bound = self.df['upper_bound'].iloc[-1]

        # Generate trading signals
        signal = "HOLD"
        if current_price <= lower_bound:
            signal = "BUY"
        elif current_price >= upper_bound:
            signal = "SELL"

        # Calculate performance metrics
        roi = self.calculate_roi()
        sharpe_ratio = self.calculate_sharpe_ratio()
        volatility_score = self.calculate_volatility_score()

        # Insert into the scorecard table
        self.insert_scorecard_data(strategy_name="Mean Reversion", signal=signal, roi=roi,
                                   sharpe_ratio=sharpe_ratio, volatility_score=volatility_score)

        return signal

    def calculate_roi(self):
        initial_price = self.df['close'].iloc[0]
        final_price = self.df['close'].iloc[-1]
        roi = (final_price - initial_price) / initial_price
        return round(roi, 4)

    def calculate_sharpe_ratio(self):
        risk_free_rate = 0.01  # Assuming a 1% annual risk-free return
        returns = self.df['close'].pct_change().dropna()
        excess_returns = returns.mean() - risk_free_rate / 252  # Adjust for daily data
        sharpe_ratio = excess_returns / returns.std() * np.sqrt(252)  # Annualize the Sharpe Ratio
        return round(sharpe_ratio, 4)

    def calculate_volatility_score(self):
        volatility = self.df['close'].pct_change().rolling(window=20).std().mean() * np.sqrt(252)
        return round(volatility, 4)

    def insert_scorecard_data(self, strategy_name, signal, roi, sharpe_ratio, volatility_score):
        connection = create_db_connection()
        query = """
            INSERT INTO scorecard (timestamp, strategy_name, signal, roi, sharpe_ratio, volatility_score)
            VALUES (NOW(), %s, %s, %s, %s, %s)
        """
        try:
            connection.execute(query, (strategy_name, signal, roi, sharpe_ratio, volatility_score))
        except Exception as e:
            print(f"Error inserting scorecard data: {e}")
        finally:
            close_db_connection(connection)


if __name__ == "__main__":
    df = fetch_historical_data()  # Assuming fetch_historical_data is defined elsewhere
    strategy = MeanReversionStrategy(df)
    signal = strategy.generate_signals()
    print(f"Generated signal: {signal}")
