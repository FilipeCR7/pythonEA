# src/algorithms/combined_test.py

import pandas as pd
import matplotlib.pyplot as plt
from src.algorithms.mean_reversion import MeanReversionStrategy
from src.algorithms.time_series_analysis import TimeSeriesAnalysis
from src.data.db_connection import fetch_historical_data, create_db_connection, close_db_connection
from sqlalchemy import text
import datetime

def assign_signal_value(signal):
    signal_mapping = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
    return signal_mapping.get(signal, 0)  # Default to 0 if signal is unrecognized

def compute_weights(sharpe_ratio_mean_rev, sharpe_ratio_time_series):
    # Handle cases where Sharpe Ratios may be negative or zero
    sharpe_ratios = [max(0, sharpe_ratio_mean_rev), max(0, sharpe_ratio_time_series)]
    total_sharpe = sum(sharpe_ratios)
    if total_sharpe == 0:
        # If both adjusted Sharpe Ratios are zero, assign equal weights
        return 0.5, 0.5
    weight_mean_rev = sharpe_ratios[0] / total_sharpe
    weight_time_series = sharpe_ratios[1] / total_sharpe
    return weight_mean_rev, weight_time_series

def compute_weighted_decision(mean_rev_signal, time_series_signal, weight_mean_rev, weight_time_series):
    mean_rev_signal_value = assign_signal_value(mean_rev_signal)
    time_series_signal_value = assign_signal_value(time_series_signal)
    weighted_score = (mean_rev_signal_value * weight_mean_rev) + (time_series_signal_value * weight_time_series)
    if weighted_score > 0:
        return 'BUY'
    elif weighted_score < 0:
        return 'SELL'
    else:
        return 'HOLD'

def run_combined_test():
    # Fetch historical data
    df = fetch_historical_data()
    if df.empty:
        print("No data fetched from the database.")
        return

    # Initialize both strategies
    mean_rev = MeanReversionStrategy(df)
    time_series = TimeSeriesAnalysis(df)

    # Run Mean Reversion Strategy
    mean_rev_signal = mean_rev.get_signal()
    roi_mean_rev, sharpe_ratio_mean_rev, volatility_score_mean_rev, rsi_value, ma_cross_signal, win_loss_ratio = mean_rev.calculate_performance_metrics()

    # Run Time Series Analysis Strategy
    time_series.fit_arima()
    time_series_signal = time_series.get_signal()
    roi_time_series, sharpe_ratio_time_series, volatility_score_time_series, _, _, _ = time_series.calculate_performance_metrics()
    arima_rmse = time_series.calculate_arima_rmse()

    # Compute Weights Based on Sharpe Ratio
    weight_mean_rev, weight_time_series = compute_weights(sharpe_ratio_mean_rev, sharpe_ratio_time_series)

    # Compute Final Weighted Decision
    final_decision = compute_weighted_decision(mean_rev_signal, time_series_signal, weight_mean_rev, weight_time_series)

    # Insert a single record into the scorecard table
    connection = create_db_connection()
    try:
        query = text("""
            INSERT INTO scorecard (
                timestamp, strategy_name, final_decision,
                mean_reversion_signal, time_series_signal,
                roi, sharpe_ratio, volatility_score, rsi, moving_average_cross, arima_rmse,
                weight_mean_reversion, weight_time_series
            )
            VALUES (
                :timestamp, :strategy_name, :final_decision,
                :mean_reversion_signal, :time_series_signal,
                :roi, :sharpe_ratio, :volatility_score, :rsi, :moving_average_cross, :arima_rmse,
                :weight_mean_reversion, :weight_time_series
            )
        """)
        params = {
            'timestamp': datetime.datetime.now(),
            'strategy_name': 'Combined Strategy',
            'final_decision': final_decision,  # Updated key from 'signal' to 'final_decision'
            'mean_reversion_signal': mean_rev_signal,
            'time_series_signal': time_series_signal,
            'roi': (roi_mean_rev + roi_time_series) / 2,  # Average ROI
            'sharpe_ratio': (sharpe_ratio_mean_rev + sharpe_ratio_time_series) / 2,  # Average Sharpe Ratio
            'volatility_score': (volatility_score_mean_rev + volatility_score_time_series) / 2,
            # Average Volatility Score
            'rsi': rsi_value,
            'moving_average_cross': ma_cross_signal,
            'arima_rmse': arima_rmse,
            'weight_mean_reversion': weight_mean_rev,
            'weight_time_series': weight_time_series
        }
        result = connection.execute(query, params)
        connection.commit()
        print(f"Inserted {result.rowcount} rows into scorecard.")
    except Exception as e:
        connection.rollback()
        print(f"Error inserting scorecard data: {e}")
    finally:
        close_db_connection(connection)

    # Print results
    print(f"Mean Reversion Signal: {mean_rev_signal}")
    print(f"Time Series Signal: {time_series_signal}")
    print(f"Final Decision: {final_decision}")
    print(f"Mean Reversion Sharpe Ratio: {sharpe_ratio_mean_rev}")
    print(f"Time Series Sharpe Ratio: {sharpe_ratio_time_series}")
    print(f"Weights - Mean Reversion: {weight_mean_rev}, Time Series: {weight_time_series}")
    print(f"ROI Mean Reversion: {roi_mean_rev}, ROI Time Series: {roi_time_series}")
    print(f"Volatility Score Mean Reversion: {volatility_score_mean_rev}, Volatility Score Time Series: {volatility_score_time_series}")
    print(f"ARIMA RMSE: {arima_rmse}")
    print(f"RSI: {rsi_value}, Moving Average Cross Signal: {ma_cross_signal}")

    # Plotting the results (same as before)
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Close Price', color='blue')
    mean_rev.calculate_indicators()
    plt.plot(df.index, mean_rev.df[f'sma_{mean_rev.window}'], label=f'SMA {mean_rev.window}', color='orange')
    plt.plot(df.index, mean_rev.df['upper_bound'], label='Upper Bound', color='green')
    plt.plot(df.index, mean_rev.df['lower_bound'], label='Lower Bound', color='red')

    # Annotate Mean Reversion Signals
    if mean_rev_signal == "BUY":
        plt.scatter(df.index[-1], df['close'].iloc[-1], marker='^', color='green', label='Mean Rev BUY')
    elif mean_rev_signal == "SELL":
        plt.scatter(df.index[-1], df['close'].iloc[-1], marker='v', color='red', label='Mean Rev SELL')

    # Annotate Time Series Analysis Signals
    if time_series_signal == "BUY":
        plt.scatter(df.index[-1], df['close'].iloc[-1], marker='^', color='cyan', label='Time Series BUY')
    elif time_series_signal == "SELL":
        plt.scatter(df.index[-1], df['close'].iloc[-1], marker='v', color='magenta', label='Time Series SELL')

    plt.title('Mean Reversion and Time Series Analysis')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_combined_test()
