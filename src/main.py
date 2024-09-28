import sys
import os
import pandas as pd
import numpy as np  # Ensure numpy is imported as np
import warnings
import logging

# Ensure the 'src' directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from src.algorithms.mean_reversion import MeanReversionStrategy
    from src.algorithms.time_series_analysis import TimeSeriesAnalysis
    from src.data.db_connection import fetch_historical_data, create_db_connection, close_db_connection
    from sqlalchemy import text
except ImportError as e:
    print(f"Error importing module: {e}")
    sys.exit(1)


def make_final_decision(mean_reversion_signal, time_series_signal, sharpe_ratio_mean_rev, sharpe_ratio_time_series):
    signal_mapping = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
    # Assign numerical values to signals
    mean_rev_value = signal_mapping.get(mean_reversion_signal, 0)
    time_series_value = signal_mapping.get(time_series_signal, 0)

    # Adjust Sharpe Ratios to handle negative values
    adjusted_sharpe_mean_rev = max(0.0001, sharpe_ratio_mean_rev)
    adjusted_sharpe_time_series = max(0.0001, sharpe_ratio_time_series)

    # Compute weights based on adjusted Sharpe Ratios
    total_sharpe = adjusted_sharpe_mean_rev + adjusted_sharpe_time_series
    weight_mean_rev = adjusted_sharpe_mean_rev / total_sharpe
    weight_time_series = adjusted_sharpe_time_series / total_sharpe

    # Compute weighted signal
    weighted_signal = (mean_rev_value * weight_mean_rev) + (time_series_value * weight_time_series)

    # Determine final decision
    if weighted_signal > 0.5:
        final_decision = 'BUY'
    elif weighted_signal < -0.5:
        final_decision = 'SELL'
    else:
        final_decision = 'HOLD'

    # Return the final decision and weights
    return final_decision, weight_mean_rev, weight_time_series


def insert_scorecard_data(data):
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
        # Convert pandas.Timestamp to datetime.datetime
        params = {
            'timestamp': pd.Timestamp.now().to_pydatetime(),
            **data
        }
        # Convert numpy data types to native Python types
        for key, value in params.items():
            if isinstance(value, np.generic):
                params[key] = value.item()
        connection.execute(query, params)
        connection.commit()
        print("Data inserted into scorecard successfully.")
    except Exception as e:
        connection.rollback()
        print(f"Error inserting data into scorecard: {e}")
    finally:
        close_db_connection(connection)


def run_combined_strategy():
    # Suppress specific warnings
    warnings.filterwarnings('ignore',
                            message=".*A date index has been provided, but it has no associated frequency information.*")
    warnings.filterwarnings('ignore', message=".*y is poorly scaled, which may affect convergence of the optimizer.*")

    # Adjust logging level
    logging.getLogger('statsmodels').setLevel(logging.ERROR)
    logging.getLogger('arch').setLevel(logging.ERROR)

    # Fetch data from the database
    df = fetch_historical_data()

    if df is None or df.empty:
        print("No data fetched from the database.")
        return

    # Data cleaning and type conversion
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])

    # Ensure 'timestamp' column is datetime and set as index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df.set_index('timestamp', inplace=True)
    else:
        print("Timestamp column not found in the data.")
        return

    # Sort the DataFrame by index
    df.sort_index(inplace=True)

    # Mean Reversion Strategy
    mean_reversion_strategy = MeanReversionStrategy(df)
    mean_reversion_signal = mean_reversion_strategy.get_signal()
    roi_mean_rev, sharpe_ratio_mean_rev, volatility_score_mean_rev, rsi_value, ma_cross_signal, win_loss_ratio_value = mean_reversion_strategy.calculate_performance_metrics()

    # Time Series Analysis
    time_series_analysis = TimeSeriesAnalysis(df)
    # Fit ARIMA model
    try:
        time_series_analysis.fit_arima()
    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
    # Get signal from Time Series Analysis
    time_series_signal = time_series_analysis.get_signal()
    roi_time_series, sharpe_ratio_time_series, volatility_score_time_series, _, _, _ = time_series_analysis.calculate_performance_metrics()
    arima_rmse = time_series_analysis.calculate_arima_rmse()

    # Make the final decision and get weights
    final_decision, weight_mean_rev, weight_time_series = make_final_decision(
        mean_reversion_signal,
        time_series_signal,
        sharpe_ratio_mean_rev,
        sharpe_ratio_time_series
    )

    # Collect data for insertion
    scorecard_data = {
        'strategy_name': 'Combined Strategy',
        'final_decision': final_decision,
        'mean_reversion_signal': mean_reversion_signal,
        'time_series_signal': time_series_signal,
        'roi': float((roi_mean_rev + roi_time_series) / 2),  # Ensure it's a float
        'sharpe_ratio': float((sharpe_ratio_mean_rev + sharpe_ratio_time_series) / 2),  # Ensure it's a float
        'volatility_score': float((volatility_score_mean_rev + volatility_score_time_series) / 2),  # Ensure it's a float
        'rsi': float(rsi_value),
        'moving_average_cross': ma_cross_signal,
        'arima_rmse': float(arima_rmse),
        'weight_mean_reversion': float(round(weight_mean_rev, 4)),
        'weight_time_series': float(round(weight_time_series, 4))
    }

    # Insert data into the scorecard table
    insert_scorecard_data(scorecard_data)

    # Print results for verification
    print(f"Mean Reversion Signal: {mean_reversion_signal}")
    print(f"Time Series Signal: {time_series_signal}")
    print(f"Final Decision: {final_decision}")
    print(f"Mean Reversion Sharpe Ratio: {sharpe_ratio_mean_rev}")
    print(f"Time Series Sharpe Ratio: {sharpe_ratio_time_series}")
    print(f"ROI Mean Reversion: {roi_mean_rev}, ROI Time Series: {roi_time_series}")
    print(f"Volatility Mean Reversion: {volatility_score_mean_rev}, Volatility Time Series: {volatility_score_time_series}")
    print(f"ARIMA RMSE: {arima_rmse}")
    print(f"RSI: {rsi_value}, Moving Average Cross Signal: {ma_cross_signal}")


if __name__ == "__main__":
    run_combined_strategy()
