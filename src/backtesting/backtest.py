import sys
import os
import time
import multiprocessing
import numpy as np  # Import numpy since it's used later
import backtrader as bt
import pandas as pd
from src.algorithms.combined_test import CombinedStrategy
from src.data.db_connection import fetch_historical_data

# Determine the project root by going two levels up from the current file
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Function to process chunks in parallel
def process_chunk(df_chunk):
    """Process a chunk of historical data to prepare it for backtesting."""
    df_chunk['datetime'] = pd.to_datetime(df_chunk['timestamp'])
    df_chunk.set_index('datetime', inplace=True)
    df_chunk.sort_index(inplace=True)

    df_chunk = df_chunk[['open', 'high', 'low', 'close', 'volume']]
    df_chunk['openinterest'] = 0  # Required column for Backtrader

    # Convert to numeric and drop NaNs efficiently
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df_chunk[numeric_columns] = df_chunk[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df_chunk.dropna(inplace=True)

    return df_chunk


def run_backtest(strategy, cash=100000.0, commission=0.0, printlog=False):
    """Run the backtest with optimized performance."""

    # Start profiling
    start_time = time.time()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # Fetch and check data
    df = fetch_historical_data()
    if df.empty:
        print("No data fetched from the database.")
        return

    print("Columns in DataFrame:", df.columns.tolist())

    # Split data into chunks for multiprocessing
    num_cores = max(1, multiprocessing.cpu_count() // 2)  # Use half available cores to prevent overheating
    df_chunks = np.array_split(df, num_cores)

    print(f"Processing data using {num_cores} CPU cores...")

    with multiprocessing.Pool(num_cores) as pool:
        processed_chunks = pool.map(process_chunk, df_chunks)

    # Combine processed chunks
    df = pd.concat(processed_chunks)

    # Verify the data
    print("DataFrame columns after preparation:", df.columns.tolist())
    print(df.head())

    # Create Data Feed
    data = bt.feeds.PandasData(dataname=df)

    # Add data and strategy to Cerebro
    cerebro.adddata(data)
    cerebro.addstrategy(strategy, printlog=printlog)

    # Add analyzers for performance metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # Run backtest
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # Extract and print analysis
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    print(f"Sharpe Ratio (Analyzer): {sharpe.get('sharperatio', 'N/A')}")
    print(f"Max Drawdown: {drawdown.drawdown:.2f}%")
    print(f"Total Trades: {trades.total.closed}")
    print(f"Wins: {trades.won.total}, Losses: {trades.lost.total}")
    win_rate = (trades.won.total / trades.total.closed * 100) if trades.total.closed > 0 else 0
    print(f"Win Rate: {win_rate:.2f}%")

    # Stop profiling
    end_time = time.time()
    print(f"Backtest execution time: {end_time - start_time:.2f} seconds")

    # Plot results
    cerebro.plot()


if __name__ == '__main__':
    run_backtest(CombinedStrategy, printlog=True)
