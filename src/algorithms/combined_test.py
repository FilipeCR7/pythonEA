# src/algorithms/combined_test.py

import backtrader as bt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

class CombinedStrategy(bt.Strategy):
    params = (
        ('mean_rev_window', 20),
        ('std_dev_multiplier', 2),
        ('arima_order', (5, 1, 0)),
        ('arima_window_size', 100),
        ('printlog', False),
    )

    def __init__(self):
        # Initialize attributes
        self.returns = []
        self.wins = 0
        self.losses = 0

        # Existing initialization code...
        self.dataclose = self.datas[0].close

        # Mean Reversion Indicators
        self.sma = bt.indicators.SimpleMovingAverage(self.dataclose, period=self.params.mean_rev_window)
        self.std_dev = bt.indicators.StandardDeviation(self.dataclose, period=self.params.mean_rev_window)
        self.lower_band = self.sma - (self.params.std_dev_multiplier * self.std_dev)
        self.upper_band = self.sma + (self.params.std_dev_multiplier * self.std_dev)

        # Time Series Analysis
        self.arima_result = None

        self.order = None
        self.bar_executed = 0  # To track when the last order was executed

    def next(self):
        # Check for pending orders
        if self.order:
            return

        # Mean Reversion Signal
        current_price = self.dataclose[0]
        lower_bound = self.lower_band[0]
        upper_bound = self.upper_band[0]
        mean_rev_signal = 0

        if current_price <= lower_bound:
            mean_rev_signal = 1  # BUY
        elif current_price >= upper_bound:
            mean_rev_signal = -1  # SELL

        # Time Series Signal
        time_series_signal = 0
        if len(self) >= self.params.arima_window_size:
            data = np.array([self.dataclose[-i] for i in range(self.params.arima_window_size, 0, -1)])
            try:
                model = ARIMA(data, order=self.params.arima_order)
                self.arima_result = model.fit()

                forecast = self.arima_result.forecast()[0]
                if forecast > current_price:
                    time_series_signal = 1  # BUY
                elif forecast < current_price:
                    time_series_signal = -1  # SELL
            except Exception as e:
                if self.params.printlog:
                    print(f'ARIMA model error: {e}')
                pass  # Handle exceptions if necessary

        # Combine Signals
        combined_signal = mean_rev_signal + time_series_signal

        # Position Sizing
        account_value = self.broker.getvalue()
        risk_per_trade = 0.01  # Risk 1% per trade
        stop_loss_pips = 50  # Adjust as needed
        pip_value = 0.0001  # Adjust based on the currency pair
        stake = int((account_value * risk_per_trade) / (stop_loss_pips * pip_value))
        stake = max(stake, 1)  # Ensure at least a stake of 1

        if not self.position:
            if combined_signal > 0:
                self.order = self.buy(size=stake, exectype=bt.Order.Market)
                self.bar_executed = len(self)
                if self.params.printlog:
                    print(f'BUY ORDER PLACED at price {current_price:.5f}, size {stake}')
            elif combined_signal < 0:
                self.order = self.sell(size=stake, exectype=bt.Order.Market)
                self.bar_executed = len(self)
                if self.params.printlog:
                    print(f'SELL ORDER PLACED at price {current_price:.5f}, size {stake}')
        else:
            # Wait for at least one bar after entry before exiting
            if len(self) >= (self.bar_executed + 1):
                if self.position.size > 0 and combined_signal <= 0:
                    self.order = self.close()
                    if self.params.printlog:
                        print(f'CLOSE LONG POSITION at price {current_price:.5f}')
                elif self.position.size < 0 and combined_signal >= 0:
                    self.order = self.close()
                    if self.params.printlog:
                        print(f'CLOSE SHORT POSITION at price {current_price:.5f}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.params.printlog:
                    print(f'BUY EXECUTED, Price: {order.executed.price:.5f}, Size: {order.executed.size}')
            elif order.issell():
                if self.params.printlog:
                    print(f'SELL EXECUTED, Price: {order.executed.price:.5f}, Size: {order.executed.size}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.params.printlog:
                print('Order Canceled/Margin/Rejected')

        self.order = None  # Reset orders

    # ... rest of your strategy code ...


    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        profit = trade.pnl
        self.returns.append(profit)

        if profit > 0:
            self.wins += 1
        else:
            self.losses += 1

        if self.params.printlog:
            print(f'TRADE CLOSED. PROFIT: {profit:.2f}')

    def calculate_performance_metrics(self):
        roi = (self.broker.get_value() / self.broker.startingcash) - 1

        if len(self.returns) > 1:
            returns = np.array(self.returns)
            risk_free_rate = 0.01 / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            mean_excess_return = np.mean(excess_returns)
            std_excess_return = np.std(excess_returns)
            sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252) if std_excess_return != 0 else 0.0
        else:
            sharpe_ratio = 0.0

        if len(self.dataclose) > 1:
            # Get daily returns
            prices = np.array([self.dataclose[-i] for i in range(len(self.dataclose), 0, -1)])
            daily_returns = pd.Series(prices).pct_change().dropna()
            volatility_score = daily_returns.rolling(window=20).std().mean() * np.sqrt(252)
        else:
            volatility_score = 0.0

        rsi_value = self.rsi[0]

        if self.ma_short[0] > self.ma_long[0]:
            ma_cross_signal = 'BUY'
        elif self.ma_short[0] < self.ma_long[0]:
            ma_cross_signal = 'SELL'
        else:
            ma_cross_signal = 'HOLD'

        if self.arima_result:
            try:
                predictions = self.arima_result.predict(start=0, end=self.params.arima_window_size - 1)
                data = np.array([self.dataclose[-i] for i in range(self.params.arima_window_size, 0, -1)])
                actuals = data
                rmse = np.sqrt(((actuals - predictions) ** 2).mean())
                arima_rmse = float(round(rmse, 4))
            except Exception:
                arima_rmse = None
        else:
            arima_rmse = None

        total_trades = self.wins + self.losses
        win_loss_ratio = self.wins / total_trades if total_trades > 0 else None

        return {
            'roi': round(roi, 4),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'volatility_score': round(volatility_score, 4),
            'rsi_value': round(rsi_value, 2),
            'ma_cross_signal': ma_cross_signal,
            'arima_rmse': arima_rmse,
            'win_loss_ratio': win_loss_ratio,
        }

    def stop(self):
        metrics = self.calculate_performance_metrics()
        print(f'ROI: {metrics["roi"]}, Sharpe Ratio: {metrics["sharpe_ratio"]}, '
              f'Volatility Score: {metrics["volatility_score"]}, RSI: {metrics["rsi_value"]}, '
              f'MA Cross Signal: {metrics["ma_cross_signal"]}, ARIMA RMSE: {metrics["arima_rmse"]}, '
              f'Win/Loss Ratio: {metrics["win_loss_ratio"]}')
