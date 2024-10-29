from src.backtesting import backtest as bt
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
        ('short_window', 20),
        ('long_window', 50),
        ('printlog', False),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close

        # Mean Reversion Indicators
        self.sma = bt.indicators.SimpleMovingAverage(self.dataclose, period=self.params.mean_rev_window)
        self.std_dev = bt.indicators.StandardDeviation(self.dataclose, period=self.params.mean_rev_window)
        self.lower_band = self.sma - (self.params.std_dev_multiplier * self.std_dev)
        self.upper_band = self.sma + (self.params.std_dev_multiplier * self.std_dev)

        # RSI Indicator
        self.rsi = bt.indicators.RelativeStrengthIndex(self.dataclose, period=14)
        # Moving Averages for MA Cross Signal
        self.ma_short = bt.indicators.SimpleMovingAverage(self.dataclose, period=self.params.short_window)
        self.ma_long = bt.indicators.SimpleMovingAverage(self.dataclose, period=self.params.long_window)

        # Time Series Analysis
        self.arima_result = None

        self.order = None
        self.returns = []
        self.wins = 0
        self.losses = 0

        # Suppress convergence warnings
        warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")

    def next(self):
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

        if not self.position:
            if combined_signal > 0:
                self.order = self.buy()
                if self.params.printlog:
                    print(f'BUY ORDER PLACED at price {current_price:.5f}')
            elif combined_signal < 0:
                self.order = self.sell()
                if self.params.printlog:
                    print(f'SELL ORDER PLACED at price {current_price:.5f}')
        else:
            if self.position.size > 0 and combined_signal <= 0:
                self.order = self.close()
                if self.params.printlog:
                    print(f'CLOSE LONG POSITION at price {current_price:.5f}')
            elif self.position.size < 0 and combined_signal >= 0:
                self.order = self.close()
                if self.params.printlog:
                    print(f'CLOSE SHORT POSITION at price {current_price:.5f}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order is active

        if order.status in [order.Completed]:
            if self.params.printlog:
                if order.isbuy():
                    print(f'BUY EXECUTED, Price: {order.executed.price:.5f}')
                elif order.issell():
                    print(f'SELL EXECUTED, Price: {order.executed.price:.5f}')
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.params.printlog:
                print('Order Canceled/Margin/Rejected')

        self.order = None  # Reset order

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
