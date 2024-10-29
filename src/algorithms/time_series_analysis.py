from src.backtesting import backtest as bt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

class TimeSeriesStrategy(bt.Strategy):
    params = (
        ('arima_order', (5, 1, 0)),
        ('window_size', 100),
        ('printlog', False),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
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

        if len(self) >= self.params.window_size:
            # Get historical close prices
            data = np.array(self.dataclose.get(size=self.params.window_size))
            try:
                # Fit ARIMA model
                model = ARIMA(data, order=self.params.arima_order)
                self.arima_result = model.fit()

                # Forecast the next value
                forecast = self.arima_result.forecast()[0]
                current_price = self.dataclose[0]

                if not self.position:
                    if forecast > current_price:
                        self.order = self.buy()
                    elif forecast < current_price:
                        self.order = self.sell()
                else:
                    # Close positions based on forecast
                    if self.position.size > 0 and forecast < current_price:
                        self.order = self.close()
                    elif self.position.size < 0 and forecast > current_price:
                        self.order = self.close()
            except Exception as e:
                pass  # Handle exceptions if needed

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order submitted/accepted to/by broker

        if order.status in [order.Completed]:
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass  # Handle order cancellation/margin/rejection if needed

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

    def stop(self):
        # Calculate ROI
        roi = (self.broker.get_value() / self.broker.startingcash) - 1

        # Calculate Sharpe Ratio
        if len(self.returns) > 1:
            returns = np.array(self.returns)
            risk_free_rate = 0.01 / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            mean_excess_return = np.mean(excess_returns)
            std_excess_return = np.std(excess_returns)
            sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252) if std_excess_return != 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Calculate Volatility Score
        if len(self.dataclose) > 1:
            daily_returns = np.diff(self.dataclose.get(size=len(self.dataclose))) / self.dataclose.get(size=len(self.dataclose)-1)
            volatility_score = np.std(daily_returns) * np.sqrt(252)
        else:
            volatility_score = 0.0

        # Calculate ARIMA RMSE
        arima_rmse = None
        if self.arima_result:
            try:
                predictions = self.arima_result.predict(start=0, end=self.params.window_size - 1)
                actuals = self.dataclose.get(size=self.params.window_size)
                rmse = np.sqrt(((actuals - predictions) ** 2).mean())
                arima_rmse = float(round(rmse, 4))
            except Exception:
                arima_rmse = None

        # Log performance metrics
        print(f'ROI: {round(roi, 4)}, Sharpe Ratio: {round(sharpe_ratio, 4)}, '
              f'Volatility Score: {round(volatility_score, 4)}, ARIMA RMSE: {arima_rmse}')
