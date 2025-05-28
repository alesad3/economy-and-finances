"""
LSTM-Based Trading Strategy Backtesting
=======================================
This script implements and backtests a trading strategy based on LSTM predictions for stock market data.
Features:
- Downloads historical stock data using yfinance
- Builds and trains an LSTM model for price prediction
- Implements trading strategies based on LSTM predictions
- Backtests strategies using the backtesting.py library
- Compares LSTM strategy performance against a simple moving average strategy
- Visualizes backtest results and performance metrics
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime, timedelta
import os
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from backtesting.lib import crossover

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class StockPredictor:
    def __init__(self, symbol, start_date=None, end_date=None, test_start_date=None):
        """
        Initialize the stock predictor.

        Args:
            symbol (str): Stock ticker symbol
            start_date (str): Start date for data in YYYY-MM-DD format
            end_date (str): End date for data in YYYY-MM-DD format
            test_start_date (str): Start date for test data in YYYY-MM-DD format
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.test_start_date = test_start_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        self.data = None
        self.train_data = None
        self.test_data = None
        self.seq_length = 20

    def load_data(self):
        """
        Download and prepare the stock data.

        Returns:
            tuple: train_hist_data, test_hist_data (pandas DataFrames)
        """
        ticker = yf.Ticker(self.symbol)

        hist_data = ticker.history(start=self.start_date, end=self.end_date)

        # Split into train and test based on test_start_date
        train_hist_data = hist_data[hist_data.index < self.test_start_date]
        test_hist_data = hist_data[hist_data.index >= self.test_start_date]

        self.train_data = train_hist_data[["Close"]].copy().reset_index()
        self.test_data = test_hist_data[["Close"]].copy().reset_index()

        self.train_hist_data = train_hist_data
        self.test_hist_data = test_hist_data

        print(f"Training data: {len(train_hist_data)} days ({train_hist_data.index[0]} to {train_hist_data.index[-1]})")
        print(f"Test data: {len(test_hist_data)} days ({test_hist_data.index[0]} to {test_hist_data.index[-1]})")

        return train_hist_data, test_hist_data

    def preprocess_data(self):
        """
        Scale the data and create sequences.

        Returns:
            tuple: X_train, y_train, X_test, y_test, train_scaled, test_scaled
        """
        # Scale the data
        train_scaled = self.scaler.fit_transform(self.train_data[["Close"]])
        test_scaled = self.scaler.transform(self.test_data[["Close"]])

        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled, self.seq_length)
        X_test, y_test = self.create_sequences(test_scaled, self.seq_length)

        return X_train, y_train, X_test, y_test, train_scaled, test_scaled

    def create_sequences(self, data, seq_length):
        """
        Create sequences for LSTM model.

        Args:
            data (numpy.ndarray): Scaled data
            seq_length (int): Sequence length for prediction

        Returns:
            tuple: X and y arrays
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    def build_model(self, X_train, use_bidirectional=False):
        """
        Build the LSTM model architecture.

        Args:
            X_train (numpy.ndarray): Training data to determine input shape
            use_bidirectional (bool): Whether to use bidirectional LSTM layers

        Returns:
            tensorflow.keras.models.Sequential: Compiled LSTM model
        """
        if use_bidirectional:
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True, recurrent_regularizer=l2(0.01)),
                              input_shape=(X_train.shape[1], 1)),
                Dropout(0.3),
                Bidirectional(LSTM(64, recurrent_regularizer=l2(0.01))),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1)
            ])
        else:
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1),
                     recurrent_regularizer=l2(0.01)),
                Dropout(0.3),
                LSTM(64, recurrent_regularizer=l2(0.01)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1)
            ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')

        return model

    def train(self, epochs=50, batch_size=16, validation_split=0.2, use_bidirectional=False):
        """
        Train the LSTM model.

        Args:
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size
            validation_split (float): Portion of training data to use for validation
            use_bidirectional (bool): Whether to use bidirectional LSTM layers

        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        # Preprocess data
        X_train, y_train, X_test, y_test, train_scaled, test_scaled = self.preprocess_data()

        # Build model
        self.model = self.build_model(X_train, use_bidirectional)

        # Callbacks for training
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            min_delta=0.0001
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )

        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Store data for evaluation
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.train_scaled, self.test_scaled = train_scaled, test_scaled

        return self.history

    def evaluate(self):
        """
        Evaluate the model performance.

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Make predictions
        train_predict = self.model.predict(self.X_train)
        test_predict = self.model.predict(self.X_test)

        # Invert predictions back to original scale
        train_predict = self.scaler.inverse_transform(train_predict)
        y_train_inv = self.scaler.inverse_transform(self.y_train)
        test_predict = self.scaler.inverse_transform(test_predict)
        y_test_inv = self.scaler.inverse_transform(self.y_test)

        # Calculate metrics
        train_mse = mean_squared_error(y_train_inv, train_predict)
        test_mse = mean_squared_error(y_test_inv, test_predict)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mape = np.mean(np.abs((y_train_inv - train_predict) / y_train_inv)) * 100
        test_mape = np.mean(np.abs((y_test_inv - test_predict) / y_test_inv)) * 100

        print(f"Train MSE: {train_mse:.2f}, RMSE: {train_rmse:.2f}, MAPE: {train_mape:.2f}%")
        print(f"Test MSE: {test_mse:.2f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%")

        # Calculate baseline metrics (naive forecast - previous day's value)
        baseline_predictions = y_test_inv[:-1]  # Previous day's actual value
        baseline_targets = y_test_inv[1:]  # Current day's actual value
        baseline_mse = mean_squared_error(baseline_targets, baseline_predictions)
        baseline_rmse = np.sqrt(baseline_mse)
        baseline_mape = np.mean(np.abs((baseline_targets - baseline_predictions) / baseline_targets)) * 100

        print(f"\nBaseline (Naive - tomorrow is same as today for original test data):")
        print(f"Baseline MSE: {baseline_mse:.2f}, RMSE: {baseline_rmse:.2f}, MAPE: {baseline_mape:.2f}%")

        metrics = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'baseline_mse': baseline_mse,
            'baseline_rmse': baseline_rmse,
            'baseline_mape': baseline_mape,
            'train_predict': train_predict,
            'test_predict': test_predict,
            'y_train_inv': y_train_inv,
            'y_test_inv': y_test_inv
        }

        return metrics


    def plot_training_history(self):
        """Plot the training history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history["loss"], label="Training Loss")
        plt.plot(self.history.history["val_loss"], label="Validation Loss")
        plt.title(f"{self.symbol} - Model Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def predict_future_sequence(self, days_ahead=5, last_sequence=None):
        """
        Predict future prices based on the last sequence.

        Args:
            days_ahead (int): Number of days to predict ahead
            last_sequence (numpy.ndarray): Last sequence of prices, if None, use the last available sequence

        Returns:
            numpy.ndarray: Predicted prices
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        if last_sequence is None:
            # Get the last sequence from the data
            last_sequence = self.test_data["Close"].values[-self.seq_length:]

        # Scale the sequence
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))

        # Initialize predictions list
        future_predictions = []
        temp_data = last_sequence_scaled.copy()

        # Predict one day at a time, updating the sequence each time
        for _ in range(days_ahead):
            X = temp_data[-self.seq_length:].reshape(1, self.seq_length, 1)
            next_pred = self.model.predict(X, verbose=0)[0, 0]
            future_predictions.append(next_pred)
            temp_data = np.vstack([temp_data[1:], next_pred])

        # Convert predictions back to original scale
        future_df = pd.DataFrame(np.array(future_predictions).reshape(-1, 1), columns=["Close"])
        future_prices = self.scaler.inverse_transform(future_df)

        return future_prices.flatten()


class SmaCross(Strategy):
    """
    Simple Moving Average Crossover Strategy.

    This strategy buys when the fast moving average crosses above the slow moving average,
    and sells when the fast moving average crosses below the slow moving average.
    """
    # Define the parameters that can be optimized
    fast = 2
    slow = 5

    def init(self):
        # Calculate the moving averages
        price = self.data.Close
        self.ma1 = self.I(SMA, price, self.fast)
        self.ma2 = self.I(SMA, price, self.slow)

    def next(self):
        # If fast MA crosses above slow MA, buy
        if crossover(self.ma1, self.ma2):
            self.buy()
        # If fast MA crosses below slow MA, sell
        elif crossover(self.ma2, self.ma1):
            self.sell()


class LSTMStrategy(Strategy):
    """
    LSTM-based Trading Strategy.

    This strategy uses LSTM predictions to generate buy/sell signals based on
    predicted price movements at different time horizons.
    """
    # Strategy parameters that can be optimized
    tomorrow_threshold = 0.3
    week_avg_threshold = 0.5
    days20_avg_threshold = 0.8
    tomorrow_sell_threshold = -0.2
    week_avg_sell_threshold = -0.4
    days20_avg_sell_threshold = -0.6

    def init(self):
        # Initialize indicators
        price = self.data.Close
        self.sma20 = self.I(SMA, price, 20)

        # Store external objects
        self.model = None
        self.scaler = None
        self.seq_length = 20

        # Create indicators for visualization
        self.predictions = self.I(lambda: np.full(len(self.data), np.nan))
        self.buy_signals = self.I(lambda: np.zeros(len(self.data)))
        self.sell_signals = self.I(lambda: np.zeros(len(self.data)))

    def set_model(self, model, scaler, seq_length=20):
        """Set the trained model and scaler."""
        self.model = model
        self.scaler = scaler
        self.seq_length = seq_length

    def get_prediction(self, days_ahead=5):
        """
        Get predictions for future prices.

        Args:
            days_ahead (int): Number of days to predict ahead

        Returns:
            tuple: (short-term predictions, 20-day predictions)
        """
        if np.isnan(self.sma20[-1]) or self.model is None or self.scaler is None:
            return None

        # Get recent prices
        recent_prices = np.array([self.data.Close[-i] for i in range(self.seq_length, 0, -1)])
        recent_df = pd.DataFrame(recent_prices, columns=["Close"])
        scaled_data = self.scaler.transform(recent_df)

        # Predict future days
        future_predictions = []
        temp_data = scaled_data.copy()

        for _ in range(days_ahead):
            X = temp_data[-self.seq_length:].reshape(1, self.seq_length, 1)
            next_pred = self.model.predict(X, verbose=0)[0, 0]
            future_predictions.append(next_pred)
            temp_data = np.vstack([temp_data, next_pred])

        future_df = pd.DataFrame(np.array(future_predictions).reshape(-1, 1), columns=["Close"])
        future_prices = self.scaler.inverse_transform(future_df)

        # Continue prediction to get 20-day forecast if needed
        future_20_predictions = []
        if days_ahead < 20:
            days_to_predict = 20 - days_ahead
            for _ in range(days_to_predict):
                X = temp_data[-self.seq_length:].reshape(1, self.seq_length, 1)
                next_pred = self.model.predict(X, verbose=0)[0, 0]
                future_20_predictions.append(next_pred)
                temp_data = np.vstack([temp_data, next_pred])

        future_20_df = pd.DataFrame(np.array(future_20_predictions).reshape(-1, 1), columns=["Close"])
        future_20_prices = self.scaler.inverse_transform(future_20_df)
        future_20_prices = np.append(future_prices.flatten(), future_20_prices.flatten())

        return future_prices.flatten(), future_20_prices

    def analyze_trend(self, predictions, future_20_prices):
        """
        Analyze price trends at different time horizons.

        Args:
            predictions (numpy.ndarray): Short-term predictions
            future_20_prices (numpy.ndarray): 20-day predictions

        Returns:
            dict: Percentage changes at different time horizons
        """
        current_price = self.data.Close[-1]

        tomorrow_change = (predictions[0] - current_price) / current_price * 100
        week_avg_change = (sum(predictions) / len(predictions) - current_price) / current_price * 100
        days20_avg_change = (sum(future_20_prices) / len(future_20_prices) - current_price) / current_price * 100

        return {
            'tomorrow_change': tomorrow_change,
            'week_avg_change': week_avg_change,
            'days20_avg_change': days20_avg_change
        }

    def next(self):
        """Execute trading logic for the current bar."""
        # Skip if we don't have enough data yet
        if np.isnan(self.sma20[-1]) or self.model is None or self.scaler is None:
            return

        # Get current price and predictions
        current_price = self.data.Close[-1]
        prediction_result = self.get_prediction(days_ahead=5)

        if prediction_result is None:
            return

        preds, future_20_prices = prediction_result

        # Analyze trends at different time horizons
        trend = self.analyze_trend(preds, future_20_prices)

        # Store the next day's prediction for visualization
        self.predictions[-1] = preds[0]

        # Buy signal logic
        if (trend['tomorrow_change'] > self.tomorrow_threshold or
                trend['week_avg_change'] > self.week_avg_threshold or
                trend['days20_avg_change'] > self.days20_avg_threshold):

            self.buy_signals[-1] = 1
            self.buy()

        # Sell signal logic
        elif (trend['tomorrow_change'] < self.tomorrow_sell_threshold or
              trend['week_avg_change'] < self.week_avg_sell_threshold or
              trend['days20_avg_change'] < self.days20_avg_sell_threshold):

            self.sell_signals[-1] = 1
            self.sell()


def compare_strategies(data, lstm_model, scaler, seq_length=20, commission=0.002, optimize=True):
    """
    Compare SMA and LSTM strategies.

    Args:
        data (pandas.DataFrame): Historical price data
        lstm_model: Trained LSTM model
        scaler: Fitted scaler
        seq_length (int): Sequence length used for prediction
        commission (float): Commission rate
        optimize (bool): Whether to optimize strategy parameters

    Returns:
        tuple: (sma_stats, lstm_stats)
    """
    sma_bt = Backtest(data, SmaCross, commission=commission, exclusive_orders=True)
    sma_stats = sma_bt.run()
    sma_bt.plot()

    class CustomLSTMStrategy(LSTMStrategy):
        def init(self):
            super().init()
            self.set_model(lstm_model, scaler, seq_length)

    lstm_bt = Backtest(data, CustomLSTMStrategy, commission=commission, exclusive_orders=True)
    lstm_stats = lstm_bt.run()
    lstm_bt.plot()

    # Compare strategies
    print("\n--- SMA Strategy Performance ---")
    print(f"Return: {sma_stats['Return [%]']:.2f}%")
    print(f"Max Drawdown: {sma_stats['Max. Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio: {sma_stats['Sharpe Ratio']:.2f}")
    print(f"Win Rate: {sma_stats['Win Rate [%]']:.2f}%")
    print(f"# Trades: {sma_stats['# Trades']}")

    print("\n--- LSTM Strategy Performance ---")
    print(f"Return: {lstm_stats['Return [%]']:.2f}%")
    print(f"Max Drawdown: {lstm_stats['Max. Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio: {lstm_stats['Sharpe Ratio']:.2f}")
    print(f"Win Rate: {lstm_stats['Win Rate [%]']:.2f}%")
    print(f"# Trades: {lstm_stats['# Trades']}")

    return sma_stats, lstm_stats


def main():
    """Main function to run the backtest."""
    symbol = "MSFT"

    # Set date ranges
    start_date = "2022-01-01"
    end_date = "2025-05-15"
    test_start_date = "2025-01-02"

    print(f"\nAnalyzing {symbol} from {start_date} to {end_date}")
    print(f"Training period: {start_date} to {test_start_date}")
    print(f"Testing period: {test_start_date} to {end_date}")

    # Create stock predictor and load data
    predictor = StockPredictor(symbol, start_date, end_date, test_start_date)
    train_hist_data, test_hist_data = predictor.load_data()

    # Train model
    print("\nTraining LSTM model...")
    predictor.train(use_bidirectional=False)

    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = predictor.evaluate()

    # Plot training history
    predictor.plot_training_history()

    # Backtest strategies
    print("\n" + "=" * 50)
    print("BACKTESTING TRADING STRATEGIES")
    print("=" * 50)

    print("\nRunning strategy comparison...")
    print("This will compare a simple SMA crossover strategy with the LSTM-based strategy")

    # Compare SMA and LSTM strategies
    sma_stats, lstm_stats = compare_strategies(
        test_hist_data,
        predictor.model,
        predictor.scaler,
        predictor.seq_length,
        commission=0.002
    )

    # Print detailed performance comparison
    print("\n" + "=" * 50)
    print("DETAILED STRATEGY COMPARISON")
    print("=" * 50)

    # Create a comparison DataFrame
    metrics_to_compare = ['Return [%]', 'Max. Drawdown [%]', 'Sharpe Ratio',
                          'Sortino Ratio', 'Calmar Ratio', 'Win Rate [%]', '# Trades',
                          'Avg. Trade [%]', 'Exposure [%]']

    comparison_data = {}
    for metric in metrics_to_compare:
        try:
            comparison_data[metric] = [sma_stats[metric], lstm_stats[metric]]
        except KeyError:
            continue

    comparison_df = pd.DataFrame(comparison_data, index=['SMA Strategy', 'LSTM Strategy'])
    print(comparison_df)

    # Generate future predictions
    days_ahead = int(input("\nHow many days do you want to predict into the future? (default: 30): ") or "30")
    future_prices = predictor.predict_future_sequence(days_ahead)

    # Display future predictions
    last_date = test_hist_data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)

    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': future_prices
    })

    print("\nFuture Price Predictions:")
    print(future_df)

    # Visualize future predictions
    plt.figure(figsize=(14, 7))

    # Plot historical data
    plt.plot(test_hist_data.index, test_hist_data['Close'], label='Historical Prices', color='blue')

    # Plot future predictions
    plt.plot(future_dates, future_prices, label=f'Future Predictions ({days_ahead} days)',
             color='red', linestyle='--', marker='o', markersize=4)

    # Add a vertical line to separate historical and prediction data
    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
    plt.text(
        last_date,
        plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),
        ' History | Future ',
        rotation=90,
        verticalalignment='bottom',
    )

    plt.title(f'{symbol} - Historical Data and Future Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return predictor, sma_stats, lstm_stats, future_df


if __name__ == '__main__':
    main()

