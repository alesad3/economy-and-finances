"""
Stock Price Prediction using LSTM Neural Networks
================================================
This script predicts stock closing prices using LSTM (Long Short-Term Memory) neural networks.
Features:
- Downloads historical stock data using yfinance library
- Preprocesses and scales data for LSTM training
- Implements a model with two LSTM layers
- Evaluates model performance against a naive baseline (next day closing value is the same as day before)
- Visualizes predictions and model training metrics
- Provides functionality to predict future prices
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional

tf.random.set_seed(42)
np.random.seed(42)

class StockPredictor:
    def __init__(self, symbol, data_period="2y", test_size=45, seq_length=30):
        """
        Initialize the stock predictor.

        Args:
            symbol (str): Stock ticker symbol
            data_period (str): Period for historical data (e.g., "2y" for 2 years)
            test_size (int): Number of days for testing data
            seq_length (int): Number of previous days to use for prediction
        """
        self.symbol = symbol
        self.data_period = data_period
        self.test_size = test_size
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        self.data = None
        self.train_data = None
        self.test_data = None

    def load_data(self):
        """Download and prepare the stock data."""
        ticker = yf.Ticker(self.symbol)
        hist_data = ticker.history(period=self.data_period)

        self.data = hist_data[["Close"]].copy()
        self.data.reset_index(inplace=True)

        self.train_data = self.data.iloc[:-self.test_size]
        self.test_data = self.data.iloc[-self.test_size:]

        print(f"Training data: {self.train_data.shape[0]} days")
        print(f"Test data: {self.test_data.shape[0]} days")
        print(f"Test period: {self.test_data['Date'].min()} to {self.test_data['Date'].max()}")

        return self.data

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

    def preprocess_data(self):
        """Scale the data and create sequences."""
        train_scaled = self.scaler.fit_transform(self.train_data[["Close"]])
        test_scaled = self.scaler.transform(self.test_data[["Close"]])

        X_train, y_train = self.create_sequences(train_scaled, self.seq_length)
        X_test, y_test = self.create_sequences(test_scaled, self.seq_length)

        return X_train, y_train, X_test, y_test, train_scaled, test_scaled

    def build_model(self, X_train):
        """
        Build the LSTM model architecture.

        Args:
            X_train (numpy.ndarray): Training data to determine input shape

        Returns:
            tensorflow.keras.models.Sequential: Compiled LSTM model
        """
        model = Sequential([
          Bidirectional(LSTM(50, return_sequences=True), input_shape=(X_train.shape[1], 1)),
          Dropout(0.2),
          Bidirectional(LSTM(50)),
          Dropout(0.2),
          Dense(1)
        ])

        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def train(self, epochs=50, batch_size=32, validation_split=0.2, patience=10):
        """
        Train the LSTM model.

        Args:
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size
            validation_split (float): Portion of training data to use for validation
            patience (int): Patience for early stopping

        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        X_train, y_train, X_test, y_test, train_scaled, test_scaled = self.preprocess_data()

        self.model = self.build_model(X_train)

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )

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
        train_predict = self.model.predict(self.X_train)
        test_predict = self.model.predict(self.X_test)

        train_predict = self.scaler.inverse_transform(train_predict)
        y_train_inv = self.scaler.inverse_transform(self.y_train)
        test_predict = self.scaler.inverse_transform(test_predict)
        y_test_inv = self.scaler.inverse_transform(self.y_test)

        train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))
        train_mape = np.mean(np.abs((y_train_inv - train_predict) / y_train_inv)) * 100
        test_mape = np.mean(np.abs((y_test_inv - test_predict) / y_test_inv)) * 100

        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Train MAPE: {train_mape:.2f}%")
        print(f"Test MAPE: {test_mape:.2f}%")

        baseline_predictions = y_test_inv[:-1]
        baseline_targets = y_test_inv[1:]
        baseline_rmse = np.sqrt(mean_squared_error(baseline_targets, baseline_predictions))
        baseline_mape = np.mean(np.abs((baseline_targets - baseline_predictions) / baseline_targets)) * 100

        print(f"Baseline RMSE: {baseline_rmse:.2f}")
        print(f"Baseline MAPE: {baseline_mape:.2f}%")

        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'baseline_rmse': baseline_rmse,
            'baseline_mape': baseline_mape,
            'train_predict': train_predict,
            'test_predict': test_predict,
            'y_train_inv': y_train_inv,
            'y_test_inv': y_test_inv
        }

        return metrics

    def visualize_results(self, metrics):
        """
        Visualize the prediction results.

        Args:
            metrics (dict): Evaluation metrics from evaluate method
        """
        plt.figure(figsize=(16, 8))

        plt.plot(self.data["Date"], self.data["Close"], label="Actual Close Price", color="blue")

        test_dates = self.test_data["Date"][self.seq_length:].reset_index(drop=True)
        plt.plot(test_dates, metrics['test_predict'], label="LSTM Prediction", color="red", linewidth=2)

        train_end_date = self.train_data["Date"].iloc[-1]
        plt.axvline(x=train_end_date, color="gray", linestyle="--", alpha=0.7)
        plt.text(
            train_end_date,
            plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),
            " Train | Test ",
            rotation=90,
            verticalalignment="bottom",
        )

        plt.title(f"{self.symbol} Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)

        plt.figtext(
            0.15,
            0.15,
            f"Test RMSE: {metrics['test_rmse']:.2f}\nTest MAPE: {metrics['test_mape']:.2f}%\nBaseline RMSE: {metrics['baseline_rmse']:.2f}",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        plt.show()

        plt.figure(figsize=(12, 6))
        test_dates = self.test_data["Date"][self.seq_length:].reset_index(drop=True)
        plt.plot(test_dates, metrics['y_test_inv'], label="Actual Close Price")
        plt.plot(test_dates, metrics['test_predict'], label="LSTM Prediction")

        plt.title(f"{self.symbol} Stock Price Prediction (Test Period)")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history["loss"], label="Training Loss")
        plt.plot(self.history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def predict_future(self, days=30):
        """
        Predict future stock prices.

        Args:
            days (int): Number of days to predict into the future

        Returns:
            tuple: (dates, predictions)
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        last_sequence = self.data["Close"].values[-self.seq_length:]

        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))

        predictions = []
        current_sequence = last_sequence_scaled.reshape(1, self.seq_length, 1)

        last_date = self.data["Date"].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)

        for _ in range(days):
            next_day_scaled = self.model.predict(current_sequence)

            next_day = self.scaler.inverse_transform(next_day_scaled)[0, 0]
            predictions.append(next_day)

            current_sequence = np.append(current_sequence[:, 1:, :],
                                         next_day_scaled.reshape(1, 1, 1),
                                         axis=1)

        return future_dates, np.array(predictions)

    def visualize_future(self, days=30):
        """
        Visualize future predictions.

        Args:
            days (int): Number of days to predict
        """
        future_dates, predictions = self.predict_future(days)

        plt.figure(figsize=(14, 7))

        plt.plot(self.data["Date"], self.data["Close"], label="Historical Close Price", color="blue")

        plt.plot(future_dates, predictions, label=f"Future Predictions ({days} days)",
                 color="red", linestyle="--", marker="o", markersize=4)

        last_date = self.data["Date"].iloc[-1]
        plt.axvline(x=last_date, color="gray", linestyle="--", alpha=0.7)
        plt.text(
            last_date,
            plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),
            " History | Future ",
            rotation=90,
            verticalalignment="bottom",
        )

        plt.title(f"{self.symbol} Stock Price - Historical Data and Future Prediction")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.show()

        prediction_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Close': predictions
        })

        print("Future predictions:")
        print(prediction_df)
        return prediction_df

def main():
    # Get user input for stock symbol or use default
    symbol = input("Enter stock symbol (e.g., AAPL, MSFT, ^GSPC, DAX): ") or "DAX"

    # Create stock predictor
    predictor = StockPredictor(symbol)

    # Load data
    predictor.load_data()

    # Train model
    print(f"\nTraining LSTM model for {symbol}...")
    predictor.train()

    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = predictor.evaluate()

    # Visualize results
    predictor.visualize_results(metrics)

    # Predict future
    days_to_predict = 30  # Default to 30 days
    print(f"\nPredicting {days_to_predict} days into the future...")
    future_predictions = predictor.visualize_future(days_to_predict)

    return predictor, future_predictions

if __name__ == "__main__":
    main()