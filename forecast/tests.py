import os
import numpy as np
import pandas as pd
from django.test import TestCase
from django.urls import reverse
from .forms import StockForecastForm
from .views import load_data, prepare_data, train_neural_network, forecast_stock_price


class StockForecastTests(TestCase):
    def setUp(self):
        # Optional: Load any necessary initial data or configurations.
        self.stock_symbol = 'AAPL'
        self.train_symbol = 'AAPL'
        self.train_start_date = '2014-01-01'
        self.train_end_date = '2021-12-30'
        self.forecast_start_date = '2022-01-01'
        self.forecast_end_date = '2022-06-01'
        self.epochs = 5

    def test_load_data(self):
        """Test loading data from yfinance."""
        data = load_data(self.stock_symbol,
                         self.train_start_date, self.train_end_date)
        self.assertFalse(data.empty)
        self.assertIn('Close', data.columns)

    def test_prepare_data(self):
        """Test preparing data for training."""
        data = load_data(self.train_symbol,
                         self.train_start_date, self.train_end_date)
        X, y, scaler = prepare_data(data)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[1], 60)  # Look back period

    def test_train_neural_network(self):
        """Test training the neural network model."""
        data = load_data(self.train_symbol,
                         self.train_start_date, self.train_end_date)
        X, y, scaler = prepare_data(data)
        X_train, X_val = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
        y_train, y_val = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

        model = train_neural_network(
            X_train, y_train, X_val, y_val, epochs=self.epochs)
        self.assertIsNotNone(model)  # Check if the model is created

    def test_forecast_stock_price(self):
        """Test forecasting stock prices."""
        data = load_data(self.train_symbol,
                         self.train_start_date, self.train_end_date)
        X, y, scaler = prepare_data(data)
        model = train_neural_network(X, y, X, y, epochs=self.epochs)

        # Prepare data for prediction
        forecast_data = load_data(
            self.stock_symbol, self.forecast_start_date, self.forecast_end_date)
        X_forecast, _, _ = prepare_data(forecast_data)
        predictions = forecast_stock_price(model, X_forecast, scaler)
        # Ensure predictions are made
        self.assertEqual(predictions.shape[0], X_forecast.shape[0])
