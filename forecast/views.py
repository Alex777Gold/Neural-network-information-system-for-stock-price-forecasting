import os
import numpy as np
import pandas as pd
import yfinance as yf
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Import mdates for date formatting
from io import BytesIO
import base64
from .forms import StockForecastForm


def load_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data


def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler


def train_neural_network(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu',
               input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        GRU(units=50, return_sequences=True),
        GRU(units=50),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer=RMSprop(), loss='mean_squared_error')
    model.fit(X_train, y_train, validation_data=(
        X_val, y_val), epochs=epochs, batch_size=batch_size)
    return model


def forecast_stock_price(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions


def plot_forecast(dates, real_prices, predicted_prices):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, real_prices, color='blue', label='Actual Stock Price')
    plt.plot(dates, predicted_prices, color='red',
             label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()

    # Форматування дати на осі X
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(
        interval=10))  # Відображати кожні 10 днів
    plt.gcf().autofmt_xdate()  # Автоматичне обертання дат

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode("utf-8")
    return graphic


def plot_single_forecast(dates, predicted_prices):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, predicted_prices, color='green',
             label='Predicted Stock Price')
    plt.title('Second Forecast (Predicted Prices)')
    plt.xlabel('Date')
    plt.ylabel('Predicted Stock Price')
    plt.legend()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode("utf-8")
    return graphic


def stock_forecast_view(request):
    graphic = None
    graphic_second = None
    forecast_table = None
    forecast_table_second = None

    if request.method == 'POST':
        form = StockForecastForm(request.POST)
        if form.is_valid():
            # Дані для першого прогнозу
            stock_symbol = form.cleaned_data['forecast_symbol']
            train_symbol = form.cleaned_data['train_symbol']
            train_start_date = form.cleaned_data['train_start_date']
            train_end_date = form.cleaned_data['train_end_date']
            start_date = form.cleaned_data['forecast_start_date']
            end_date = form.cleaned_data['forecast_end_date']
            epochs = form.cleaned_data['epochs']

            # Дані для другого прогнозу
            # New field
            second_forecast_symbol = form.cleaned_data['second_forecast_symbol']
            second_start_date = form.cleaned_data['second_forecast_start_date']
            second_end_date = form.cleaned_data['second_forecast_end_date']

            # Завантаження і підготовка даних для навчання
            train_data = load_data(
                train_symbol, train_start_date, train_end_date)
            X_train, y_train, scaler = prepare_data(train_data)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, shuffle=False)

            # Навчання моделі
            model = train_neural_network(
                X_train, y_train, X_val, y_val, epochs=epochs)

            # Перший прогноз: з реальними даними
            forecast_data = load_data(stock_symbol, start_date, end_date)
            if forecast_data.empty:
                print("No forecast data found for the specified dates.")
            else:
                X_forecast, y_forecast, _ = prepare_data(forecast_data)
                predictions = forecast_stock_price(model, X_forecast, scaler)
                real_prices = scaler.inverse_transform(
                    y_forecast.reshape(-1, 1)).flatten()
                predicted_prices = predictions.flatten()
                dates = pd.date_range(
                    start=start_date, periods=len(real_prices)).to_list()
                graphic = plot_forecast(dates, real_prices, predicted_prices)

                # Таблиця для першого прогнозу
                forecast_table = pd.DataFrame({
                    'Date': dates,
                    'Real_Price': real_prices,
                    'Predicted_Price': predicted_prices
                })
                forecast_table = forecast_table.reset_index(
                    drop=True).to_dict(orient='records')

                # Другий прогноз: майбутні дати без реальних даних
                if second_start_date and second_end_date:
                    future_dates = pd.date_range(
                        start=second_start_date, end=second_end_date, freq='B')
                    num_future_days = len(future_dates)

                    # Завантаження даних для другого прогнозу
                    second_forecast_data = load_data(
                        second_forecast_symbol, second_start_date, second_end_date)
                    if second_forecast_data.empty:
                        print(
                            f"No future data found for {second_forecast_symbol} between {second_start_date} and {second_end_date}.")
                    else:
                        # Підготовка даних для другого прогнозу
                        _, _, second_scaler = prepare_data(
                            second_forecast_data)
                        # Використання останніх 60 днів для X_future
                        last_known_data = second_forecast_data['Close'].values[-60:]
                        last_known_data_scaled = second_scaler.transform(
                            last_known_data.reshape(-1, 1))
                        X_future = last_known_data_scaled.reshape(1, 60, 1)

                        # Список для майбутніх прогнозів
                        future_predictions = []

                        for _ in range(num_future_days):
                            # Прогноз на основі поточного X_future
                            prediction = model.predict(X_future)
                            future_predictions.append(prediction[0, 0])
                            new_value = prediction.reshape(1, 1, 1)
                            X_future = np.concatenate(
                                (X_future[:, 1:, :], new_value), axis=1)

                        # Перетворення прогнозів у реальні значення
                        future_predictions = second_scaler.inverse_transform(
                            np.array(future_predictions).reshape(-1, 1)).flatten()
                        future_predictions = np.maximum(0, future_predictions)

                        # Перевірка чи є дані для таблиці
                        if future_predictions.size > 0:
                            graphic_second = plot_single_forecast(
                                future_dates, future_predictions)

                            # Таблиця для другого прогнозу
                            forecast_table_second = pd.DataFrame({
                                'Date': future_dates,
                                'Predicted_Price': future_predictions
                            }).reset_index(drop=True).to_dict(orient='records')
                        else:
                            print("No future predictions were made.")

    else:
        form = StockForecastForm()

    return render(request, 'forecast/forecast.html', {
        'form': form,
        'graphic': graphic,
        'forecast_table': forecast_table,
        'graphic_second': graphic_second,
        'forecast_table_second': forecast_table_second
    })
