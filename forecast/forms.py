# forecast/forms.py
from django import forms


class StockForecastForm(forms.Form):
    # Поля для навчання
    train_symbol = forms.CharField(
        label='Training Symbol', max_length=5, initial='TSLA')
    train_start_date = forms.DateField(
        label='Training Start Date', initial='2014-01-01')
    train_end_date = forms.DateField(
        label='Training End Date', initial='2021-12-30')

    # Поля для першого прогнозу
    forecast_symbol = forms.CharField(
        label='Forecast Symbol', max_length=5, initial='TSLA')
    forecast_start_date = forms.DateField(
        label='Forecast Start Date', initial='2022-01-01')
    forecast_end_date = forms.DateField(
        label='Forecast End Date', initial='2022-06-01')

    # Поля для другого прогнозу
    second_forecast_symbol = forms.CharField(
        label='Second Forecast Symbol', max_length=5, initial='TSLA')
    second_forecast_start_date = forms.DateField(
        label='Second Forecast Start Date', initial='2024-01-01', required=False)
    second_forecast_end_date = forms.DateField(
        label='Second Forecast End Date', initial='2024-06-01',  required=False)

    # Параметри навчання
    epochs = forms.IntegerField(label='Epochs', initial=5, min_value=1)
