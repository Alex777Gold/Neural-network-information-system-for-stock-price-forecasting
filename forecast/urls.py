# forecast/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.stock_forecast_view, name='stock_forecast'),
]
