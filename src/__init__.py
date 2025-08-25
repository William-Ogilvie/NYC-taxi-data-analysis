# src/__init__.py
from .forecast_helpers import run_forecasts
from .data_processing import process_taxi_data

# What can be imported from src
__all__ = ["run_forecasts", "process_taxi_data"]




''''
This files means src/ is treated as its own python package
and we can thus import functions from it easily
'''