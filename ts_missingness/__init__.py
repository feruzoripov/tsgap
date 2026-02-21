"""Time-Series Missingness Simulation Library

A library for simulating realistic missingness patterns in time-series data
for imputation benchmarking.
"""

from .core import simulate_missingness, simulate_many_rates, MissingnessSimulator

__version__ = "0.1.0"
__all__ = ["simulate_missingness", "simulate_many_rates", "MissingnessSimulator"]
