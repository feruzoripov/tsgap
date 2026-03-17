"""TSGap — Composable Time-Series Missingness Simulation

A library for simulating realistic missingness patterns in time-series data
for imputation benchmarking.

Separates two concepts:
1. MECHANISMS (why data is missing): MCAR, MAR, MNAR
2. PATTERNS (how data is missing): pointwise, block, monotone, decay, markov
"""

from .core import simulate_missingness, simulate_many_rates, MissingnessSimulator
from .mechanisms import MECHANISMS
from .patterns import PATTERNS

__version__ = "0.3.0"
__all__ = [
    "simulate_missingness",
    "simulate_many_rates",
    "MissingnessSimulator",
    "MECHANISMS",
    "PATTERNS",
]
