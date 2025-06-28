"""
HedgeForge: Portfolio Risk Modeling Engine

A Python-based quantitative finance project for simulating and optimizing 
portfolio strategies using Monte Carlo simulations, stochastic processes, 
and risk analytics.
"""

__version__ = "0.1.0"
__author__ = "Brice A. Nelson"
__email__ = "brice.nelson@example.com"

# Core imports
from . import optimizer
from . import constraints
from . import risk
from . import forecasting
from . import backtest
from . import utils

__all__ = [
    "optimizer",
    "constraints", 
    "risk",
    "forecasting",
    "backtest",
    "utils"
] 