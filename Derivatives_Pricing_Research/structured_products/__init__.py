"""
Structured Equity Products Library

A comprehensive library for pricing and simulating structured equity products.
"""

from .autocallable import Autocallable
from .reverse_convertible import ReverseConvertible
from .barrier_option import BarrierOption
from .range_accrual import RangeAccrual
from .pricing_engine import PricingEngine

__version__ = "1.0.0"
__all__ = [
    "Autocallable",
    "ReverseConvertible", 
    "BarrierOption",
    "RangeAccrual",
    "PricingEngine"
]