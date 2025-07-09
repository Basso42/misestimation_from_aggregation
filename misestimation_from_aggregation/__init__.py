"""
Misestimation from Aggregation

A Python package for analyzing the misestimation that occurs when aggregating 
firm-level production networks to sector-level networks.

This package implements the methodology from:
Diem, C., Borsos, A., Reisch, T., Kertész, J., & Thurner, S. (2023).
"Estimating the loss of economic predictability from aggregating firm-level production networks."
"""

__version__ = "0.1.0"
__author__ = "Christoph Diem, András Borsos, Tobias Reisch, János Kertész, Stefan Thurner"

from .network_aggregation import NetworkAggregator
from .similarity_measures import SimilarityCalculator
from .shock_sampling import ShockSampler
from .utils import validate_network, validate_sectors

__all__ = [
    "NetworkAggregator",
    "SimilarityCalculator", 
    "ShockSampler",
    "validate_network",
    "validate_sectors"
]