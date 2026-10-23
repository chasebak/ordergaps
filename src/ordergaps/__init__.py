"""ordergaps: market volume from sequential order-number gaps.

Exact sequential-ID arithmetic + a temporal allocation model, validated
against a simulator with ground truth.
"""

from ordergaps.config import CAVEAT, EstimatorParams, StoreHours
from ordergaps.estimator import EstimateResult, estimate
from ordergaps.simulator import SimConfig, SimResult, simulate, with_share

__version__ = "0.1.0"

__all__ = [
    "CAVEAT",
    "EstimateResult",
    "EstimatorParams",
    "SimConfig",
    "SimResult",
    "StoreHours",
    "estimate",
    "simulate",
    "with_share",
]
