"""Prediction-time and training-time privacy defenses for QuRiFT.

The package deliberately keeps defense code separate from the frozen SaTML
experiment stack.  Every prediction-time defense consumes the same
``PredictionOracle`` interface so attacks cannot accidentally use a different
model path for defended and undefended conditions.
"""

from .base import PredictionBatch, PredictionOracle
from .discriminator import MembershipDiscriminator, fit_membership_discriminator
from .dynanoise import DynaNoiseOracle
from .guards import (
    LatticeRoundOracle,
    LogitGuardOracle,
    MeasurementGuardOracle,
    StickyInputOracle,
    project_expectation_lattice,
)
from .hamp import CalibrationSupportGenerator, HAMPOutputOracle
from .memguard import MemGuardOracle
from .oracle import RawOracle

__all__ = [
    "CalibrationSupportGenerator",
    "DynaNoiseOracle",
    "HAMPOutputOracle",
    "LatticeRoundOracle",
    "LogitGuardOracle",
    "MeasurementGuardOracle",
    "MemGuardOracle",
    "MembershipDiscriminator",
    "PredictionBatch",
    "PredictionOracle",
    "RawOracle",
    "StickyInputOracle",
    "fit_membership_discriminator",
    "project_expectation_lattice",
]
