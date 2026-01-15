"""
Doppelganger Training Module

This module provides training tasks and imputation methods.

Available Tasks
---------------
- TrainGraphImputer: Train graph-based imputation model
- ApplyGraphImputer: Apply trained graph imputer
- SDVGen: Generate synthetic data using SDV
- AlternativeImputation: Create alternative imputations using different strategies

Available Functions
-------------------
- create_alternative_imputations: Standalone function for creating imputations
"""

from .graph_imputation import TrainGraphImputer, ApplyGraphImputer
from .sdv import SDVGen
from .alternative_imputation import (
    AlternativeImputation,
    create_alternative_imputations,
)

__all__ = [
    'TrainGraphImputer',
    'ApplyGraphImputer',
    'SDVGen',
    'AlternativeImputation',
    'create_alternative_imputations',
]
