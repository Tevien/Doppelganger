"""
Base Score Calculator Framework
================================

Provides the abstract base class for all clinical risk score calculators
and a registry for dynamic discovery.

Every score calculator must:
1. Subclass ``BaseScoreCalculator``
2. Declare the features it needs via ``required_features``
3. Implement ``_calculate`` to produce a DataFrame of score columns

The ``ScoreRegistry`` auto-discovers all registered subclasses so that
the pipeline can look up calculators by name.

Author: SB
Date: 2025-10-30
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ScoreRegistry:
    """
    Central registry for score calculators.

    Calculators register themselves by calling
    ``ScoreRegistry.register(name, cls)`` or by using the
    ``@ScoreRegistry.auto`` class decorator.

    Examples
    --------
    >>> @ScoreRegistry.auto
    ... class MyScore(BaseScoreCalculator):
    ...     name = "my_score"
    ...     ...
    >>> ScoreRegistry.get("my_score")
    <class 'MyScore'>
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, calculator_cls: type) -> None:
        """Register a score calculator class under *name*."""
        cls._registry[name.lower()] = calculator_cls
        logger.debug(f"Registered score calculator: {name}")

    @classmethod
    def get(cls, name: str) -> type:
        """
        Look up a score calculator by name.

        Raises
        ------
        KeyError
            If the name is not registered.
        """
        key = name.lower()
        if key not in cls._registry:
            raise KeyError(
                f"Score '{name}' is not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key]

    @classmethod
    def available(cls) -> List[str]:
        """Return a sorted list of registered score names."""
        return sorted(cls._registry.keys())

    @classmethod
    def auto(cls, calculator_cls: type) -> type:
        """
        Class decorator that auto-registers a ``BaseScoreCalculator`` subclass.

        The class **must** have a ``name`` class attribute.
        """
        if not hasattr(calculator_cls, "name") or not calculator_cls.name:
            raise ValueError(
                f"{calculator_cls.__name__} must define a 'name' class attribute "
                "to be auto-registered."
            )
        cls.register(calculator_cls.name, calculator_cls)
        return calculator_cls


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseScoreCalculator(ABC):
    """
    Abstract base class for clinical risk-score calculators.

    Subclasses must implement:
    * ``name``               – unique string identifier
    * ``required_features``  – list of canonical feature keys
    * ``_calculate``         – compute the score given data + column mapping

    The public ``calculate`` method adds input validation and logging on top.

    Parameters
    ----------
    column_mapping : dict
        Maps canonical feature names (e.g. ``"age"``, ``"sex"``) to
        actual column names in the dataset.
    """

    # Must be overridden by subclasses
    name: str = ""

    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        self.column_mapping = column_mapping or {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def required_features(self) -> List[str]:
        """
        Return the list of canonical feature keys this score requires.

        The keys correspond to the *keys* of ``column_mapping``.
        """
        ...

    @abstractmethod
    def _calculate(self, data: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Compute the score.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset.  All columns referenced in *column_mapping*
            are guaranteed to exist (though they may contain NaNs).
        column_mapping : dict
            Maps canonical feature keys to dataset column names.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the same index as *data* containing one or
            more score columns.
        """
        ...

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def output_columns(self) -> List[str]:
        """
        Return the names of the columns produced by this calculator.

        Override if you want the pipeline to know ahead of time.
        """
        return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate the score with input validation.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset.
        column_mapping : dict, optional
            If provided, overrides the mapping given at construction time.

        Returns
        -------
        pd.DataFrame
            Score columns with the same index as *data*.
        """
        mapping = column_mapping or self.column_mapping
        if not mapping:
            raise ValueError(
                f"No column_mapping provided for score '{self.name}'."
            )

        # Check required features are mapped
        missing_keys = set(self.required_features()) - set(mapping.keys())
        if missing_keys:
            raise ValueError(
                f"column_mapping is missing keys for score '{self.name}': "
                f"{sorted(missing_keys)}"
            )

        # Check that mapped columns exist in the data
        required_mapped_columns = [mapping[key] for key in self.required_features()]
        missing_cols = [
            col for col in required_mapped_columns if col not in data.columns
        ]

        if missing_cols:
            logger.warning(
                f"[{self.name}] Dataset is missing required columns: "
                f"{missing_cols}. "
                "Rows with missing values will yield NaN scores."
            )

        # Add absent *required* columns as NaN on a copy so _calculate
        # never hits a KeyError (same behaviour as FeatureDropper).
        # Non-required mapping values (e.g. literal config strings like
        # "High" for risk_region) are intentionally left alone — the
        # individual calculator's _resolve helpers handle those.
        if missing_cols:
            data = data.copy()
            for col in missing_cols:
                data[col] = np.nan

        logger.info(f"Calculating score: {self.name}  (n={len(data)})")
        result = self._calculate(data, mapping)
        logger.info(
            f"Score '{self.name}' produced columns: {result.columns.tolist()}"
        )
        return result

    def get_mapped_columns(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> Set[str]:
        """Return the set of dataset column names required by this score."""
        mapping = column_mapping or self.column_mapping
        return set(mapping.values())

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
