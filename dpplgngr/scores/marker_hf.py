"""
MARKER-HF Risk Score
=====================

Wraps the ``pyMarker`` BDT-based heart-failure risk score
(Claudio Campagnari, UC Santa Barbara) in the ``BaseScoreCalculator``
framework so it can be used from the score-analysis pipeline.

The underlying ``pyMarker`` function expects an ordered list of 8
biomarkers; this wrapper handles the column-mapping and DataFrame
interface automatically.

Required features (canonical keys)
-----------------------------------
``diastolic_bp``, ``creatinine``, ``bun``, ``hemoglobin``, ``wbc``,
``platelets``, ``albumin``, ``rdw``

.. note::
   Creatinine here is in **mg/dL** (not µmol/L as in MAGGIC).

Author: SB
Date: 2025-10-31
"""

from typing import Dict, List

import logging
import numpy as np
import pandas as pd

from dpplgngr.scores.base import BaseScoreCalculator, ScoreRegistry
from dpplgngr.scores.pyMarker import pyMarker

logger = logging.getLogger(__name__)

# Canonical feature keys in the order pyMarker expects
MARKER_FEATURES = [
    "diastolic_bp",  # BPDIAS  mm Hg
    "creatinine",    # CREATN  mg/dL
    "bun",           # BUN     mg/dL
    "hemoglobin",    # HGB     g/dL
    "wbc",           # WBC     10^3 per µL
    "platelets",     # PLT     10^3 per µL
    "albumin",       # ALB     g/dL
    "rdw",           # RDW     %
]


@ScoreRegistry.auto
class MARKERHFScore(BaseScoreCalculator):
    """
    MARKER-HF risk score.

    Uses a boosted decision tree (BDT) translated from the original
    C++/TMVA implementation.  The output is a continuous risk score
    (higher = worse prognosis).

    Parameters
    ----------
    column_mapping : dict
        Maps the 8 canonical keys (see ``MARKER_FEATURES``) to actual
        column names in the dataset.
    check_boundaries : bool
        Whether to enforce physiological boundary checks on input
        values (default ``True``).
    """

    name = "marker_hf"

    def __init__(self, column_mapping=None, check_boundaries: bool = True):
        super().__init__(column_mapping)
        self.check_boundaries = check_boundaries

    def required_features(self) -> List[str]:
        return list(MARKER_FEATURES)

    def output_columns(self) -> List[str]:
        return ["marker_hf"]

    def _calculate(
        self,
        data: pd.DataFrame,
        column_mapping: Dict[str, str],
    ) -> pd.DataFrame:
        results = []
        for idx, row in data.iterrows():
            values = []
            any_missing = False
            for key in MARKER_FEATURES:
                col = column_mapping[key]
                val = row.get(col, np.nan)
                if pd.isna(val):
                    any_missing = True
                    break
                values.append(float(val))

            if any_missing:
                results.append(np.nan)
            else:
                score = pyMarker(values, checkBoundaries=self.check_boundaries)
                results.append(score if score != -99 else np.nan)

        return pd.DataFrame({"marker_hf": results}, index=data.index)
