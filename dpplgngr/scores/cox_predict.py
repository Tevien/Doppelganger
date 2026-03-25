"""
Cox Proportional Hazards Predictor
====================================

Provides ``CoxPHPredictorWithUncertainty`` for applying pre-fitted Cox
coefficients to new data, and ``CoxPHScore`` — a ``BaseScoreCalculator``
wrapper so it can be used from the score-analysis pipeline.

The Cox model is specified by providing hazard ratios (and their standard
errors) as dictionaries.  The canonical feature keys in the column
mapping should match the keys in the hazard-ratio dictionaries.

Author: SB
Date: 2025-10-30
"""

from typing import Dict, List, Optional

import logging
import numpy as np
import pandas as pd

from dpplgngr.scores.base import BaseScoreCalculator, ScoreRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core predictor (unchanged logic)
# ---------------------------------------------------------------------------

class CoxPHPredictorWithUncertainty:
    def __init__(self, hazard_ratios, hazard_ratios_uncertainties):
        """
        Initialize the CoxPHPredictorWithUncertainty with hazard ratios and their uncertainties.

        Args:
        - hazard_ratios (dict): A dictionary containing the hazard ratios for each covariate.
                                The keys are the names of the covariates, and the values are
                                the corresponding hazard ratios.
        - hazard_ratios_uncertainties (dict): A dictionary containing the uncertainties (e.g., standard errors)
                                              associated with the hazard ratios.
        """
        self.hazard_ratios = hazard_ratios
        self.hazard_ratios_uncertainties = hazard_ratios_uncertainties

    def predict_cox_output_with_uncertainty(self, covariates):
        """
        Predict survival probability for a given set of covariates with uncertainties.

        Args:
        - covariates (dict): A dictionary containing the values of covariates for an individual.
                             The keys are the names of the covariates, and the values are
                             the corresponding covariate values.

        Returns:
        - cox_output (float): The predicted survival probability for the individual.
        - uncertainty (float): The uncertainty associated with the predicted survival probability.
        """
        hazard_sum = 0
        hazard_sum_uncertainty = 0
        
        for covariate, value in covariates.items():
            if covariate in self.hazard_ratios:
                hazard_sum += self.hazard_ratios[covariate] * value
                hazard_sum_uncertainty += (self.hazard_ratios_uncertainties[covariate] * value) ** 2
        
        cox_output = np.exp(-hazard_sum)
        uncertainty = np.sqrt(hazard_sum_uncertainty)
        
        return cox_output, uncertainty


# ---------------------------------------------------------------------------
# BaseScoreCalculator wrapper
# ---------------------------------------------------------------------------

@ScoreRegistry.auto
class CoxPHScore(BaseScoreCalculator):
    """
    Cox PH score calculator for the pipeline.

    Parameters
    ----------
    column_mapping : dict
        Maps canonical covariate names to dataset columns.
    hazard_ratios : dict
        Covariate name → hazard ratio.
    hazard_ratios_uncertainties : dict, optional
        Covariate name → SE of hazard ratio.
    """

    name = "cox_ph"

    def __init__(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
        hazard_ratios: Optional[Dict[str, float]] = None,
        hazard_ratios_uncertainties: Optional[Dict[str, float]] = None,
    ):
        super().__init__(column_mapping)
        self.hazard_ratios = hazard_ratios or {}
        self.hazard_ratios_uncertainties = hazard_ratios_uncertainties or {
            k: 0.0 for k in (hazard_ratios or {})
        }
        self._predictor = CoxPHPredictorWithUncertainty(
            self.hazard_ratios, self.hazard_ratios_uncertainties
        )

    def required_features(self) -> List[str]:
        return list(self.hazard_ratios.keys())

    def output_columns(self) -> List[str]:
        return ["cox_survival", "cox_uncertainty"]

    def _calculate(
        self,
        data: pd.DataFrame,
        column_mapping: Dict[str, str],
    ) -> pd.DataFrame:
        survivals = []
        uncertainties = []

        for _, row in data.iterrows():
            covariates = {}
            any_missing = False
            for key in self.hazard_ratios:
                col = column_mapping.get(key)
                if col is None or col not in data.columns:
                    any_missing = True
                    break
                val = row[col]
                if pd.isna(val):
                    any_missing = True
                    break
                covariates[key] = float(val)

            if any_missing:
                survivals.append(np.nan)
                uncertainties.append(np.nan)
            else:
                surv, unc = self._predictor.predict_cox_output_with_uncertainty(covariates)
                survivals.append(surv)
                uncertainties.append(unc)

        return pd.DataFrame(
            {"cox_survival": survivals, "cox_uncertainty": uncertainties},
            index=data.index,
        )