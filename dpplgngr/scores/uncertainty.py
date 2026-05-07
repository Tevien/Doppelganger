"""
Uncertainty-Aware Score Calculation via Multiple Imputation
============================================================

This module propagates imputation uncertainty through risk-score calculators
using Rubin's multiple-imputation (MI) framework.

**Methodology**

Given M complete imputed datasets from the graph imputer's
``impute_multiple()`` method (which combines parametric sampling from
the learned heteroscedastic variance heads with MC Dropout for
epistemic diversity):

1. Each dataset m ∈ {1, …, M} is passed through the score calculator
   independently, yielding per-patient score values  s_i^{(m)}.

2. Point estimate (pooled across imputations):

       s̄_i  =  (1/M) Σ_m  s_i^{(m)}

3. Between-imputation variance:

       B_i  =  [1/(M-1)] Σ_m  (s_i^{(m)} − s̄_i)²

4. Total variance (Rubin's simplified rule, W ≈ 0 for deterministic
   score formulae):

       T_i  =  (1 + 1/M) × B_i

5. Confidence intervals are computed as empirical percentiles from the
   M samples (non-parametric, avoids normality assumption on score
   distribution):

       CI_95  =  [ percentile_2.5(s_i^{(·)}),  percentile_97.5(s_i^{(·)}) ]

6. Feature-level attribution:  For each score, we identify which input
   features were imputed (vs. observed) per patient and measure how
   the number / quality of imputed inputs correlates with score
   uncertainty.

Author: SB
Date: 2025
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UncertaintyAwareScorer:
    """
    Run a score calculator on M imputed datasets and pool results.

    Parameters
    ----------
    score_calculator : callable
        A score calculator instance that implements ``calculate(df, mapping)``.
        Typically a :class:`BaseScoreCalculator` subclass from
        ``dpplgngr.scores``.
    column_mapping : dict
        Maps canonical feature names to DataFrame column names, exactly as
        you would pass to ``score_calculator.calculate()``.
    score_columns : list of str
        Names of the numeric score output columns to track uncertainty for
        (e.g. ``["score2"]`` or ``["prevent_10yr_total_cvd"]``).
    """

    def __init__(
        self,
        score_calculator,
        column_mapping: Dict[str, str],
        score_columns: List[str],
    ):
        self.calculator = score_calculator
        self.mapping = column_mapping
        self.score_columns = score_columns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_multiple(
        self,
        imputed_datasets: np.ndarray,
        feature_names: List[str],
        base_df: pd.DataFrame,
        missing_mask: np.ndarray,
        model_feature_names: List[str],
        confidence_scores: Optional[np.ndarray] = None,
        extra_derive_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Score M imputed datasets and return pooled uncertainty estimates.

        Parameters
        ----------
        imputed_datasets : ndarray of shape (M, N, D_model)
            Output of ``GraphImputationModel.impute_multiple()``.
        feature_names : list of str, length D_model
            Feature names that correspond to the D_model columns of
            *imputed_datasets* (i.e. the graph imputer's model features).
        base_df : pd.DataFrame of shape (N, …)
            The original (un-imputed) DataFrame.  Non-model columns (such
            as ``AgeAtOpname``, ``Geslacht``, binary flags, etc.) are taken
            from here.
        missing_mask : ndarray of shape (N, D_model), dtype bool
            Which cells were imputed.
        model_feature_names : list of str
            Same as *feature_names* (kept for clarity).
        confidence_scores : ndarray of shape (N,), optional
            Per-patient global confidence from the graph imputer.
        extra_derive_fn : callable, optional
            A function ``f(df) → df`` that derives additional columns
            needed by the score (e.g. eGFR from creatinine).  Called on
            each imputed DataFrame before scoring.

        Returns
        -------
        results : pd.DataFrame
            One row per patient.  Contains for each score column ``s``:

            * ``s``               – pooled point estimate (mean)
            * ``s_std``           – standard deviation across M scores
            * ``s_ci_lower``      – 2.5th percentile (lower 95% CI)
            * ``s_ci_upper``      – 97.5th percentile (upper 95% CI)
            * ``s_iqr``           – inter-quartile range
            * ``s_rubin_var``     – total variance via Rubin's rule
            * ``s_cv``            – coefficient of variation (std / |mean|)

            Plus per-patient metadata:

            * ``n_imputed_features``  – count of model features imputed
            * ``imputed_feature_list`` – comma-separated names
            * ``imputer_confidence``  – global confidence from imputer
            * ``frac_imputed_score_inputs`` – fraction of *this score's*
              inputs that came from imputation (vs. observed)
        """
        M, N, D = imputed_datasets.shape

        # Identify which model features feed into the score
        score_input_cols = set(self.mapping.values())
        model_feat_set = set(model_feature_names)

        # Per-patient: which model features were imputed?
        per_patient_n_imputed = np.sum(missing_mask, axis=1)  # [N]
        per_patient_imputed_names = []
        for i in range(N):
            names = [
                model_feature_names[j]
                for j in range(D)
                if missing_mask[i, j]
            ]
            per_patient_imputed_names.append(",".join(names) if names else "")

        # Per-patient: fraction of score inputs that were imputed
        # (only count model features that appear in the score mapping)
        score_model_features = [
            f for f in model_feature_names if f in score_input_cols
        ]
        # Also check if mapping values are base names that match model features
        mapped_to_model = []
        for canonical, col_name in self.mapping.items():
            if col_name in model_feat_set:
                idx = model_feature_names.index(col_name)
                mapped_to_model.append((canonical, idx))

        frac_imputed = np.zeros(N)
        if mapped_to_model:
            for i in range(N):
                n_imp = sum(
                    1 for _, idx in mapped_to_model if missing_mask[i, idx]
                )
                frac_imputed[i] = n_imp / len(mapped_to_model)

        # ----- Score each imputed dataset -----
        all_scores = {col: np.full((M, N), np.nan) for col in self.score_columns}

        for m in range(M):
            # Build a DataFrame for this imputation sample
            df_m = base_df.copy()

            # Overlay imputed model features
            for j, feat_name in enumerate(feature_names):
                df_m[feat_name] = imputed_datasets[m, :, j]

            # Derive any extra columns (e.g. eGFR)
            if extra_derive_fn is not None:
                df_m = extra_derive_fn(df_m)

            # Calculate the score (supports both callable and .calculate() API)
            try:
                if hasattr(self.calculator, "calculate"):
                    score_result = self.calculator.calculate(df_m, self.mapping)
                else:
                    # Wrapper function returned by get_score_calculator()
                    score_result = self.calculator(df_m, self.mapping)
                for col in self.score_columns:
                    if col in score_result.columns:
                        all_scores[col][m] = score_result[col].values
            except Exception as e:
                logger.warning(f"Imputation sample {m} scoring failed: {e}")
                continue

        # ----- Pool results (Rubin's rules + percentile CIs) -----
        results = pd.DataFrame(index=base_df.index)

        for col in self.score_columns:
            scores_arr = all_scores[col]  # [M, N]

            # Count valid (non-NaN) imputations per patient
            valid_counts = np.sum(~np.isnan(scores_arr), axis=0)  # [N]

            # Pooled point estimate
            score_mean = np.nanmean(scores_arr, axis=0)
            score_std = np.nanstd(scores_arr, axis=0, ddof=1)

            # Rubin's total variance: T = (1 + 1/M) * B
            B = np.nanvar(scores_arr, axis=0, ddof=1)
            T = (1.0 + 1.0 / M) * B

            # Percentile-based 95% CI (non-parametric)
            ci_lower = np.nanpercentile(scores_arr, 2.5, axis=0)
            ci_upper = np.nanpercentile(scores_arr, 97.5, axis=0)

            # IQR
            iqr = np.nanpercentile(scores_arr, 75, axis=0) - np.nanpercentile(
                scores_arr, 25, axis=0
            )

            # Coefficient of variation
            with np.errstate(divide="ignore", invalid="ignore"):
                cv = np.where(np.abs(score_mean) > 1e-10, score_std / np.abs(score_mean), 0.0)

            results[col] = score_mean
            results[f"{col}_std"] = score_std
            results[f"{col}_ci_lower"] = ci_lower
            results[f"{col}_ci_upper"] = ci_upper
            results[f"{col}_iqr"] = iqr
            results[f"{col}_rubin_var"] = T
            results[f"{col}_cv"] = cv
            results[f"{col}_n_valid_imputations"] = valid_counts

        # Patient-level metadata
        results["n_imputed_features"] = per_patient_n_imputed
        results["imputed_feature_list"] = per_patient_imputed_names
        results["frac_imputed_score_inputs"] = frac_imputed
        if confidence_scores is not None:
            results["imputer_confidence"] = confidence_scores

        logger.info(
            f"Uncertainty scoring complete for {len(self.score_columns)} "
            f"score column(s) across {M} imputations, {N} patients."
        )
        for col in self.score_columns:
            mean_std = results[f"{col}_std"].mean()
            mean_ci_width = (results[f"{col}_ci_upper"] - results[f"{col}_ci_lower"]).mean()
            logger.info(
                f"  {col}: mean_std={mean_std:.4f}, "
                f"mean_95CI_width={mean_ci_width:.4f}"
            )

        return results


def summarize_uncertainty(
    results: pd.DataFrame,
    score_columns: List[str],
) -> Dict:
    """
    Produce a JSON-serializable summary of the uncertainty analysis.

    Parameters
    ----------
    results : pd.DataFrame
        Output of :meth:`UncertaintyAwareScorer.score_multiple`.
    score_columns : list of str
        The primary score column names.

    Returns
    -------
    dict
        Summary statistics suitable for logging or saving to JSON.
    """
    summary = {}

    for col in score_columns:
        std_col = f"{col}_std"
        ci_lo = f"{col}_ci_lower"
        ci_hi = f"{col}_ci_upper"
        cv_col = f"{col}_cv"

        ci_width = results[ci_hi] - results[ci_lo]

        summary[col] = {
            "point_estimate": {
                "mean": float(results[col].mean()),
                "median": float(results[col].median()),
                "std": float(results[col].std()),
            },
            "uncertainty": {
                "mean_score_std": float(results[std_col].mean()),
                "median_score_std": float(results[std_col].median()),
                "mean_95ci_width": float(ci_width.mean()),
                "median_95ci_width": float(ci_width.median()),
                "mean_cv": float(results[cv_col].mean()),
                "max_score_std": float(results[std_col].max()),
            },
            "imputation_dependency": {
                "mean_n_imputed_features": float(
                    results["n_imputed_features"].mean()
                ),
                "corr_std_vs_n_imputed": float(
                    results[[std_col, "n_imputed_features"]]
                    .corr()
                    .iloc[0, 1]
                )
                if results["n_imputed_features"].std() > 0
                else 0.0,
                "corr_std_vs_frac_imputed": float(
                    results[[std_col, "frac_imputed_score_inputs"]]
                    .corr()
                    .iloc[0, 1]
                )
                if results["frac_imputed_score_inputs"].std() > 0
                else 0.0,
            },
        }

        if "imputer_confidence" in results.columns:
            summary[col]["imputation_dependency"]["corr_std_vs_confidence"] = float(
                results[[std_col, "imputer_confidence"]].corr().iloc[0, 1]
            )

    return summary
