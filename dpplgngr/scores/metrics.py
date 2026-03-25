"""
Evaluation Metrics
===================

Comprehensive performance evaluation for clinical risk scores.

This module computes all standard binary-classification metrics
(and their confidence intervals via bootstrapping) needed to judge
how well a risk score predicts a clinical outcome.

Metrics included
----------------
* AUROC (area under the ROC curve)
* AUPRC (area under the precision–recall curve)
* F1 score (at a chosen threshold)
* PPV / precision (positive predictive value)
* NPV (negative predictive value)
* Sensitivity / recall
* Specificity
* Brier score
* Calibration curve data
* Optimal threshold (Youden's J)

Author: SB
Date: 2025-10-31
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ScoreEvaluation:
    """Container for all evaluation metrics of a single score–outcome pair."""

    score_name: str
    outcome_column: str
    score_column: str
    n_samples: int
    n_events: int
    event_rate: float

    # Discrimination
    auroc: Optional[float] = None
    auroc_ci: Optional[Tuple[float, float]] = None
    auprc: Optional[float] = None
    auprc_ci: Optional[Tuple[float, float]] = None

    # Threshold-dependent (at optimal or user-specified threshold)
    threshold: Optional[float] = None
    f1: Optional[float] = None
    ppv: Optional[float] = None          # positive predictive value
    npv: Optional[float] = None          # negative predictive value
    sensitivity: Optional[float] = None  # recall / true positive rate
    specificity: Optional[float] = None  # true negative rate
    accuracy: Optional[float] = None

    # Calibration
    brier_score: Optional[float] = None
    calibration_prob_true: Optional[List[float]] = None
    calibration_prob_pred: Optional[List[float]] = None

    # Bootstrap CIs for threshold-dependent metrics
    f1_ci: Optional[Tuple[float, float]] = None
    ppv_ci: Optional[Tuple[float, float]] = None
    npv_ci: Optional[Tuple[float, float]] = None
    sensitivity_ci: Optional[Tuple[float, float]] = None
    specificity_ci: Optional[Tuple[float, float]] = None

    # Score distribution stats
    score_mean: Optional[float] = None
    score_std: Optional[float] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    score_median: Optional[float] = None
    score_n_missing: int = 0

    # Metadata
    resolver_used: str = ""
    resolver_info: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize to a JSON-friendly dictionary."""
        return _make_serializable(asdict(self))

    def summary(self) -> str:
        """One-line summary string."""
        parts = [f"AUROC={self.auroc:.3f}" if self.auroc else "AUROC=N/A"]
        if self.f1 is not None:
            parts.append(f"F1={self.f1:.3f}")
        if self.ppv is not None:
            parts.append(f"PPV={self.ppv:.3f}")
        if self.npv is not None:
            parts.append(f"NPV={self.npv:.3f}")
        if self.sensitivity is not None:
            parts.append(f"Sens={self.sensitivity:.3f}")
        if self.specificity is not None:
            parts.append(f"Spec={self.specificity:.3f}")
        if self.brier_score is not None:
            parts.append(f"Brier={self.brier_score:.4f}")
        return f"[{self.score_name}] n={self.n_samples}  " + "  ".join(parts)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    score_name: str = "",
    outcome_column: str = "",
    score_column: str = "",
    threshold: Optional[float] = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    resolver_used: str = "",
    resolver_info: Optional[Dict] = None,
) -> ScoreEvaluation:
    """
    Compute comprehensive evaluation metrics for a risk score.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground-truth outcome (0/1).
    y_score : array-like of shape (n,)
        Predicted risk score (continuous, in [0, 1] or similar).
    score_name : str
        Name of the score (for labelling).
    outcome_column : str
        Name of the outcome variable.
    score_column : str
        Name of the score variable.
    threshold : float, optional
        Decision threshold.  If ``None``, the optimal threshold
        (Youden's J statistic) is chosen automatically.
    n_bootstrap : int
        Number of bootstrap iterations for confidence intervals.
    ci_level : float
        Confidence level (default 0.95 = 95 %).
    resolver_used : str
        Which feature resolver was used upstream.
    resolver_info : dict, optional
        Metadata from the feature resolver.

    Returns
    -------
    ScoreEvaluation
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
        confusion_matrix,
        roc_curve,
    )
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    # Remove NaN pairs
    valid = ~(np.isnan(y_true) | np.isnan(y_score))
    if valid.sum() < len(valid):
        n_dropped = int((~valid).sum())
        logger.warning(f"Dropped {n_dropped} rows with NaN in outcome or score")
    y_true = y_true[valid]
    y_score = y_score[valid]

    n_samples = len(y_true)
    n_events = int(y_true.sum())
    event_rate = float(y_true.mean()) if n_samples > 0 else 0.0

    warnings: List[str] = []
    result = ScoreEvaluation(
        score_name=score_name,
        outcome_column=outcome_column,
        score_column=score_column,
        n_samples=n_samples,
        n_events=n_events,
        event_rate=event_rate,
        resolver_used=resolver_used,
        resolver_info=resolver_info or {},
    )

    if n_samples < 10:
        warnings.append(f"Very few samples ({n_samples})")
        result.warnings = warnings
        return result

    if len(np.unique(y_true)) < 2:
        warnings.append("Outcome has no variation (all 0 or all 1)")
        result.warnings = warnings
        return result

    # ---- Discrimination (threshold-free) ----
    try:
        result.auroc = float(roc_auc_score(y_true, y_score))
    except Exception as e:
        warnings.append(f"AUROC failed: {e}")

    try:
        result.auprc = float(average_precision_score(y_true, y_score))
    except Exception as e:
        warnings.append(f"AUPRC failed: {e}")

    # ---- Calibration ----
    try:
        result.brier_score = float(brier_score_loss(y_true, y_score))
    except Exception as e:
        warnings.append(f"Brier failed: {e}")

    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_score, n_bins=10, strategy="uniform"
        )
        result.calibration_prob_true = prob_true.tolist()
        result.calibration_prob_pred = prob_pred.tolist()
    except Exception as e:
        warnings.append(f"Calibration curve failed: {e}")

    # ---- Optimal threshold (Youden's J) ----
    if threshold is None:
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            threshold = float(thresholds[best_idx])
        except Exception:
            threshold = 0.5
    result.threshold = threshold

    # ---- Threshold-dependent metrics ----
    y_pred = (y_score >= threshold).astype(int)

    try:
        result.f1 = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception as e:
        warnings.append(f"F1 failed: {e}")

    try:
        result.ppv = float(precision_score(y_true, y_pred, zero_division=0))
    except Exception as e:
        warnings.append(f"PPV failed: {e}")

    try:
        result.sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception as e:
        warnings.append(f"Sensitivity failed: {e}")

    try:
        result.accuracy = float(accuracy_score(y_true, y_pred))
    except Exception as e:
        warnings.append(f"Accuracy failed: {e}")

    # NPV and specificity from confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        result.specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        result.npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    except Exception as e:
        warnings.append(f"Confusion matrix failed: {e}")

    # ---- Bootstrap CIs ----
    if n_bootstrap > 0 and n_samples >= 30:
        ci_results = _bootstrap_ci(
            y_true, y_score, threshold, n_bootstrap, ci_level
        )
        result.auroc_ci = ci_results.get("auroc")
        result.auprc_ci = ci_results.get("auprc")
        result.f1_ci = ci_results.get("f1")
        result.ppv_ci = ci_results.get("ppv")
        result.npv_ci = ci_results.get("npv")
        result.sensitivity_ci = ci_results.get("sensitivity")
        result.specificity_ci = ci_results.get("specificity")

    result.warnings = warnings
    return result


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    ci_level: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for all metrics.
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )

    rng = np.random.RandomState(42)
    n = len(y_true)
    alpha = (1 - ci_level) / 2

    metric_values: Dict[str, List[float]] = {
        "auroc": [],
        "auprc": [],
        "f1": [],
        "ppv": [],
        "npv": [],
        "sensitivity": [],
        "specificity": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]

        # Skip degenerate samples
        if len(np.unique(yt)) < 2:
            continue

        y_pred = (ys >= threshold).astype(int)

        try:
            metric_values["auroc"].append(roc_auc_score(yt, ys))
        except Exception:
            pass
        try:
            metric_values["auprc"].append(average_precision_score(yt, ys))
        except Exception:
            pass
        try:
            metric_values["f1"].append(f1_score(yt, y_pred, zero_division=0))
        except Exception:
            pass
        try:
            metric_values["ppv"].append(precision_score(yt, y_pred, zero_division=0))
        except Exception:
            pass
        try:
            metric_values["sensitivity"].append(recall_score(yt, y_pred, zero_division=0))
        except Exception:
            pass
        try:
            tn, fp, fn, tp = confusion_matrix(yt, y_pred, labels=[0, 1]).ravel()
            metric_values["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            metric_values["npv"].append(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
        except Exception:
            pass

    ci: Dict[str, Tuple[float, float]] = {}
    for name, vals in metric_values.items():
        if len(vals) >= 10:
            ci[name] = (
                float(np.percentile(vals, alpha * 100)),
                float(np.percentile(vals, (1 - alpha) * 100)),
            )

    return ci


# ---------------------------------------------------------------------------
# Multi-dataset comparison
# ---------------------------------------------------------------------------

def compare_evaluations(
    evaluations: Dict[str, ScoreEvaluation],
) -> pd.DataFrame:
    """
    Create a comparison table across multiple evaluations.

    Parameters
    ----------
    evaluations : dict
        Mapping of label → ScoreEvaluation.

    Returns
    -------
    pd.DataFrame
        One row per label, columns for every metric.
    """
    rows = []
    for label, ev in evaluations.items():
        row = {
            "dataset": label,
            "n_samples": ev.n_samples,
            "n_events": ev.n_events,
            "event_rate": ev.event_rate,
            "auroc": ev.auroc,
            "auprc": ev.auprc,
            "f1": ev.f1,
            "ppv": ev.ppv,
            "npv": ev.npv,
            "sensitivity": ev.sensitivity,
            "specificity": ev.specificity,
            "brier_score": ev.brier_score,
            "threshold": ev.threshold,
            "accuracy": ev.accuracy,
            "resolver": ev.resolver_used,
        }
        # Add CIs as separate columns
        for metric_name in ("auroc", "auprc", "f1", "ppv", "npv", "sensitivity", "specificity"):
            ci = getattr(ev, f"{metric_name}_ci", None)
            if ci is not None:
                row[f"{metric_name}_ci_lo"] = ci[0]
                row[f"{metric_name}_ci_hi"] = ci[1]

        rows.append(row)

    return pd.DataFrame(rows).set_index("dataset")


# ---------------------------------------------------------------------------
# Score distribution statistics
# ---------------------------------------------------------------------------

def score_distribution_stats(
    scores: pd.Series,
) -> Dict[str, float]:
    """
    Compute summary statistics for a score column.
    """
    valid = scores.dropna()
    return {
        "n_valid": len(valid),
        "n_missing": int(scores.isna().sum()),
        "mean": float(valid.mean()) if len(valid) > 0 else float("nan"),
        "std": float(valid.std()) if len(valid) > 0 else float("nan"),
        "min": float(valid.min()) if len(valid) > 0 else float("nan"),
        "q25": float(valid.quantile(0.25)) if len(valid) > 0 else float("nan"),
        "median": float(valid.median()) if len(valid) > 0 else float("nan"),
        "q75": float(valid.quantile(0.75)) if len(valid) > 0 else float("nan"),
        "max": float(valid.max()) if len(valid) > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_serializable(obj):
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj
