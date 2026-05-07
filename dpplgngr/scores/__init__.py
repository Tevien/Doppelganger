"""
Clinical Scores Module
=======================

This module provides a complete framework for:

1. **Calculating** clinical risk scores (MAGGIC, MARKER-HF, Cox PH, …)
2. **Resolving** missing features via graph-model latent-space re-forming,
   simple imputation, or dropping
3. **Evaluating** predictive performance (AUROC, AUPRC, F1, PPV, NPV,
   sensitivity, specificity, Brier score, calibration) with bootstrap CIs
4. **Comparing** performance across datasets and/or imputation strategies

Architecture
------------
* ``base``              – ``BaseScoreCalculator`` ABC and ``ScoreRegistry``
* ``feature_resolver``  – ``GraphReformer``, ``FeatureDropper``, ``SimpleFeatureImputer``
* ``metrics``           – ``evaluate_score``, ``compare_evaluations``
* ``pipeline``          – ``ScoreAnalysisPipeline`` (end-to-end orchestrator)

Available Scores
----------------
- **maggic**    – MAGGIC heart-failure mortality risk
- **marker_hf** – MARKER-HF BDT risk score
- **cox_ph**    – Cox proportional hazards (user-supplied coefficients)
- **score2**    – SCORE2 / SCORE2-OP cardiovascular risk
- **prevent**   – PREVENT 10-year and 30-year cardiovascular event risk
- **qrisk3**    – QRISK3 10-year cardiovascular disease risk

Quick Start
-----------
>>> from dpplgngr.scores import ScoreAnalysisPipeline
>>>
>>> result = ScoreAnalysisPipeline.run_single(
...     data="data/preprocessed.parquet",
...     score_name="maggic",
...     column_mapping={
...         "sex": "Female", "age": "Age_years", "lvef": "LVEF_percent",
...         # … remaining MAGGIC features
...     },
...     outcome_column="death_1yr",
...     feature_method="drop",        # or "graph" or "simple"
... )
>>> print(result.evaluation.summary())

Multi-dataset comparison:

>>> comparison = ScoreAnalysisPipeline.run_comparison(
...     datasets={
...         "real": "data/real.parquet",
...         "synthetic": "data/synth.parquet",
...     },
...     score_name="maggic",
...     column_mapping={...},
...     outcome_column="death_1yr",
...     feature_method="graph",
...     feature_kwargs={"model_path": "models/graph_imputer_model.pkl"},
... )
>>> print(comparison.comparison_table)

Strategy comparison (same data, different resolvers):

>>> comparison = ScoreAnalysisPipeline.run_strategy_comparison(
...     data="data/preprocessed.parquet",
...     score_name="maggic",
...     column_mapping={...},
...     outcome_column="death_1yr",
...     strategies={
...         "drop":   {"method": "drop"},
...         "median": {"method": "simple", "strategy": "median"},
...         "graph":  {"method": "graph", "model_path": "models/graph_imputer_model.pkl"},
...     },
... )
"""

# ---- Base framework ----
from dpplgngr.scores.base import BaseScoreCalculator, ScoreRegistry

# ---- Feature resolution ----
from dpplgngr.scores.feature_resolver import (
    FeatureResolver,
    GraphReformer,
    FeatureDropper,
    SimpleFeatureImputer,
    get_feature_resolver,
)

# ---- Metrics ----
from dpplgngr.scores.metrics import (
    ScoreEvaluation,
    evaluate_score,
    compare_evaluations,
    score_distribution_stats,
)

# ---- Pipeline ----
from dpplgngr.scores.pipeline import (
    ScoreAnalysisPipeline,
    SingleAnalysisResult,
    ComparisonResult,
)

# ---- Score calculators (importing triggers @ScoreRegistry.auto) ----
from dpplgngr.scores.maggic import calculateMAGGIC, MAGGICScore
from dpplgngr.scores.marker_hf import MARKERHFScore
from dpplgngr.scores.cox_predict import CoxPHPredictorWithUncertainty, CoxPHScore
from dpplgngr.scores.score2 import SCORE2Score
from dpplgngr.scores.prevent import PREVENTScore
from dpplgngr.scores.qrisk3 import QRISK3Score

# ---- Legacy Luigi tasks (backward-compatible) ----
from dpplgngr.scores.calculate_scores import (
    CalculateScores,
    calculate_scores_standalone,
    get_score_calculator,
)
from dpplgngr.scores.evaluate_scores import (
    EvaluateScorePerformance,
    evaluate_score_performance,
)

__all__ = [
    # Base framework
    "BaseScoreCalculator",
    "ScoreRegistry",
    # Feature resolution
    "FeatureResolver",
    "GraphReformer",
    "FeatureDropper",
    "SimpleFeatureImputer",
    "get_feature_resolver",
    # Metrics
    "ScoreEvaluation",
    "evaluate_score",
    "compare_evaluations",
    "score_distribution_stats",
    # Pipeline
    "ScoreAnalysisPipeline",
    "SingleAnalysisResult",
    "ComparisonResult",
    # Score calculators
    "calculateMAGGIC",
    "MAGGICScore",
    "MARKERHFScore",
    "CoxPHPredictorWithUncertainty",
    "CoxPHScore",
    "SCORE2Score",
    "PREVENTScore",
    "QRISK3Score",
    # Legacy Luigi tasks
    "CalculateScores",
    "EvaluateScorePerformance",
    "calculate_scores_standalone",
    "get_score_calculator",
    "evaluate_score_performance",
]

# Version information
__version__ = "3.0.0"
__author__ = "SB"
