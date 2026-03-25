"""
Score Analysis Pipeline
========================

End-to-end orchestrator that:

1. Loads a dataset (real or synthetic)
2. Resolves missing features (graph re-forming **or** dropping **or** simple imputation)
3. Calculates one or more clinical risk scores
4. Evaluates predictive performance against a binary outcome
5. Optionally compares across multiple datasets / strategies

This module is the main entry point for running a complete score analysis.

Usage
-----
**Minimal (single dataset, single score):**

>>> from dpplgngr.scores.pipeline import ScoreAnalysisPipeline
>>> results = ScoreAnalysisPipeline.run_single(
...     data_path="data/preprocessed.parquet",
...     score_name="maggic",
...     column_mapping={...},
...     outcome_column="death_1yr",
...     feature_method="drop",
... )
>>> print(results.evaluation.summary())

**Multi-dataset comparison:**

>>> results = ScoreAnalysisPipeline.run_comparison(
...     datasets={"real": "data/real.parquet", "synth": "data/synth.parquet"},
...     score_name="maggic",
...     column_mapping={...},
...     outcome_column="death_1yr",
...     feature_method="graph",
...     graph_model_path="models/graph_imputer_model.pkl",
... )
>>> print(results.comparison_table)

Author: SB
Date: 2025-10-31
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import json
import logging
import os

import numpy as np
import pandas as pd

from dpplgngr.scores.base import BaseScoreCalculator, ScoreRegistry
from dpplgngr.scores.feature_resolver import FeatureResolver, get_feature_resolver
from dpplgngr.scores.metrics import (
    ScoreEvaluation,
    compare_evaluations,
    evaluate_score,
    score_distribution_stats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SingleAnalysisResult:
    """Result of a single dataset × score analysis."""

    label: str
    score_name: str
    data: pd.DataFrame              # data with score columns added
    evaluation: ScoreEvaluation
    resolver_info: Dict = field(default_factory=dict)
    score_columns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "score_name": self.score_name,
            "evaluation": self.evaluation.to_dict(),
            "resolver_info": self.resolver_info,
            "score_columns": self.score_columns,
        }


@dataclass
class ComparisonResult:
    """Result of comparing scores across multiple datasets or strategies."""

    individual: Dict[str, SingleAnalysisResult]
    comparison_table: pd.DataFrame
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "individual": {k: v.to_dict() for k, v in self.individual.items()},
            "comparison_table": self.comparison_table.reset_index().to_dict(orient="records"),
            "metadata": self.metadata,
        }

    def save(self, path: str) -> None:
        """Save the comparison result as JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Comparison results saved to {path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ScoreAnalysisPipeline:
    """
    Orchestrates the full score-analysis workflow.

    This class is stateless — every method is a ``@staticmethod`` or
    ``@classmethod`` so you can call them without instantiation.
    """

    # ------------------------------------------------------------------
    # Single-dataset analysis
    # ------------------------------------------------------------------

    @staticmethod
    def run_single(
        data: Union[str, pd.DataFrame],
        score_name: str,
        column_mapping: Dict[str, str],
        outcome_column: str,
        score_column: Optional[str] = None,
        label: str = "dataset",
        feature_method: str = "drop",
        feature_kwargs: Optional[Dict] = None,
        threshold: Optional[float] = None,
        n_bootstrap: int = 1000,
    ) -> SingleAnalysisResult:
        """
        Run a full analysis on a single dataset.

        Parameters
        ----------
        data : str or pd.DataFrame
            Path to a dataset file (.parquet, .csv, .feather) or a
            DataFrame directly.
        score_name : str
            Name of the registered score calculator (e.g. ``"maggic"``).
        column_mapping : dict
            Maps canonical feature keys to dataset column names.
        outcome_column : str
            Name of the binary outcome column in the dataset.
        score_column : str, optional
            Which output column from the score calculator to evaluate.
            If ``None``, the first output column is used.
        label : str
            Human-readable label for this dataset.
        feature_method : str
            Feature resolution method: ``"graph"``, ``"drop"``, or
            ``"simple"``.
        feature_kwargs : dict, optional
            Extra keyword arguments for the feature resolver
            (e.g. ``model_path`` for graph reformer, ``strategy`` for
            simple imputer).
        threshold : float, optional
            Decision threshold for F1/PPV/NPV.  ``None`` = auto
            (Youden's J).
        n_bootstrap : int
            Number of bootstrap iterations for confidence intervals.

        Returns
        -------
        SingleAnalysisResult
        """
        # ---- 1. Load data ----
        df = ScoreAnalysisPipeline._load_data(data)
        logger.info(f"[{label}] Loaded data: {df.shape[0]} rows × {df.shape[1]} cols")

        # ---- 2. Get score calculator ----
        calc_cls = ScoreRegistry.get(score_name)
        calculator: BaseScoreCalculator = calc_cls(column_mapping=column_mapping)

        # ---- 3. Resolve features ----
        required_cols = calculator.get_mapped_columns()
        resolver = get_feature_resolver(feature_method, **(feature_kwargs or {}))
        df_resolved, resolver_info = resolver.resolve(df, required_cols)
        logger.info(
            f"[{label}] Features resolved via '{resolver.name}' – "
            f"{len(required_cols)} required columns"
        )

        # ---- 4. Calculate score ----
        score_result = calculator.calculate(df_resolved, column_mapping)
        score_cols = score_result.columns.tolist()

        # Merge score columns into resolved data
        for col in score_cols:
            df_resolved[col] = score_result[col]

        # ---- 5. Determine which score column to evaluate ----
        if score_column is None:
            # Pick the first output column from the calculator
            score_column = score_cols[0]
        if score_column not in df_resolved.columns:
            raise ValueError(
                f"Score column '{score_column}' not found. "
                f"Available: {score_cols}"
            )

        # ---- 6. Evaluate ----
        y_true = df_resolved[outcome_column].values if outcome_column in df_resolved.columns else None
        y_score = df_resolved[score_column].values

        if y_true is not None:
            evaluation = evaluate_score(
                y_true=y_true,
                y_score=y_score,
                score_name=score_name,
                outcome_column=outcome_column,
                score_column=score_column,
                threshold=threshold,
                n_bootstrap=n_bootstrap,
                resolver_used=resolver.name,
                resolver_info=resolver_info,
            )
        else:
            logger.warning(
                f"[{label}] Outcome column '{outcome_column}' not found – "
                "skipping evaluation"
            )
            evaluation = ScoreEvaluation(
                score_name=score_name,
                outcome_column=outcome_column,
                score_column=score_column,
                n_samples=len(df_resolved),
                n_events=0,
                event_rate=0.0,
                resolver_used=resolver.name,
                resolver_info=resolver_info,
                warnings=[f"Outcome column '{outcome_column}' not found"],
            )

        # ---- 7. Score distribution ----
        dist = score_distribution_stats(df_resolved[score_column])
        evaluation.score_mean = dist["mean"]
        evaluation.score_std = dist["std"]
        evaluation.score_min = dist["min"]
        evaluation.score_max = dist["max"]
        evaluation.score_median = dist["median"]
        evaluation.score_n_missing = dist["n_missing"]

        logger.info(f"[{label}] {evaluation.summary()}")

        return SingleAnalysisResult(
            label=label,
            score_name=score_name,
            data=df_resolved,
            evaluation=evaluation,
            resolver_info=resolver_info,
            score_columns=score_cols,
        )

    # ------------------------------------------------------------------
    # Multi-dataset / multi-strategy comparison
    # ------------------------------------------------------------------

    @staticmethod
    def run_comparison(
        datasets: Dict[str, Union[str, pd.DataFrame]],
        score_name: str,
        column_mapping: Dict[str, str],
        outcome_column: str,
        score_column: Optional[str] = None,
        feature_method: str = "drop",
        feature_kwargs: Optional[Dict] = None,
        threshold: Optional[float] = None,
        n_bootstrap: int = 1000,
        output_path: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Run the same score analysis on multiple datasets and compare.

        Parameters
        ----------
        datasets : dict
            Mapping of ``label → path_or_dataframe``.
        score_name, column_mapping, outcome_column, score_column,
        feature_method, feature_kwargs, threshold, n_bootstrap
            Same as :meth:`run_single`.
        output_path : str, optional
            If given, save the comparison JSON to this path.

        Returns
        -------
        ComparisonResult
        """
        individual: Dict[str, SingleAnalysisResult] = {}

        for label, data_src in datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Analysing: {label}")
            logger.info(f"{'='*60}")

            result = ScoreAnalysisPipeline.run_single(
                data=data_src,
                score_name=score_name,
                column_mapping=column_mapping,
                outcome_column=outcome_column,
                score_column=score_column,
                label=label,
                feature_method=feature_method,
                feature_kwargs=feature_kwargs,
                threshold=threshold,
                n_bootstrap=n_bootstrap,
            )
            individual[label] = result

        # Build comparison table
        evals = {k: v.evaluation for k, v in individual.items()}
        comparison_table = compare_evaluations(evals)

        logger.info(f"\n{'='*60}")
        logger.info("COMPARISON TABLE")
        logger.info(f"{'='*60}")
        logger.info(f"\n{comparison_table.to_string()}")

        comp = ComparisonResult(
            individual=individual,
            comparison_table=comparison_table,
            metadata={
                "score_name": score_name,
                "outcome_column": outcome_column,
                "feature_method": feature_method,
                "n_datasets": len(datasets),
            },
        )

        if output_path:
            comp.save(output_path)

        return comp

    # ------------------------------------------------------------------
    # Multi-strategy comparison (same data, different resolvers)
    # ------------------------------------------------------------------

    @staticmethod
    def run_strategy_comparison(
        data: Union[str, pd.DataFrame],
        score_name: str,
        column_mapping: Dict[str, str],
        outcome_column: str,
        score_column: Optional[str] = None,
        strategies: Optional[Dict[str, Dict]] = None,
        threshold: Optional[float] = None,
        n_bootstrap: int = 1000,
        output_path: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Compare different feature-resolution strategies on the *same*
        dataset.

        Parameters
        ----------
        data : str or pd.DataFrame
            The single dataset to analyse.
        strategies : dict, optional
            Mapping of ``label → {"method": ..., **kwargs}``.
            Default strategies: ``drop``, ``simple_median``,
            ``simple_mean``.
        output_path : str, optional
            If given, save the comparison JSON to this path.

        Returns
        -------
        ComparisonResult
        """
        if strategies is None:
            strategies = {
                "drop_features": {"method": "drop"},
                "median_impute": {"method": "simple", "strategy": "median"},
                "mean_impute": {"method": "simple", "strategy": "mean"},
            }

        individual: Dict[str, SingleAnalysisResult] = {}

        for label, strategy_cfg in strategies.items():
            method = strategy_cfg.pop("method")
            logger.info(f"\n{'='*60}")
            logger.info(f"Strategy: {label} (method={method})")
            logger.info(f"{'='*60}")

            result = ScoreAnalysisPipeline.run_single(
                data=data,
                score_name=score_name,
                column_mapping=column_mapping,
                outcome_column=outcome_column,
                score_column=score_column,
                label=label,
                feature_method=method,
                feature_kwargs=strategy_cfg,
                threshold=threshold,
                n_bootstrap=n_bootstrap,
            )
            individual[label] = result
            # Restore method key so the dict is reusable
            strategy_cfg["method"] = method

        evals = {k: v.evaluation for k, v in individual.items()}
        comparison_table = compare_evaluations(evals)

        logger.info(f"\n{'='*60}")
        logger.info("STRATEGY COMPARISON TABLE")
        logger.info(f"{'='*60}")
        logger.info(f"\n{comparison_table.to_string()}")

        comp = ComparisonResult(
            individual=individual,
            comparison_table=comparison_table,
            metadata={
                "score_name": score_name,
                "outcome_column": outcome_column,
                "strategies": list(strategies.keys()),
                "n_strategies": len(strategies),
            },
        )

        if output_path:
            comp.save(output_path)

        return comp

    # ------------------------------------------------------------------
    # Config-driven run
    # ------------------------------------------------------------------

    @staticmethod
    def run_from_config(config_path: str) -> ComparisonResult:
        """
        Run a full analysis from a JSON configuration file.

        Expected config structure::

            {
                "score_name": "maggic",
                "column_mapping": { ... },
                "outcome_column": "death_1yr",
                "score_column": "maggic (1-year risk of death)",
                "feature_method": "graph",
                "feature_kwargs": {
                    "model_path": "models/graph_imputer_model.pkl"
                },
                "datasets": {
                    "real_data": "data/real.parquet",
                    "synthetic_sg": "data/synth_sg.parquet"
                },
                "output_path": "results/score_comparison.json",
                "threshold": null,
                "n_bootstrap": 1000
            }

        Parameters
        ----------
        config_path : str
            Path to the JSON configuration file.

        Returns
        -------
        ComparisonResult
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        logger.info(f"Running score analysis from config: {config_path}")

        return ScoreAnalysisPipeline.run_comparison(
            datasets=config["datasets"],
            score_name=config["score_name"],
            column_mapping=config["column_mapping"],
            outcome_column=config["outcome_column"],
            score_column=config.get("score_column"),
            feature_method=config.get("feature_method", "drop"),
            feature_kwargs=config.get("feature_kwargs"),
            threshold=config.get("threshold"),
            n_bootstrap=config.get("n_bootstrap", 1000),
            output_path=config.get("output_path"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_data(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Load data from file path or return DataFrame directly."""
        if isinstance(data, pd.DataFrame):
            return data.copy()

        path = str(data)
        ext = Path(path).suffix.lower()

        if ext == ".parquet":
            return pd.read_parquet(path)
        elif ext == ".csv":
            return pd.read_csv(path)
        elif ext == ".feather":
            return pd.read_feather(path)
        elif ext == ".json":
            return pd.read_json(path)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                "Supported: .parquet, .csv, .feather, .json, .xlsx"
            )
