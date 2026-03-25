"""
Feature Resolver
=================

Provides two strategies for ensuring that all features required by a risk
score calculator are present in a dataset:

1. **GraphReformer** – Pass the data through the trained GNN graph-imputer
   so that missing features are reconstructed from the latent-space
   representation.  This preserves inter-feature correlations learned
   during training.

2. **FeatureDropper** – Simply drop (remove) features that are absent
   from the dataset and let the score calculator handle the resulting NaNs.

3. **SimpleImputer** – Fill missing features with a simple strategy
   (median / mean / zero) without using the graph model.

All strategies implement the ``FeatureResolver`` interface.

Author: SB
Date: 2025-10-31
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class FeatureResolver(ABC):
    """
    Abstract interface for resolving missing features before score
    calculation.
    """

    name: str = ""

    @abstractmethod
    def resolve(
        self,
        data: pd.DataFrame,
        required_columns: Set[str],
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Ensure that *required_columns* are present (and as complete as
        possible) in *data*.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset (will **not** be modified in place).
        required_columns : set of str
            Column names that the downstream score calculator needs.

        Returns
        -------
        resolved_data : pd.DataFrame
            A copy of *data* with the required columns present.
        info : dict
            Metadata about what was done (e.g. which columns were
            imputed, confidence estimates, etc.).
        """
        ...


# ---------------------------------------------------------------------------
# Strategy 1 – Graph re-forming through the GNN latent space
# ---------------------------------------------------------------------------

class GraphReformer(FeatureResolver):
    """
    Resolve missing features by passing the data through a trained
    ``GraphImputationModel``.

    The model encodes each patient into a latent representation via the
    graph-attention layers and decodes back to feature space.  Features
    that were missing are filled with the model's prediction (plus
    uncertainty estimates).

    Parameters
    ----------
    model_path : str
        Path to a saved ``GraphImputationModel`` pickle file.
    model : object, optional
        A pre-loaded ``GraphImputationModel`` instance.  If given,
        *model_path* is ignored.
    """

    name = "graph_reformer"

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[object] = None,
    ):
        if model is not None:
            self._model = model
        elif model_path is not None:
            self._model = self._load_model(model_path)
        else:
            raise ValueError("Either model_path or model must be provided.")

    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(model_path: str):
        from dpplgngr.models.graph_imputer import GraphImputationModel

        logger.info(f"Loading graph imputation model from {model_path}")
        model = GraphImputationModel.load(model_path)
        logger.info(
            f"Model loaded – {len(model.feature_names)} features: "
            f"{model.feature_names[:5]}{'...' if len(model.feature_names) > 5 else ''}"
        )
        return model

    # ------------------------------------------------------------------

    def resolve(
        self,
        data: pd.DataFrame,
        required_columns: Set[str],
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Pass *data* through the graph model to impute missing features.

        Features that are in ``required_columns`` but NOT in the model's
        feature set are filled with median imputation as a fallback.
        """
        model = self._model
        model_features = set(model.feature_names)

        df = data.copy()

        # ---- 1. Add any columns the model knows about that are absent ----
        for feat in model.feature_names:
            if feat not in df.columns:
                df[feat] = np.nan

        # ---- 2. Prepare the model-feature matrix ----
        df_model = df[model.feature_names].copy()

        # Convert non-numeric types for safety
        for col in df_model.columns:
            if not pd.api.types.is_numeric_dtype(df_model[col]):
                df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

        missing_mask = df_model.isna().values
        n_missing = int(np.sum(missing_mask))
        logger.info(
            f"[GraphReformer] {n_missing} missing values "
            f"({np.mean(missing_mask):.1%}) across {len(df_model)} rows × "
            f"{len(df_model.columns)} model features"
        )

        if n_missing == 0:
            imputed_df = df_model
            uncertainty_info: Dict = {"mean_confidence": 1.0}
        else:
            X_imputed, uncertainty_info = model.impute(
                df_model.values, missing_mask, return_uncertainty=True
            )
            imputed_df = pd.DataFrame(
                X_imputed, columns=df_model.columns, index=df_model.index
            )
            logger.info(
                f"[GraphReformer] Imputation done – "
                f"mean confidence: {uncertainty_info['mean_confidence']:.4f}"
            )

        # ---- 3. Handle required columns NOT in the model ----
        extra_cols = required_columns - model_features
        extra_info: Dict = {}
        if extra_cols:
            logger.info(
                f"[GraphReformer] {len(extra_cols)} required columns not in "
                f"model → median imputation fallback"
            )
            for col in extra_cols:
                if col in df.columns:
                    median_val = df[col].median()
                    imputed_df[col] = df[col].fillna(median_val)
                    extra_info[col] = {"strategy": "median", "fill_value": median_val}
                else:
                    imputed_df[col] = 0.0
                    extra_info[col] = {"strategy": "constant", "fill_value": 0.0}

        # ---- 4. Keep any original columns that are not in model/extra ----
        other_cols = [c for c in data.columns if c not in imputed_df.columns]
        for col in other_cols:
            imputed_df[col] = data[col].values

        info = {
            "resolver": self.name,
            "n_missing_imputed": n_missing,
            "model_features": list(model_features),
            "extra_columns_fallback": extra_info,
            "uncertainty": {
                "mean_confidence": float(uncertainty_info.get("mean_confidence", 0)),
            },
        }

        return imputed_df, info


# ---------------------------------------------------------------------------
# Strategy 2 – Drop missing features
# ---------------------------------------------------------------------------

class FeatureDropper(FeatureResolver):
    """
    Simply ensure columns exist (filled with NaN) so the score calculator
    can proceed.  The calculator itself is expected to handle NaN rows
    (typically producing NaN scores for those patients).

    This is the "do nothing" strategy — no imputation, no modelling.
    """

    name = "feature_dropper"

    def resolve(
        self,
        data: pd.DataFrame,
        required_columns: Set[str],
    ) -> Tuple[pd.DataFrame, Dict]:
        df = data.copy()

        added: List[str] = []
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
                added.append(col)

        if added:
            logger.info(
                f"[FeatureDropper] Added {len(added)} missing columns "
                f"(filled with NaN): {sorted(added)}"
            )

        info = {
            "resolver": self.name,
            "added_columns": added,
            "n_added": len(added),
        }
        return df, info


# ---------------------------------------------------------------------------
# Strategy 3 – Simple statistical imputation
# ---------------------------------------------------------------------------

class SimpleFeatureImputer(FeatureResolver):
    """
    Fill missing values using a simple per-column strategy.

    Parameters
    ----------
    strategy : str
        One of ``"median"`` (default), ``"mean"``, ``"zero"``.
    """

    name = "simple_imputer"

    STRATEGIES = ("median", "mean", "zero")

    def __init__(self, strategy: str = "median"):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {self.STRATEGIES}"
            )
        self.strategy = strategy

    def resolve(
        self,
        data: pd.DataFrame,
        required_columns: Set[str],
    ) -> Tuple[pd.DataFrame, Dict]:
        df = data.copy()
        fill_info: Dict = {}

        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan

            # Coerce non-numeric columns (string, object, categorical)
            # to float so that median/mean aggregation works.
            if df[col].dtype.kind not in ("f", "i", "u", "b"):
                df[col] = pd.to_numeric(df[col], errors="coerce")

            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue

            if self.strategy == "median":
                fill_val = df[col].median()
            elif self.strategy == "mean":
                fill_val = df[col].mean()
            else:  # zero
                fill_val = 0.0

            # If entirely NaN, fall back to zero
            if pd.isna(fill_val):
                fill_val = 0.0

            df[col] = df[col].fillna(fill_val)
            fill_info[col] = {
                "strategy": self.strategy,
                "fill_value": float(fill_val) if not pd.isna(fill_val) else 0.0,
                "n_filled": n_missing,
            }

        if fill_info:
            logger.info(
                f"[SimpleFeatureImputer] Filled {len(fill_info)} columns "
                f"using '{self.strategy}' strategy"
            )

        info = {
            "resolver": self.name,
            "strategy": self.strategy,
            "columns_filled": fill_info,
        }
        return df, info


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_feature_resolver(
    method: str = "drop",
    **kwargs,
) -> FeatureResolver:
    """
    Convenience factory to create a ``FeatureResolver``.

    Parameters
    ----------
    method : str
        One of:
        * ``"graph"``   – :class:`GraphReformer`  (requires ``model_path`` or ``model``)
        * ``"drop"``    – :class:`FeatureDropper`
        * ``"simple"``  – :class:`SimpleFeatureImputer`  (accepts ``strategy``)
    **kwargs
        Forwarded to the resolver constructor.

    Returns
    -------
    FeatureResolver
    """
    method = method.lower()
    if method in ("graph", "graph_reformer"):
        return GraphReformer(**kwargs)
    elif method in ("drop", "feature_dropper"):
        return FeatureDropper()
    elif method in ("simple", "simple_imputer"):
        return SimpleFeatureImputer(**kwargs)
    else:
        raise ValueError(
            f"Unknown feature resolver method '{method}'. "
            "Choose from: 'graph', 'drop', 'simple'."
        )
