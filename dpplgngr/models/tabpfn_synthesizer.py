"""
TabPFN Synthesizer Wrapper for the Doppelganger Pipeline
=========================================================

Wraps TabPFN's unsupervised data generation capability (via tabpfn-extensions)
into an SDV-compatible interface: fit(df) / sample(num_rows=n) / save(filepath).

The TabPFNUnsupervisedModel from tabpfn-extensions uses the TabPFN prior-based
Bayesian framework to model the joint distribution of a tabular dataset and draw
new samples from it. This is fundamentally different from GAN/VAE-based approaches:
rather than training a generative network, it uses the pre-trained TabPFN
meta-prior to perform approximate Bayesian inference over the data distribution.

Key properties:
- Handles NaN natively (TabPFN was designed for incomplete data)
- Ordinal-encodes categorical columns internally; decodes after generation
- Configurable temperature and permutation averaging for quality/speed tradeoff
- Sklearn-compatible (fit/sample interface)
- SDV-compatible (accepts metadata= kwarg; ignores it)

Requires:
    pip install tabpfn
    pip install "tabpfn-extensions @ git+https://github.com/PriorLabs/tabpfn-extensions.git"

References:
    Hollmann et al. (2025) "Accurate predictions on small data with a tabular
    foundation model." Nature. https://doi.org/10.1038/s41586-024-08328-6

Author: SB
Date: 2026-03-11
"""

import logging
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TabPFNSynthesizer:
    """
    SDV-compatible wrapper around TabPFN's unsupervised synthetic data generation.

    Uses ``TabPFNUnsupervisedModel`` from ``tabpfn-extensions`` to generate
    synthetic tabular data via the TabPFN meta-learned prior. Categorical columns
    are ordinal-encoded before fitting and decoded after sampling. NaN values are
    preserved and handled natively by TabPFN.

    Parameters
    ----------
    metadata : ignored
        Accepted for API compatibility with SDV synthesizers; not used.
    n_estimators : int
        Number of TabPFN ensemble members used in the classifier and regressor.
        More estimators → better quality, slower runtime. Default: 8.
    device : str or None
        Inference device. One of ``'cuda'``, ``'cpu'``, or ``None`` (auto-select,
        prefers CUDA if available). Default: None.
    temperature : float
        Sampling temperature applied during generation.
        ``t=1.0`` is calibrated; ``t>1.0`` increases diversity (noisier);
        ``t<1.0`` sharpens the distribution (less diverse). Default: 1.0.
    n_permutations : int
        Number of column-order permutations to average during generation.
        More permutations → better quality, slower. Default: 3.

    Examples
    --------
    >>> synth = TabPFNSynthesizer(n_estimators=4, temperature=1.0)
    >>> synth.fit(df_train)
    >>> df_synthetic = synth.sample(num_rows=5000)
    >>> synth.save("synth_TabPFN.pkl")
    """

    def __init__(
        self,
        metadata=None,
        n_estimators: int = 8,
        device: Optional[str] = None,
        temperature: float = 1.0,
        n_permutations: int = 3,
    ):
        # metadata is accepted for SDV compatibility but is not used
        self.n_estimators = n_estimators
        self.device = device
        self.temperature = temperature
        self.n_permutations = n_permutations

        # State populated during fit()
        self._model = None
        self._column_names: Optional[List[str]] = None
        self._cat_columns: Optional[List[str]] = None
        self._cont_columns: Optional[List[str]] = None
        self._cat_encoders: Optional[Dict[str, Dict[str, int]]] = None
        self._original_dtypes: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode a DataFrame to a float32 numpy array suitable for TabPFN.

        Categorical columns are replaced with their ordinal integer codes.
        NaN values are preserved as ``float('nan')``.
        """
        df_enc = df[self._column_names].copy()
        for col in self._cat_columns:
            mapping = self._cat_encoders[col]
            df_enc[col] = df[col].apply(
                lambda x: float(mapping.get(str(x), np.nan)) if pd.notna(x) else np.nan
            )
        return df_enc.values.astype(np.float32)

    def _decode(self, X: np.ndarray) -> pd.DataFrame:
        """
        Decode a float32 numpy array back to a DataFrame with original dtypes.

        Categorical columns are decoded from their ordinal integer codes.
        Out-of-range ordinal values are clipped to valid category indices.
        """
        df = pd.DataFrame(X, columns=self._column_names)
        for col in self._cat_columns:
            rev = {v: k for k, v in self._cat_encoders[col].items()}
            n_cat = len(rev)

            def _decode_val(x, _rev=rev, _n=n_cat):
                if pd.isna(x):
                    return np.nan
                idx = max(0, min(int(round(float(x))), _n - 1))
                return _rev.get(idx, np.nan)

            df[col] = df[col].apply(_decode_val)

        # Restore integer dtype for columns that were originally integer
        for col in self._cont_columns:
            orig_dtype = self._original_dtypes.get(col)
            if orig_dtype is not None and pd.api.types.is_integer_dtype(orig_dtype):
                df[col] = pd.to_numeric(df[col], errors="coerce").round()

        return df

    # ------------------------------------------------------------------
    # Public interface (SDV-compatible)
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TabPFNSynthesizer":
        """
        Fit the synthesizer on the provided DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Training data. NaN values are permitted.

        Returns
        -------
        self
        """
        try:
            import torch
            from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
            from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
        except ImportError as exc:
            raise ImportError(
                "TabPFNSynthesizer requires 'tabpfn' and 'tabpfn-extensions'.\n"
                "Install with:\n"
                "  pip install tabpfn\n"
                "  pip install 'tabpfn-extensions @ "
                "git+https://github.com/PriorLabs/tabpfn-extensions.git'"
            ) from exc

        self._column_names = df.columns.tolist()
        self._original_dtypes = df.dtypes.to_dict()

        # Detect categorical columns (object, category, bool)
        self._cat_columns = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        self._cont_columns = [c for c in self._column_names if c not in self._cat_columns]

        # Build ordinal encoders: {column: {str_value: int_index, ...}}
        self._cat_encoders = {}
        for col in self._cat_columns:
            unique_vals = sorted(str(v) for v in df[col].dropna().unique())
            self._cat_encoders[col] = {v: i for i, v in enumerate(unique_vals)}
            logger.info(
                f"  [TabPFN] Encoded categorical '{col}': {len(unique_vals)} categories"
            )

        logger.info(
            f"[TabPFN] Fitting on {len(df)} rows × {len(self._column_names)} columns "
            f"({len(self._cont_columns)} continuous, {len(self._cat_columns)} categorical). "
            f"n_estimators={self.n_estimators}, device={self.device or 'auto'}"
        )

        X = self._encode(df)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        clf = TabPFNClassifier(n_estimators=self.n_estimators, device=self.device)
        reg = TabPFNRegressor(n_estimators=self.n_estimators, device=self.device)
        self._model = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
        self._model.fit(X_tensor)

        logger.info("[TabPFN] Fitting complete.")
        return self

    def sample(self, num_rows: int) -> pd.DataFrame:
        """
        Generate synthetic rows.

        Parameters
        ----------
        num_rows : int
            Number of synthetic samples to generate.

        Returns
        -------
        pd.DataFrame
            Synthetic data with the same columns and approximate dtypes as the
            training set.
        """
        if self._model is None:
            raise RuntimeError(
                "TabPFNSynthesizer must be fitted before calling sample()."
            )

        logger.info(
            f"[TabPFN] Generating {num_rows} synthetic samples "
            f"(temperature={self.temperature}, n_permutations={self.n_permutations})…"
        )

        synthetic_tensor = self._model.generate_synthetic_data(
            n_samples=num_rows,
            t=self.temperature,
            n_permutations=self.n_permutations,
        )

        synthetic_np = synthetic_tensor.detach().cpu().numpy()
        df_synth = self._decode(synthetic_np)

        logger.info(
            f"[TabPFN] Generation complete. "
            f"NaN count: {df_synth.isna().sum().sum()}"
        )
        return df_synth

    def save(self, filepath: str) -> None:
        """
        Pickle the fitted synthesizer to disk.

        Parameters
        ----------
        filepath : str
            Destination path (e.g. ``synth_TabPFN.pkl``).
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"[TabPFN] Synthesizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TabPFNSynthesizer":
        """Load a previously saved TabPFNSynthesizer from disk."""
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"[TabPFN] Synthesizer loaded from {filepath}")
        return obj
