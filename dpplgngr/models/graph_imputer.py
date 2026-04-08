"""
Graph Neural Network-based Imputation Model
============================================

This module provides a graph-based imputation model that can handle missing data
by leveraging patient similarity through graph structure. It produces synthetic
data with t-SNE distributions matching the original data.

Author: SB
Date: 2025-10-29
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import pickle
import io
import os
from typing import Optional, Tuple, Dict, List, Union
import logging

logger = logging.getLogger(__name__)


class EnhancedGNNImputer(nn.Module):
    """Enhanced Graph Neural Network for missing feature imputation with uncertainty estimation."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 4, 
                 num_heads: int = 8, feature_types: Optional[List[str]] = None):
        super(EnhancedGNNImputer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feature_types = feature_types or ['continuous'] * input_dim
        
        # Feature embedding layer to handle missing values
        self.feature_embedding = nn.ModuleList([
            nn.Linear(1, hidden_dim // 4) for _ in range(input_dim)
        ])
        
        # Missing indicator embeddings
        self.missing_embedding = nn.Embedding(2, hidden_dim // 4)  # 0: observed, 1: missing
        
        # Feature type embeddings
        self.type_embedding = nn.Embedding(len(set(self.feature_types)), hidden_dim // 4)
        
        # Input projection (from combined embedding size to hidden_dim)
        # Combined embedding size: 3 * (hidden_dim // 4) = 3 * hidden_dim // 4
        embed_dim = 3 * (hidden_dim // 4)
        self.input_projection = nn.Linear(embed_dim, hidden_dim)
        
        # Multi-layer graph attention network with residual connections
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # First layer
        self.gnn_layers.append(TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.2))
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gnn_layers.append(TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.2))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Final layer
        self.gnn_layers.append(TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.2))
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Feature-specific prediction heads with uncertainty estimation
        self.feature_predictors = nn.ModuleList()
        self.uncertainty_estimators = nn.ModuleList()
        
        for i, ftype in enumerate(self.feature_types):
            # Mean prediction
            self.feature_predictors.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            ))
            
            # Uncertainty estimation (log variance)
            self.uncertainty_estimators.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Softplus()  # Ensure positive variance
            ))
        
        # Global confidence estimator
        self.global_confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                missing_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Node features [N, D] with missing values as NaN or 0
            edge_index: Graph connectivity
            missing_mask: Boolean mask indicating missing features [N, D]
            
        Returns:
            predictions: Imputed values [N, D]
            uncertainties: Uncertainty estimates [N, D]
            global_conf: Global confidence scores [N, 1]
        """
        batch_size, n_features = x.shape
        device = x.device
        
        # Create embeddings for each feature
        feature_embeds = []
        for i in range(n_features):
            # Feature value embedding
            feat_embed = self.feature_embedding[i](x[:, i:i+1])
            
            # Missing indicator embedding
            missing_embed = self.missing_embedding(missing_mask[:, i].long())
            
            # Feature type embedding
            type_idx = torch.tensor([hash(self.feature_types[i]) % len(set(self.feature_types))], device=device)
            type_embed = self.type_embedding(type_idx).expand(batch_size, -1)
            
            # Combine embeddings
            combined_embed = torch.cat([feat_embed, missing_embed, type_embed], dim=1)
            feature_embeds.append(combined_embed)
        
        # Combine all feature embeddings
        h = torch.stack(feature_embeds, dim=1)  # [N, D, embed_dim] where embed_dim = 3*(hidden_dim//4)
        h = torch.mean(h, dim=1)  # Pool across features: [N, embed_dim]
        
        # Input projection (embed_dim -> hidden_dim)
        h = self.input_projection(h)  # [N, hidden_dim]
        
        # Graph message passing with residual connections
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_residual = h
            h = gnn_layer(h, edge_index)
            h = layer_norm(h + h_residual)  # Residual connection
            
            if i < len(self.gnn_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=0.2, training=self.training)
        
        # Predict missing features with uncertainty
        predictions = []
        uncertainties = []
        
        for i in range(n_features):
            pred_mean = self.feature_predictors[i](h)
            pred_var = self.uncertainty_estimators[i](h)
            
            predictions.append(pred_mean.squeeze())
            uncertainties.append(pred_var.squeeze())
        
        predictions = torch.stack(predictions, dim=1)
        uncertainties = torch.stack(uncertainties, dim=1)
        
        # Global confidence
        global_conf = self.global_confidence(h)
        
        return predictions, uncertainties, global_conf


class GraphImputationModel:
    """
    Advanced Graph-based Imputation Model with t-SNE distribution matching.
    
    This model uses graph neural networks to impute missing values while preserving
    the statistical properties and t-SNE distributions of the original data.
    """
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, num_heads: int = 8,
                 k_neighbors: int = 20, learning_rate: float = 0.001, epochs: int = 300,
                 feature_types: Optional[List[str]] = None, device: Optional[str] = None,
                 allow_missing_training: bool = False):
        """
        Initialize the Graph Imputation Model.
        
        Args:
            hidden_dim: Dimension of hidden layers
            num_layers: Number of GNN layers
            num_heads: Number of attention heads in graph attention
            k_neighbors: Number of neighbors to connect in graph
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            feature_types: List of feature types ('continuous' or 'categorical')
            device: Device to use ('cuda' or 'cpu')
            allow_missing_training: If True, allows training on data with pre-existing missing values
                                    This enables the model to learn from incomplete data rather than
                                    only from artificially created missing scenarios
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.k_neighbors = k_neighbors
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.feature_types = feature_types
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.feature_means = None
        self.feature_stds = None
        self.allow_missing_training = allow_missing_training
        self.is_integer = None  # Track which features should be integers
        self.feature_min = None  # Track min values for clipping
        self.feature_max = None  # Track max values for clipping
        
        logger.info(f"Initialized GraphImputationModel on device: {self.device}")
        if allow_missing_training:
            logger.info("Missing data training enabled: model will learn from real missing patterns")
        
    # ------------------------------------------------------------------
    # Vectorised graph construction (replaces O(N²) Python loop)
    # ------------------------------------------------------------------

    def _create_adaptive_graph(self, X: np.ndarray, missing_mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create k-NN graph that adapts to missing-data patterns.

        Strategy
        --------
        1.  Group samples by their observation pattern (which features are
            observed).  For clinical data with d=12 features there are at most
            2^12 = 4096 patterns, so the number of groups is small.
        2.  Within each group *all* samples share the same observed feature set,
            so pairwise distances can be computed with a single vectorised call.
        3.  Between groups that share ≥ 2 common observed features, distances
            are computed on the common subset — again vectorised.
        4.  Per sample, the k nearest neighbours across all groups are kept.

        This replaces the previous O(N²) pure-Python double loop and is
        typically 100–500× faster for N > 1 000.
        """
        n_samples, n_features = X.shape
        k = self.k_neighbors

        # --- 1. Group samples by observation pattern -----------------------
        obs_mask = ~missing_mask  # True = observed
        # Represent each pattern as a tuple for hashing
        pattern_to_indices: dict[tuple, np.ndarray] = {}
        pattern_keys = [tuple(row) for row in obs_mask]
        for idx, key in enumerate(pattern_keys):
            if key not in pattern_to_indices:
                pattern_to_indices[key] = []
            pattern_to_indices[key].append(idx)
        pattern_to_indices = {k: np.array(v) for k, v in pattern_to_indices.items()}

        patterns = list(pattern_to_indices.keys())
        n_patterns = len(patterns)
        logger.debug(f"Graph construction: {n_samples} samples, {n_patterns} observation patterns")

        # Pre-allocate per-sample neighbour arrays (distances + indices)
        # We keep the top-k closest across all pattern-pair comparisons.
        # Using lists-of-lists and consolidating at the end is fastest for
        # the typical case (few patterns, many samples).
        all_nn_dists = [[] for _ in range(n_samples)]   # list[list[float]]
        all_nn_idxs  = [[] for _ in range(n_samples)]   # list[list[int]]

        # --- 2. Compute pairwise distances per pattern pair ----------------
        from scipy.spatial.distance import cdist

        for pi in range(n_patterns):
            pat_i = np.array(patterns[pi], dtype=bool)
            idxs_i = pattern_to_indices[patterns[pi]]
            if pat_i.sum() == 0:
                continue

            for pj in range(pi, n_patterns):
                pat_j = np.array(patterns[pj], dtype=bool)
                idxs_j = pattern_to_indices[patterns[pj]]
                if pat_j.sum() == 0:
                    continue

                common = pat_i & pat_j
                if common.sum() < 2:
                    continue

                Xi = X[np.ix_(idxs_i, common)]
                Xj = X[np.ix_(idxs_j, common)]

                # cdist is C-level and handles (|I|, |J|) in one call
                D = cdist(Xi, Xj, metric='euclidean')  # (|I|, |J|)

                same_group = (pi == pj)

                # For each row in idxs_i, find the k closest in idxs_j
                k_local = min(k, D.shape[1] - (1 if same_group else 0))
                if k_local <= 0:
                    continue

                for li, gi in enumerate(idxs_i):
                    row = D[li]
                    if same_group:
                        # Exclude self-distance (which is 0)
                        row = row.copy()
                        self_col = np.searchsorted(idxs_j, gi)
                        if self_col < len(idxs_j) and idxs_j[self_col] == gi:
                            row[self_col] = np.inf

                    topk = min(k, len(row))
                    if topk >= len(row):
                        part_idx = np.arange(len(row))
                    else:
                        part_idx = np.argpartition(row, topk)[:topk]
                    all_nn_dists[gi].extend(row[part_idx].tolist())
                    all_nn_idxs[gi].extend(idxs_j[part_idx].tolist())

                # If pi != pj, also collect neighbours for idxs_j → idxs_i
                if not same_group:
                    for lj, gj in enumerate(idxs_j):
                        col = D[:, lj]
                        topk = min(k, len(col))
                        if topk >= len(col):
                            part_idx = np.arange(len(col))
                        else:
                            part_idx = np.argpartition(col, topk)[:topk]
                        all_nn_dists[gj].extend(col[part_idx].tolist())
                        all_nn_idxs[gj].extend(idxs_i[part_idx].tolist())

        # --- 3. Per-sample: keep global top-k neighbours ------------------
        src_list = []
        dst_list = []
        w_list   = []

        for i in range(n_samples):
            dists = np.array(all_nn_dists[i])
            idxs  = np.array(all_nn_idxs[i], dtype=np.intp)
            if len(dists) == 0:
                continue
            k_actual = min(k, len(dists))
            if k_actual >= len(dists):
                best = np.arange(len(dists))
            else:
                best = np.argpartition(dists, k_actual)[:k_actual]
            for b in best:
                src_list.append(i)
                dst_list.append(idxs[b])
                w_list.append(float(np.exp(-dists[b])))

        if not src_list:
            # Fallback: chain-connect first few samples
            for i in range(min(n_samples - 1, 10)):
                src_list.append(i); dst_list.append(i + 1); w_list.append(1.0)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weights = torch.tensor(w_list, dtype=torch.float)

        # Make undirected + self-loops
        edge_index, edge_weights = to_undirected(edge_index, edge_weights, num_nodes=n_samples)
        edge_index, edge_weights = add_self_loops(edge_index, edge_weights, fill_value=1.0, num_nodes=n_samples)

        return edge_index, edge_weights
    
    def _create_training_scenarios(self, X: np.ndarray, missing_rates: List[float] = [0.1, 0.2, 0.3, 0.4]
                                   ) -> List[Tuple[float, np.ndarray]]:
        """Create hierarchical training scenarios with progressively more missing data.

        Vectorised: generates the mask for all samples at once instead of
        looping row-by-row in Python.
        """
        n_samples, n_features = X.shape
        training_scenarios = []

        for rate in missing_rates:
            n_missing = int(rate * n_features)
            if n_missing == 0:
                continue
            for _ in range(10):  # Multiple scenarios per rate
                # For each sample, independently choose n_missing features to mask.
                # argsort of uniform randoms gives a random permutation per row.
                rand = np.random.rand(n_samples, n_features)
                order = rand.argsort(axis=1)  # (N, D) — per-row permutation
                missing_mask = np.zeros((n_samples, n_features), dtype=bool)
                # Mark the first n_missing columns of each permuted row
                cols = order[:, :n_missing]                     # (N, n_missing)
                rows = np.arange(n_samples)[:, None]            # (N, 1)
                missing_mask[rows, cols] = True
                training_scenarios.append((rate, missing_mask))

        return training_scenarios
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None, verbose: bool = True) -> 'GraphImputationModel':
        """
        Train the graph imputation model.
        
        Args:
            X: Training data. If allow_missing_training=True, can contain NaN values.
               If allow_missing_training=False (default), must be complete data.
            y: Optional target labels (not used for imputation but stored for reference)
            feature_names: List of feature names
            verbose: Whether to print training progress
            
        Returns:
            self: Fitted model
        """
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Ensure float64 — nullable integer dtypes (Int64) become object arrays
        # after .values, which breaks np.isnan.
        if X.dtype.kind not in ('f', 'c'):
            X = X.astype(np.float64)
        
        # Detect pre-existing missing data
        has_missing = np.isnan(X).any()
        real_missing_mask = np.isnan(X)
        n_missing_total = np.sum(real_missing_mask)
        
        if has_missing:
            if not self.allow_missing_training:
                raise ValueError(
                    f"Training data contains {n_missing_total} missing values, but allow_missing_training=False. "
                    "Either remove missing values or set allow_missing_training=True."
                )
            else:
                missing_rate = n_missing_total / X.size
                if verbose:
                    logger.info(f"Training data contains {n_missing_total} missing values ({missing_rate:.2%} missing rate)")
                    logger.info("Using real missing patterns for training")
        
        # Scale features (handle NaN values during scaling)
        if has_missing:
            # Use nanmean/nanstd for scaling with missing data
            from sklearn.impute import SimpleImputer
            temp_imputer = SimpleImputer(strategy='mean')
            X_for_scaling = temp_imputer.fit_transform(X)
            self.scaler.fit(X_for_scaling)
            X_scaled = self.scaler.transform(X_for_scaling)
            # Restore NaN values after scaling
            X_scaled[real_missing_mask] = np.nan
            # Calculate means/stds ignoring NaN
            self.feature_means = np.nanmean(X_scaled, axis=0)
            self.feature_stds = np.nanstd(X_scaled, axis=0)
            # Replace any remaining NaN in means/stds with 0/1
            self.feature_means = np.nan_to_num(self.feature_means, nan=0.0)
            self.feature_stds = np.nan_to_num(self.feature_stds, nan=1.0)
        else:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_means = np.mean(X_scaled, axis=0)
            self.feature_stds = np.std(X_scaled, axis=0)
        
        # Detect feature types if not provided
        if self.feature_types is None:
            self.feature_types = ['continuous'] * X_scaled.shape[1]
        
        # Detect integer features
        self._detect_integer_features(X, verbose=verbose)
        
        # Initialize model
        input_dim = X_scaled.shape[1]
        self.model = EnhancedGNNImputer(
            input_dim, self.hidden_dim, self.num_layers, 
            self.num_heads, self.feature_types
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        
        # Create training scenarios
        # If data has real missing values and allow_missing_training=True, use them
        if has_missing and self.allow_missing_training:
            # Use real missing patterns + augment with artificial scenarios
            training_scenarios = []
            
            # Add real missing pattern
            training_scenarios.append(('real', real_missing_mask.copy()))
            
            # Add augmented scenarios based on real missing patterns
            for _ in range(20):  # Create variations
                augmented_mask = real_missing_mask.copy()
                # Randomly flip some missing/observed status to create variations
                flip_rate = 0.1
                flip_mask = np.random.rand(*augmented_mask.shape) < flip_rate
                augmented_mask = augmented_mask ^ flip_mask  # XOR to flip
                training_scenarios.append(('augmented', augmented_mask))
            
            # Also add some artificially created scenarios for robustness
            artificial_scenarios = self._create_training_scenarios(X_scaled, missing_rates=[0.1, 0.2, 0.3])
            training_scenarios.extend(artificial_scenarios)
            
            if verbose:
                logger.info(f"Using real missing patterns: {n_missing_total} missing values")
                logger.info(f"Created {len(training_scenarios)} training scenarios (including real + augmented)")
        else:
            # Original behavior: create artificial missing scenarios
            training_scenarios = self._create_training_scenarios(X_scaled)
            if verbose:
                logger.info(f"Created {len(training_scenarios)} artificial training scenarios")

        # ------------------------------------------------------------------
        # PRE-BUILD GRAPHS + TENSORS for every scenario (one-time cost)
        # ------------------------------------------------------------------
        # Previously, _create_adaptive_graph was called *inside* the epoch
        # loop for every selected scenario — 5 calls × 300 epochs = 1 500
        # graph constructions.  The missing masks are fixed across epochs, so
        # we build each graph once, move the tensors to the device, and reuse.
        # ------------------------------------------------------------------
        import time as _time

        if verbose:
            logger.info(f"Pre-building graphs for {len(training_scenarios)} scenarios …")
            _t_graph_start = _time.time()

        scenario_cache: list[dict] = []   # one dict per scenario

        for si, (rate, scen_mask) in enumerate(training_scenarios):
            # ---------- input / target arrays ----------------------------
            X_input = np.nan_to_num(X_scaled.copy(), nan=0.0)
            X_input[scen_mask] = 0.0

            X_target_clean = np.nan_to_num(X_scaled.copy(), nan=0.0)

            # trainable mask: positions we artificially masked AND where
            # the real data was observed (so we have ground truth).
            if has_missing and self.allow_missing_training:
                trainable_mask = scen_mask & ~real_missing_mask
            else:
                trainable_mask = scen_mask

            # ---------- graph -------------------------------------------
            edge_index, edge_weights = self._create_adaptive_graph(X_input, scen_mask)

            # ---------- move everything to device once -------------------
            entry = {
                "rate":          rate,
                "X_input":       torch.tensor(X_input, dtype=torch.float32).to(self.device),
                "X_target":      torch.tensor(X_target_clean, dtype=torch.float32).to(self.device),
                "scen_mask":     torch.tensor(scen_mask, dtype=torch.bool).to(self.device),
                "trainable_mask":torch.tensor(trainable_mask, dtype=torch.bool).to(self.device),
                "n_trainable":   int(trainable_mask.sum()),
                "edge_index":    edge_index.to(self.device),
            }
            scenario_cache.append(entry)

            if verbose and (si + 1) % 10 == 0:
                _elapsed = _time.time() - _t_graph_start
                logger.info(f"  … {si + 1}/{len(training_scenarios)} graphs built ({_elapsed:.1f}s)")

        if verbose:
            _t_graph_total = _time.time() - _t_graph_start
            logger.info(f"All {len(training_scenarios)} graphs built in {_t_graph_total:.1f}s")

        if verbose:
            logger.info("="*80)
            logger.info("TRAINING DETAILS")
            logger.info("="*80)
            logger.info(f"Training samples: {len(X)}")
            logger.info(f"Number of features: {X_scaled.shape[1]}")
            logger.info(f"Training epochs: {self.epochs}")
            logger.info(f"Device: {self.device}")
            
            # GPU information if available
            if torch.cuda.is_available():
                logger.info(f"GPU available: YES")
                logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                logger.info(f"Current GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
                logger.info(f"Current GPU memory cached: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
            else:
                logger.info(f"GPU available: NO (using CPU)")
            
            # Model information
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Model memory size: {total_params * 4 / 1e6:.2f} MB (float32)")
            
            logger.info(f"Training scenarios: {len(training_scenarios)}")
            logger.info(f"Learning rate: {self.learning_rate}")
            logger.info(f"K-neighbors: {self.k_neighbors}")
            logger.info("="*80)
            logger.info("Starting training...")
            logger.info("="*80)
        
        # Training metrics tracking
        best_loss = float('inf')
        best_epoch = 0
        loss_history = []
        _train_start = _time.time()
        
        for epoch in range(self.epochs):
            _epoch_start = _time.time()
            self.model.train()
            epoch_losses = []
            epoch_mse_losses = []
            epoch_uncertainty_losses = []
            
            # Sample training scenarios (focus more on hard scenarios in later epochs)
            # Also prioritize real missing patterns if available
            scenario_weights = []
            for entry in scenario_cache:
                rate = entry["rate"]
                if rate == 'real':
                    weight = 5  # High priority for real missing patterns
                elif rate == 'augmented':
                    weight = 3  # Medium-high priority for augmented patterns
                elif epoch < self.epochs // 3:
                    weight = 3 if rate <= 0.2 else 1
                elif epoch < 2 * self.epochs // 3:
                    weight = 2 if rate <= 0.2 else 2 if rate <= 0.3 else 1
                else:
                    weight = 1 if rate <= 0.2 else 2 if rate <= 0.3 else 3
                scenario_weights.append(weight)
            
            # Sample a few scenarios per epoch
            n_scenarios_per_epoch = min(5, len(scenario_cache))
            selected_scenarios = np.random.choice(
                len(scenario_cache), n_scenarios_per_epoch, 
                p=np.array(scenario_weights) / np.sum(scenario_weights)
            )
            
            for scenario_idx in selected_scenarios:
                entry = scenario_cache[scenario_idx]
                rate = entry["rate"]
                
                # All tensors are already on device — no copies needed
                X_tensor       = entry["X_input"]
                X_target       = entry["X_target"]
                scen_mask_t    = entry["scen_mask"]
                trainable_mask_t = entry["trainable_mask"]
                edge_index     = entry["edge_index"]
                n_trainable    = entry["n_trainable"]
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions, uncertainties, global_conf = self.model(X_tensor, edge_index, scen_mask_t)
                
                # Reconstruction loss
                if has_missing and self.allow_missing_training:
                    if n_trainable > 0:
                        mse_loss = F.mse_loss(predictions[trainable_mask_t], X_target[trainable_mask_t])
                    else:
                        mse_loss = torch.tensor(0.0, device=self.device)
                else:
                    mse_loss = F.mse_loss(predictions[scen_mask_t], X_target[scen_mask_t])
                
                # Uncertainty loss (encourage well-calibrated uncertainty)
                if has_missing and self.allow_missing_training:
                    if n_trainable > 0:
                        uncertainty_loss = torch.mean(
                            torch.log(uncertainties[trainable_mask_t] + 1e-6) + 
                            (predictions[trainable_mask_t] - X_target[trainable_mask_t])**2 / 
                            (2 * uncertainties[trainable_mask_t] + 1e-6)
                        )
                    else:
                        uncertainty_loss = torch.tensor(0.0, device=self.device)
                else:
                    uncertainty_loss = torch.mean(
                        torch.log(uncertainties[scen_mask_t] + 1e-6) + 
                        (predictions[scen_mask_t] - X_target[scen_mask_t])**2 / 
                        (2 * uncertainties[scen_mask_t] + 1e-6)
                    )
                
                # Combined loss
                if isinstance(rate, str):  # 'real' or 'augmented'
                    weight_uncertainty = 0.2  # Higher weight for real patterns
                else:
                    weight_uncertainty = 0.1 + 0.1 * rate
                total_loss = mse_loss + weight_uncertainty * uncertainty_loss
                
                if total_loss.requires_grad:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_losses.append(total_loss.item())
                    epoch_mse_losses.append(mse_loss.item())
                    epoch_uncertainty_losses.append(uncertainty_loss.item())
            
            scheduler.step()
            
            # Track metrics
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            avg_mse = np.mean(epoch_mse_losses) if epoch_mse_losses else 0
            avg_uncertainty = np.mean(epoch_uncertainty_losses) if epoch_uncertainty_losses else 0
            loss_history.append(avg_loss)
            
            # Warn if no training occurred in this epoch
            if not epoch_losses:
                logger.warning(f"Epoch {epoch}: No trainable scenarios found! Model parameters were not updated.")
                logger.warning("This may indicate all data is pre-existing missing values with no ground truth for training.")
            
            # Update best loss
            if avg_loss < best_loss and epoch_losses:  # Only update if we had training
                best_loss = avg_loss
                best_epoch = epoch
            
            # ----- Per-epoch progress line (every epoch) ------------------
            _epoch_dur = _time.time() - _epoch_start
            _elapsed_total = _time.time() - _train_start
            _epochs_done = epoch + 1
            _epochs_left = self.epochs - _epochs_done
            _avg_epoch = _elapsed_total / _epochs_done
            _eta = _avg_epoch * _epochs_left

            if verbose:
                # Compact single-line progress every epoch
                _eta_str = f"{int(_eta//3600)}h{int((_eta%3600)//60):02d}m" if _eta >= 3600 else f"{int(_eta//60)}m{int(_eta%60):02d}s"
                _elapsed_str = f"{int(_elapsed_total//3600)}h{int((_elapsed_total%3600)//60):02d}m" if _elapsed_total >= 3600 else f"{int(_elapsed_total//60)}m{int(_elapsed_total%60):02d}s"
                gpu_str = ""
                if torch.cuda.is_available():
                    gpu_str = f" | GPU {torch.cuda.memory_allocated(0)/1e9:.2f}GB"
                logger.info(
                    f"Epoch {epoch+1:4d}/{self.epochs} | "
                    f"loss {avg_loss:.5f} (mse {avg_mse:.5f} unc {avg_uncertainty:.5f}) | "
                    f"best {best_loss:.5f} @{best_epoch+1} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e} | "
                    f"{_epoch_dur:.2f}s/ep | "
                    f"elapsed {_elapsed_str} ETA {_eta_str}"
                    f"{gpu_str}"
                )

            # Detailed block every 30 epochs
            if verbose and (epoch + 1) % 30 == 0:
                logger.info("-" * 80)
                logger.info(f"  Detailed stats at epoch {epoch+1}:")
                logger.info(f"    Total Loss:       {avg_loss:.6f}")
                logger.info(f"    MSE Loss:         {avg_mse:.6f}")
                logger.info(f"    Uncertainty Loss: {avg_uncertainty:.6f}")
                logger.info(f"    Learning Rate:    {scheduler.get_last_lr()[0]:.6f}")
                logger.info(f"    Best Loss:        {best_loss:.6f} (epoch {best_epoch+1})")
                logger.info(f"    Avg time/epoch:   {_avg_epoch:.2f}s")
                
                # GPU memory tracking
                if torch.cuda.is_available():
                    logger.info(f"    GPU Memory:       {torch.cuda.memory_allocated(0) / 1e9:.4f} GB allocated, "
                               f"{torch.cuda.memory_reserved(0) / 1e9:.4f} GB cached")
                
                # Loss trend
                if len(loss_history) > 30:
                    recent_trend = np.mean(loss_history[-10:]) - np.mean(loss_history[-30:-20])
                    trend_str = "↓ improving" if recent_trend < 0 else "↑ degrading" if recent_trend > 0 else "→ stable"
                    logger.info(f"    Loss Trend:       {trend_str} (Δ={recent_trend:.6f})")
                logger.info("-" * 80)
        
        if verbose:
            _total_train = _time.time() - _train_start
            _total_str = f"{int(_total_train//3600)}h{int((_total_train%3600)//60):02d}m{int(_total_train%60):02d}s"
            logger.info("="*80)
            logger.info("TRAINING COMPLETE")
            logger.info("="*80)
            logger.info(f"Total time:       {_total_str}")
            logger.info(f"Avg time/epoch:   {_total_train / self.epochs:.2f}s")
            logger.info(f"Final Loss:       {loss_history[-1]:.6f}")
            logger.info(f"Best Loss:        {best_loss:.6f} (epoch {best_epoch+1})")
            if loss_history[0] > 0:
                logger.info(f"Loss Reduction:   {((loss_history[0] - best_loss) / loss_history[0] * 100):.2f}%")
            
            # Final GPU stats
            if torch.cuda.is_available():
                logger.info(f"Peak GPU Memory:  {torch.cuda.max_memory_allocated(0) / 1e9:.4f} GB")
                torch.cuda.reset_peak_memory_stats()
            
            logger.info("="*80)
        
        return self
    
    def impute(self, X: Union[pd.DataFrame, np.ndarray], missing_mask: np.ndarray,
               return_uncertainty: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Impute missing values with uncertainty estimates.
        
        Args:
            X: Data with missing values (marked by missing_mask)
            missing_mask: Boolean mask indicating missing features [N, D]
            return_uncertainty: Whether to return uncertainty information
            
        Returns:
            X_imputed: Data with missing values imputed
            uncertainty_info: Dictionary with uncertainty estimates (if return_uncertainty=True)
        """
        self.model.eval()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Handle NaN values before scaling
        # First, replace NaN with 0 temporarily for scaling
        X_clean = X.copy()
        X_clean = np.nan_to_num(X_clean, nan=0.0)
        
        # Scale input data
        X_scaled = self.scaler.transform(X_clean)
        
        # Replace missing values with feature means initially
        X_input = X_scaled.copy()
        for i in range(X_input.shape[1]):
            X_input[missing_mask[:, i], i] = self.feature_means[i]
        
        # Create adaptive graph
        edge_index, edge_weights = self._create_adaptive_graph(X_input, missing_mask)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(self.device)
        missing_mask_tensor = torch.tensor(missing_mask, dtype=torch.bool).to(self.device)
        edge_index = edge_index.to(self.device)
        
        with torch.no_grad():
            predictions, uncertainties, global_conf = self.model(X_tensor, edge_index, missing_mask_tensor)
            
            # Check for NaN in predictions
            pred_np = predictions.cpu().numpy()
            if np.isnan(pred_np).any():
                n_nan = np.sum(np.isnan(pred_np))
                logger.error(f"Model produced {n_nan} NaN predictions! This indicates training issues.")
                logger.error("Replacing NaN predictions with feature means as fallback.")
                # Replace NaN predictions with feature means
                nan_mask = np.isnan(pred_np)
                for i in range(pred_np.shape[1]):
                    pred_np[nan_mask[:, i], i] = self.feature_means[i]
                predictions = torch.tensor(pred_np, device=predictions.device)
            
            # Replace missing values with predictions
            X_imputed = X_input.copy()
            X_imputed[missing_mask] = predictions[missing_mask_tensor].cpu().numpy()
            
            # Get uncertainty estimates
            uncertainty_values = uncertainties[missing_mask_tensor].cpu().numpy()
            confidence_scores = global_conf.squeeze().cpu().numpy()
            
            # Calculate confidence intervals (95%)
            std_devs = np.sqrt(uncertainty_values)
            lower_bounds = predictions[missing_mask_tensor].cpu().numpy() - 1.96 * std_devs
            upper_bounds = predictions[missing_mask_tensor].cpu().numpy() + 1.96 * std_devs
        
        # Inverse transform to original scale
        X_imputed = self.scaler.inverse_transform(X_imputed)
        
        # Round integer features and clip to observed ranges
        if self.is_integer is not None:
            for i in range(X_imputed.shape[1]):
                if self.is_integer[i]:
                    # Round to nearest integer
                    X_imputed[:, i] = np.round(X_imputed[:, i])
                
                # Clip all features to observed min/max ranges
                if self.feature_min is not None and self.feature_max is not None:
                    X_imputed[:, i] = np.clip(X_imputed[:, i], self.feature_min[i], self.feature_max[i])
        
        # Final check for NaN in output
        if np.isnan(X_imputed).any():
            logger.error(f"Final imputed data contains {np.sum(np.isnan(X_imputed))} NaN values!")
            logger.error("This should not happen after fallback handling. Using feature means for remaining NaN.")
            # Last resort: replace any remaining NaN with 0
            X_imputed = np.nan_to_num(X_imputed, nan=0.0)
        
        if return_uncertainty:
            uncertainty_info = {
                'uncertainty_values': uncertainty_values,
                'confidence_scores': confidence_scores,
                'mean_confidence': np.mean(confidence_scores),
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds
            }
            return X_imputed, uncertainty_info
        else:
            return X_imputed

    def impute_multiple(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        missing_mask: np.ndarray,
        n_imputations: int = 20,
        use_mc_dropout: bool = True,
        seed: Optional[int] = 42,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate multiple imputation samples for uncertainty quantification.

        Uses two complementary sources of uncertainty:

        1. **Parametric sampling** – For each missing cell the model predicts
           a mean μ and variance σ² (via the learned heteroscedastic variance
           heads).  Each imputation sample draws from N(μ, σ²), capturing
           *aleatoric* uncertainty (inherent data noise).

        2. **MC Dropout** (optional) – Running the forward pass with dropout
           active produces different μ/σ² per run, capturing *epistemic*
           uncertainty (model uncertainty).

        Downstream, M complete datasets are passed through risk-score
        calculators independently.  The spread of M score values per patient
        provides a non-parametric confidence interval for the score that
        properly accounts for imputation-induced uncertainty.

        This follows Rubin's multiple-imputation framework where:
        - Between-imputation variance B = var(scores across M imputations)
        - Total variance T ≈ (1 + 1/M) × B
        - 95% CI: score_mean ± t × √T

        Parameters
        ----------
        X : array-like of shape (N, D)
            Data matrix with missing values indicated by *missing_mask*.
        missing_mask : ndarray of shape (N, D), dtype bool
            ``True`` where values are missing.
        n_imputations : int
            Number of complete datasets to generate (M).  20–50 is typical
            for stable MI variance estimates.
        use_mc_dropout : bool
            If True, enable dropout during the forward pass for each sample
            (MC Dropout).  The model uses LayerNorm (no running-mean
            statistics), so ``model.train()`` only affects dropout layers.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        X_multiple : ndarray of shape (M, N, D)
            M complete imputed datasets in original (unscaled) space.
        uncertainty_info : dict
            Detailed per-cell and per-feature uncertainty statistics:

            * ``imputed_mean``  – (N, D) mean across M samples
            * ``imputed_std``   – (N, D) std across M samples
            * ``model_variance`` – (N, D) deterministic learned variance
            * ``confidence_scores`` – (N,) global confidence per patient
            * ``feature_uncertainty`` – per-feature summary dict
            * ``missing_mask`` – the original mask for downstream use
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if isinstance(X, pd.DataFrame):
            X = X.values

        N, D = X.shape

        # ----- Prepare input (mirrors impute()) -----
        X_clean = np.nan_to_num(X.copy(), nan=0.0)
        X_scaled = self.scaler.transform(X_clean)

        X_input = X_scaled.copy()
        for i in range(D):
            X_input[missing_mask[:, i], i] = self.feature_means[i]

        # Build graph once (depends only on the observed structure)
        edge_index, edge_weights = self._create_adaptive_graph(X_input, missing_mask)

        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(self.device)
        missing_mask_tensor = torch.tensor(missing_mask, dtype=torch.bool).to(self.device)
        edge_index = edge_index.to(self.device)

        # ----- Collect M imputation samples -----
        all_imputed = np.zeros((n_imputations, N, D))
        all_preds_scaled = []  # scaled-space mean predictions per run
        all_vars_scaled = []   # scaled-space variance per run

        logger.info(
            f"Generating {n_imputations} imputation samples "
            f"(MC dropout={use_mc_dropout}) ..."
        )

        for m in range(n_imputations):
            if use_mc_dropout:
                self.model.train()   # activates dropout (LayerNorm is fine)
            else:
                self.model.eval()

            with torch.no_grad():
                predictions, uncertainties, global_conf = self.model(
                    X_tensor, edge_index, missing_mask_tensor
                )

            pred_np = predictions.cpu().numpy()   # [N, D]  (mean, scaled)
            var_np = uncertainties.cpu().numpy()   # [N, D]  (variance, scaled)

            # Fallback for any NaN predictions
            nan_pred = np.isnan(pred_np)
            if nan_pred.any():
                for i in range(D):
                    pred_np[nan_pred[:, i], i] = self.feature_means[i]

            # Sample from N(μ, σ²) for each missing cell (parametric draw)
            std_np = np.sqrt(np.clip(var_np, 1e-8, None))
            noise = np.random.randn(N, D) * std_np
            sampled = pred_np + noise  # still in scaled space

            # Build complete dataset in scaled space
            X_imp_scaled = X_input.copy()
            X_imp_scaled[missing_mask] = sampled[missing_mask]

            # Inverse transform to original space
            X_imp = self.scaler.inverse_transform(X_imp_scaled)

            # Round integer features and clip to observed ranges
            if self.is_integer is not None:
                for i in range(D):
                    if self.is_integer[i]:
                        X_imp[:, i] = np.round(X_imp[:, i])
                    if self.feature_min is not None and self.feature_max is not None:
                        X_imp[:, i] = np.clip(
                            X_imp[:, i], self.feature_min[i], self.feature_max[i]
                        )

            X_imp = np.nan_to_num(X_imp, nan=0.0)
            all_imputed[m] = X_imp
            all_preds_scaled.append(pred_np)
            all_vars_scaled.append(var_np)

        # Reset to eval mode
        self.model.eval()

        # ----- Deterministic reference pass (for confidence scores) -----
        with torch.no_grad():
            det_pred, det_unc, det_conf = self.model(
                X_tensor, edge_index, missing_mask_tensor
            )
        det_var_np = det_unc.cpu().numpy()      # [N, D]
        det_conf_np = det_conf.squeeze().cpu().numpy()  # [N]

        # ----- Aggregate uncertainty statistics -----
        imp_mean = np.mean(all_imputed, axis=0)   # [N, D]
        imp_std = np.std(all_imputed, axis=0, ddof=1)  # [N, D]

        # Per-feature summary (only for missing cells)
        feature_uncertainty = {}
        for i in range(D):
            feat_mask = missing_mask[:, i]
            if not feat_mask.any():
                continue
            feat_name = (
                self.feature_names[i] if self.feature_names else f"feature_{i}"
            )
            feature_uncertainty[feat_name] = {
                "n_imputed": int(feat_mask.sum()),
                "mean_std": float(np.mean(imp_std[feat_mask, i])),
                "median_std": float(np.median(imp_std[feat_mask, i])),
                "mean_model_variance": float(np.mean(det_var_np[feat_mask, i])),
                "mean_confidence": float(np.mean(det_conf_np[feat_mask])),
            }

        logger.info(
            f"Multiple imputation complete: {n_imputations} samples, "
            f"mean cross-sample std = {np.mean(imp_std[missing_mask]):.4f} "
            f"(scaled), mean confidence = {np.mean(det_conf_np):.4f}"
        )

        uncertainty_info = {
            "n_imputations": n_imputations,
            "use_mc_dropout": use_mc_dropout,
            "imputed_mean": imp_mean,
            "imputed_std": imp_std,
            "model_variance": det_var_np,
            "confidence_scores": det_conf_np,
            "feature_uncertainty": feature_uncertainty,
            "missing_mask": missing_mask,
        }

        return all_imputed, uncertainty_info

    def visualize_tsne(self, X_real: np.ndarray, X_synthetic: np.ndarray,
                       y_real: Optional[np.ndarray] = None, y_synthetic: Optional[np.ndarray] = None,
                       save_path: Optional[str] = None) -> None:
        """
        Visualize real vs synthetic data using t-SNE to verify distribution matching.
        
        Args:
            X_real: Real data
            X_synthetic: Synthetic/imputed data
            y_real: Real labels (optional)
            y_synthetic: Synthetic labels (optional)
            save_path: Path to save the plot (optional)
        """
        # Combine data
        X_combined = np.vstack([X_real, X_synthetic])
        types = ['Real'] * len(X_real) + ['Synthetic'] * len(X_synthetic)
        
        if y_real is not None and y_synthetic is not None:
            y_combined = np.concatenate([y_real, y_synthetic])
        else:
            y_combined = None
        
        # Check if we have enough samples for t-SNE
        total_samples = len(X_combined)
        min_samples_for_tsne = 6  # Need at least 6 samples for perplexity of 5
        
        if total_samples < min_samples_for_tsne:
            logger.warning(
                f"Skipping t-SNE visualization: only {total_samples} samples available, "
                f"need at least {min_samples_for_tsne} for meaningful t-SNE. "
                f"Increase test set size or disable t-SNE for quick tests."
            )
            return
        
        # Auto-adjust perplexity based on sample size
        default_perplexity = 30
        # Perplexity should be less than n_samples, use at most n_samples - 1
        adjusted_perplexity = min(default_perplexity, max(5, (total_samples - 1) // 2))
        
        if adjusted_perplexity < default_perplexity:
            logger.warning(
                f"Adjusted t-SNE perplexity from {default_perplexity} to {adjusted_perplexity} "
                f"for {total_samples} samples (perplexity must be less than n_samples)"
            )
        
        if total_samples < 30:
            logger.warning(f"Small sample size ({total_samples}) - t-SNE visualization may not be very meaningful")
        
        # Apply t-SNE with adjusted perplexity
        logger.info(f"Generating t-SNE visualization with perplexity={adjusted_perplexity} for {total_samples} samples")
        tsne = TSNE(n_components=2, random_state=42, perplexity=adjusted_perplexity)
        X_tsne = tsne.fit_transform(X_combined)

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning(f"Skipping t-SNE plot because matplotlib is unavailable: {exc}")
            return
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        if y_combined is not None:
            for label in np.unique(y_combined):
                mask_real = (y_combined == label) & (np.array(types) == 'Real')
                mask_synth = (y_combined == label) & (np.array(types) == 'Synthetic')
                
                plt.scatter(X_tsne[mask_real, 0], X_tsne[mask_real, 1],
                           label=f'Real - Class {label}', alpha=0.6, s=50)
                
                if np.any(mask_synth):
                    plt.scatter(X_tsne[mask_synth, 0], X_tsne[mask_synth, 1],
                               marker='x', label=f'Synthetic - Class {label}', alpha=0.6, s=50)
        else:
            mask_real = np.array(types) == 'Real'
            mask_synth = np.array(types) == 'Synthetic'
            
            plt.scatter(X_tsne[mask_real, 0], X_tsne[mask_real, 1],
                       label='Real', alpha=0.6, s=50)
            plt.scatter(X_tsne[mask_synth, 0], X_tsne[mask_synth, 1],
                       marker='x', label='Synthetic', alpha=0.6, s=50)
        
        plt.title('t-SNE: Real vs Synthetic Data Distribution')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"t-SNE plot saved to {save_path}")
        
        plt.show()
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        model_state = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'feature_types': self.feature_types,
            'is_integer': self.is_integer,
            'feature_min': self.feature_min,
            'feature_max': self.feature_max,
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'k_neighbors': self.k_neighbors,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> 'GraphImputationModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            device: Device to load the model on (if None, will auto-detect)
            
        Returns:
            Loaded model instance
        """
        # Determine target device
        if device is None:
            target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            target_device = torch.device(device)
        
        # Create a custom unpickler that maps tensors to the target device
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location=target_device)
                else:
                    return super().find_class(module, name)
        
        # Load model state with device mapping
        with open(filepath, 'rb') as f:
            if target_device.type == 'cpu':
                # Use custom unpickler for CPU loading
                model_state = CPU_Unpickler(f).load()
            else:
                # Standard loading for GPU
                model_state = pickle.load(f)
        
        config = model_state['config']
        model = cls(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            k_neighbors=config['k_neighbors'],
            learning_rate=config['learning_rate'],
            epochs=config['epochs'],
            feature_types=model_state['feature_types'],
            device=str(target_device)
        )
        
        model.scaler = model_state['scaler']
        model.feature_names = model_state['feature_names']
        model.feature_means = model_state['feature_means']
        model.feature_stds = model_state['feature_stds']
        model.is_integer = model_state.get('is_integer', None)
        model.feature_min = model_state.get('feature_min', None)
        model.feature_max = model_state.get('feature_max', None)
        
        # Initialize and load model weights
        input_dim = len(model.feature_names)
        model.model = EnhancedGNNImputer(
            input_dim, model.hidden_dim, model.num_layers,
            model.num_heads, model.feature_types
        ).to(model.device)
        
        if model_state['model_state_dict']:
            # Load state dict with proper device mapping
            state_dict = model_state['model_state_dict']
            # Ensure all tensors in state_dict are on the correct device
            model.model.load_state_dict(state_dict, strict=True)
        
        logger.info(f"Model loaded from {filepath} to device: {model.device}")
        
        return model
    
    def _detect_integer_features(self, X: np.ndarray, verbose: bool = True) -> None:
        """
        Detect which features should be treated as integers/discrete.
        
        A feature is considered integer if:
        1. All non-NaN values are integers (or very close to integers)
        2. Has a relatively small number of unique values (indicating discrete)
        
        Args:
            X: Original data (before scaling)
            verbose: Whether to log detection results
        """
        n_features = X.shape[1]
        self.is_integer = np.zeros(n_features, dtype=bool)
        self.feature_min = np.zeros(n_features)
        self.feature_max = np.zeros(n_features)
        
        integer_features = []
        
        for i in range(n_features):
            col_data = X[:, i]
            # Remove NaN values
            col_data_clean = col_data[~np.isnan(col_data)]
            
            if len(col_data_clean) == 0:
                continue
            
            # Store min/max for clipping
            self.feature_min[i] = np.min(col_data_clean)
            self.feature_max[i] = np.max(col_data_clean)
            
            # Check if values are integers (within small tolerance for floating point errors)
            is_int_valued = np.allclose(col_data_clean, np.round(col_data_clean), atol=1e-6)
            
            # Count unique values (normalized by total values)
            n_unique = len(np.unique(col_data_clean))
            n_total = len(col_data_clean)
            uniqueness_ratio = n_unique / n_total
            
            # Consider as integer if:
            # 1. All values are integers AND has ≤100 unique values, OR
            # 2. Has low uniqueness ratio (< 0.1) indicating discrete/categorical
            # This prevents treating continuous variables (like blood pressure) as integers
            # just because they happen to be integer-valued
            if (is_int_valued and n_unique <= 100) or uniqueness_ratio < 0.1:
                self.is_integer[i] = True
                feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
                integer_features.append((feature_name, n_unique, uniqueness_ratio))
        
        if verbose and len(integer_features) > 0:
            logger.info(f"\nDetected {len(integer_features)} integer/discrete features:")
            for fname, n_uniq, ratio in integer_features[:10]:  # Show first 10
                logger.info(f"  - {fname}: {n_uniq} unique values (ratio: {ratio:.4f})")
            if len(integer_features) > 10:
                logger.info(f"  ... and {len(integer_features) - 10} more")
