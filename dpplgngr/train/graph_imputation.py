"""
Luigi Pipeline Task for Graph Imputation Model Training
========================================================

This module provides a Luigi pipeline step for training a graph-based imputation
model that can be used to impute missing values in datasets while preserving
the original data distribution (verified via t-SNE).

Author: SB
Date: 2025-10-29
"""

import pandas as pd
import numpy as np
import json
import logging
import os
import luigi
from typing import Optional

from dpplgngr.models.graph_imputer import GraphImputationModel
from dpplgngr.etl.prep_dataset_tabular import TuplesProcess

logger = logging.getLogger('luigi-interface')

# Metadata
__author__ = 'SB'
__date__ = '2025-10-29'


class TrainGraphImputer(luigi.Task):
    """
    Luigi task to train a graph-based imputation model.
    
    This task trains a GNN-based imputation model that can later be used to impute
    missing values in other datasets while preserving the statistical properties
    and t-SNE distributions of the original data.
    
    Config Parameters (in graph_imputer_config.json):
    - input_file: Path to the training data (parquet format)
    - working_dir: Directory to save the trained model and outputs
    - columns: List of columns to use for training (optional, uses all if not specified)
    - allow_missing_training: If True, train on data with pre-existing missing values (default: False)
                              When enabled, the model learns from real missing patterns
                              rather than only artificially created scenarios
    - model_params: Dictionary of model hyperparameters
        - hidden_dim: Size of hidden layers (default: 128)
        - num_layers: Number of GNN layers (default: 4)
        - num_heads: Number of attention heads (default: 8)
        - k_neighbors: Number of neighbors for graph construction (default: 20)
        - learning_rate: Learning rate (default: 0.001)
        - epochs: Number of training epochs (default: 300)
    - feature_types: List of feature types ('continuous' or 'categorical')
    - generate_tsne_plot: Whether to generate t-SNE visualization (default: True)
    - test_imputation: Whether to test imputation on held-out data (default: True)
    - test_missing_rate: Missing rate for testing (default: 0.3)
    """
    
    imputer_config = luigi.Parameter(default="config/graph_imputer.json")
    etl_config = luigi.Parameter(default="config/etl.json")
    override_etl = luigi.BoolParameter(default=False)

    def output(self):
        """Define output target for the trained model."""
        with open(self.imputer_config, 'r') as f:
            config = json.load(f)
        
        working_dir = config.get('working_dir', 'models/graph_imputer')
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        
        model_path = os.path.join(working_dir, 'graph_imputer_model.pkl')
        return luigi.LocalTarget(model_path)
    
    def requires(self):
        """Define dependencies - requires ETL preprocessing unless overridden."""
        if self.override_etl:
            return []
        else:
            return TuplesProcess(etl_config=self.etl_config)

    def run(self):
        """Train the graph imputation model."""
        # Load configuration
        with open(self.imputer_config, 'r') as f:
            config = json.load(f)
        
        logger.info("="*80)
        logger.info("TRAINING GRAPH IMPUTATION MODEL")
        logger.info("="*80)
        
        # Extract configuration parameters
        input_file = config.get('input_file')
        working_dir = config.get('working_dir', 'models/graph_imputer')
        columns = config.get('columns', None)
        model_params = config.get('model_params', {})
        feature_types = config.get('feature_types', None)
        generate_tsne = config.get('generate_tsne_plot', True)
        test_imputation = config.get('test_imputation', True)
        test_missing_rate = config.get('test_missing_rate', 0.3)
        allow_missing_training = config.get('allow_missing_training', False)
        
        # Create working directory if it doesn't exist
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        
        # Load data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_parquet(input_file)
        
        # Handle Decimal columns (convert to float)
        from decimal import Decimal
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it's Decimal
                if df[col].apply(lambda x: isinstance(x, Decimal) if pd.notna(x) else False).any():
                    df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle datetime/timestamp columns (convert to Unix timestamp)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                logger.info(f"Converting datetime column '{col}' to Unix timestamp")
                df[col] = df[col].astype('int64') / 10**9  # Convert to seconds since epoch
        
        # Handle timedelta columns
        for col in df.columns:
            if df[col].dtype.kind == 'm':  # 'm' indicates timedelta
                logger.info(f"Converting timedelta column '{col}' to days")
                df[col] = df[col].dt.days  # Convert to days as float
        
        # Drop non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            logger.warning(f"Dropping non-numeric columns: {non_numeric_cols}")
            df = df.select_dtypes(include=[np.number])
        
        # Select columns if specified
        if columns is not None:
            logger.info(f"Using specified columns: {len(columns)} features")
            available_cols = [col for col in columns if col in df.columns]
            if len(available_cols) < len(columns):
                missing_cols = set(columns) - set(available_cols)
                logger.warning(f"Columns not found in data: {missing_cols}")
            df = df[available_cols]
        else:
            logger.info(f"Using all numeric columns: {len(df.columns)} features")
        
        # Check for missing values in data
        n_missing_before = df.isnull().sum().sum()
        if n_missing_before > 0:
            if allow_missing_training:
                logger.info(f"Data contains {n_missing_before} missing values")
                logger.info("allow_missing_training=True: Will use partial data for training")
            else:
                logger.info(f"Data contains {n_missing_before} missing values")
                logger.info("Removing rows with missing values (set allow_missing_training=True to train on partial data)")
                # Remove rows with any missing values (need complete data for traditional training)
                n_before = len(df)
                df = df.dropna()
                n_after = len(df)
                logger.info(f"Removed {n_before - n_after} rows with missing values")
                logger.info(f"Training on {n_after} complete samples")
        else:
            logger.info("No missing values in training data")
        
        # Split into train and test sets if testing is enabled
        if test_imputation:
            from sklearn.model_selection import train_test_split
            X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
            logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        else:
            X_train = df
            X_test = None
        
        # Initialize model with parameters from config
        logger.info("="*80)
        logger.info("INITIALIZING GRAPH IMPUTATION MODEL")
        logger.info("="*80)
        logger.info(f"Model Configuration:")
        logger.info(f"  Hidden dimension:     {model_params.get('hidden_dim', 128)}")
        logger.info(f"  Number of layers:     {model_params.get('num_layers', 4)}")
        logger.info(f"  Attention heads:      {model_params.get('num_heads', 8)}")
        logger.info(f"  K-neighbors:          {model_params.get('k_neighbors', 20)}")
        logger.info(f"  Learning rate:        {model_params.get('learning_rate', 0.001)}")
        logger.info(f"  Training epochs:      {model_params.get('epochs', 300)}")
        logger.info(f"  Allow missing train:  {allow_missing_training}")
        
        # Check CUDA availability
        import torch
        if torch.cuda.is_available():
            logger.info(f"PyTorch GPU:")
            logger.info(f"  CUDA available:       YES")
            logger.info(f"  CUDA version:         {torch.version.cuda}")
            logger.info(f"  Device count:         {torch.cuda.device_count()}")
        else:
            logger.info(f"PyTorch GPU:")
            logger.info(f"  CUDA available:       NO")
            logger.info(f"  Training on:          CPU")
            logger.info(f"  Note: Training will be slower without GPU")
        
        logger.info("="*80)
        
        model = GraphImputationModel(
            hidden_dim=model_params.get('hidden_dim', 128),
            num_layers=model_params.get('num_layers', 4),
            num_heads=model_params.get('num_heads', 8),
            k_neighbors=model_params.get('k_neighbors', 20),
            learning_rate=model_params.get('learning_rate', 0.001),
            epochs=model_params.get('epochs', 300),
            feature_types=feature_types,
            allow_missing_training=allow_missing_training
        )
        
        # Train the model
        logger.info("")
        logger.info("Starting model training...")
        logger.info("")
        model.fit(X_train, verbose=True)
        
        # Save the trained model
        model_path = self.output().path
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Test imputation if enabled
        if test_imputation and X_test is not None:
            logger.info("="*80)
            logger.info("TESTING IMPUTATION PERFORMANCE")
            logger.info("="*80)
            
            # Create artificial missing data
            X_test_values = X_test.values
            missing_mask = np.zeros_like(X_test_values, dtype=bool)
            
            for i in range(len(X_test_values)):
                n_missing = int(test_missing_rate * X_test_values.shape[1])
                if n_missing > 0:
                    missing_features = np.random.choice(
                        X_test_values.shape[1], n_missing, replace=False
                    )
                    missing_mask[i, missing_features] = True
            
            logger.info(f"Test Configuration:")
            logger.info(f"  Test samples:         {len(X_test)}")
            logger.info(f"  Features:             {X_test_values.shape[1]}")
            logger.info(f"  Missing rate:         {test_missing_rate:.1%}")
            logger.info(f"  Total missing values: {np.sum(missing_mask)}")
            logger.info("")
            
            # Apply missing mask
            X_test_missing = X_test_values.copy()
            X_test_missing[missing_mask] = np.nan
            
            # Impute
            logger.info("Running imputation on test set...")
            import time
            start_time = time.time()
            X_imputed, uncertainty_info = model.impute(X_test_missing, missing_mask, return_uncertainty=True)
            impute_time = time.time() - start_time
            
            logger.info(f"Imputation completed in {impute_time:.2f} seconds")
            logger.info(f"Average time per sample: {impute_time / len(X_test) * 1000:.2f} ms")
            logger.info("")
            
            # Calculate imputation metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            true_missing = X_test_values[missing_mask]
            imputed_missing = X_imputed[missing_mask]
            
            # Calculate per-feature metrics
            feature_metrics = []
            for feat_idx in range(X_test_values.shape[1]):
                feat_mask = missing_mask[:, feat_idx]
                if np.sum(feat_mask) > 0:
                    feat_true = X_test_values[feat_mask, feat_idx]
                    feat_imputed = X_imputed[feat_mask, feat_idx]
                    feat_rmse = np.sqrt(mean_squared_error(feat_true, feat_imputed))
                    feat_mae = mean_absolute_error(feat_true, feat_imputed)
                    try:
                        feat_r2 = r2_score(feat_true, feat_imputed)
                    except:
                        feat_r2 = 0.0
                    feature_metrics.append({
                        'feature_idx': feat_idx,
                        'feature_name': X_train.columns[feat_idx] if hasattr(X_train, 'columns') else f'feature_{feat_idx}',
                        'rmse': float(feat_rmse),
                        'mae': float(feat_mae),
                        'r2': float(feat_r2),
                        'n_missing': int(np.sum(feat_mask))
                    })
            
            # Overall metrics
            rmse = np.sqrt(mean_squared_error(true_missing, imputed_missing))
            mae = mean_absolute_error(true_missing, imputed_missing)
            r2 = r2_score(true_missing, imputed_missing)
            
            # Relative error
            mape = np.mean(np.abs((true_missing - imputed_missing) / (np.abs(true_missing) + 1e-8))) * 100
            
            logger.info("="*80)
            logger.info("IMPUTATION PERFORMANCE METRICS")
            logger.info("="*80)
            logger.info(f"Overall Performance:")
            logger.info(f"  Root Mean Squared Error (RMSE):  {rmse:.6f}")
            logger.info(f"  Mean Absolute Error (MAE):       {mae:.6f}")
            logger.info(f"  R² Score:                        {r2:.6f}")
            logger.info(f"  Mean Absolute % Error (MAPE):    {mape:.2f}%")
            logger.info(f"  Mean Confidence:                 {uncertainty_info['mean_confidence']:.6f}")
            logger.info("")
            
            # Show top 5 best and worst features
            feature_metrics_sorted = sorted(feature_metrics, key=lambda x: x['rmse'])
            logger.info("Top 5 Best Imputed Features (lowest RMSE):")
            for i, fm in enumerate(feature_metrics_sorted[:5], 1):
                logger.info(f"  {i}. {fm['feature_name'][:40]:40s} RMSE={fm['rmse']:.6f}, R²={fm['r2']:.4f}, n={fm['n_missing']}")
            logger.info("")
            
            logger.info("Top 5 Worst Imputed Features (highest RMSE):")
            for i, fm in enumerate(feature_metrics_sorted[-5:][::-1], 1):
                logger.info(f"  {i}. {fm['feature_name'][:40]:40s} RMSE={fm['rmse']:.6f}, R²={fm['r2']:.4f}, n={fm['n_missing']}")
            logger.info("="*80)
            
            # Save test metrics
            metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'mean_confidence': float(uncertainty_info['mean_confidence']),
                'test_missing_rate': test_missing_rate,
                'n_test_samples': len(X_test),
                'imputation_time_seconds': float(impute_time),
                'time_per_sample_ms': float(impute_time / len(X_test) * 1000),
                'feature_metrics': feature_metrics
            }
            
            metrics_path = os.path.join(working_dir, 'test_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Test metrics saved to {metrics_path}")
            
            # Generate t-SNE plot if enabled
            if generate_tsne:
                logger.info("Generating t-SNE visualization...")
                tsne_path = os.path.join(working_dir, 'tsne_imputation_test.png')
                
                try:
                    model.visualize_tsne(
                        X_real=X_test_values,
                        X_synthetic=X_imputed,
                        save_path=tsne_path
                    )
                    logger.info(f"t-SNE plot saved to {tsne_path}")
                except Exception as e:
                    logger.warning(f"Could not generate t-SNE plot: {e}")
        
        # Save model configuration and metadata
        metadata = {
            'input_file': input_file,
            'n_samples': len(X_train),
            'n_features': len(X_train.columns),
            'feature_names': X_train.columns.tolist(),
            'feature_types': feature_types,
            'model_params': model_params,
            'trained_on': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = os.path.join(working_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Model metadata saved to {metadata_path}")
        
        logger.info("="*80)
        logger.info("GRAPH IMPUTATION MODEL TRAINING COMPLETE")
        logger.info("="*80)


class ApplyGraphImputer(luigi.Task):
    """
    Luigi task to apply a trained graph imputation model to a dataset with missing values.
    
    This task loads a trained graph imputation model and applies it to a new dataset
    with arbitrary missing columns.
    
    Config Parameters (in graph_imputer_apply_config.json):
    - model_path: Path to the trained model (.pkl file)
    - input_file: Path to the data with missing values (parquet format)
    - output_file: Path to save the imputed data (parquet format)
    - columns: List of columns to impute (optional, uses all available if not specified)
    - save_uncertainty: Whether to save uncertainty estimates (default: True)
    - generate_tsne_plot: Whether to generate t-SNE visualization (default: False)
    - score_config: Path to score configuration file (optional, ensures required columns exist)
    """
    
    apply_config = luigi.Parameter(default="config/graph_imputer_apply.json")
    etl_config = luigi.Parameter(default="")
    score_config = luigi.Parameter(default="")  # Optional: path to score config file

    def output(self):
        """Define output target for the imputed data."""
        with open(self.apply_config, 'r') as f:
            config = json.load(f)
        
        output_file = config.get('output_file')
        return luigi.LocalTarget(output_file)
    
    def requires(self):
        """Require TuplesProcess to complete before applying graph imputation."""
        if self.etl_config:
            from dpplgngr.etl.prep_dataset_tabular import TuplesProcess
            return TuplesProcess(etl_config=self.etl_config)
        return []

    def run(self):
        """Apply the graph imputation model to new data."""
        # Load configuration
        with open(self.apply_config, 'r') as f:
            config = json.load(f)
        
        logger.info("="*80)
        logger.info("APPLYING GRAPH IMPUTATION MODEL")
        logger.info("="*80)
        
        # Extract configuration parameters
        model_path = config.get('model_path')
        input_file = config.get('input_file')
        output_file = config.get('output_file')
        columns = config.get('columns', None)
        save_uncertainty = config.get('save_uncertainty', True)
        generate_tsne = config.get('generate_tsne_plot', False)
        
        # Check for score_config from parameter or config file
        score_config_path = self.score_config or config.get('score_config', '')
        
        # Load the trained model
        logger.info(f"Loading trained model from {model_path}")
        model = GraphImputationModel.load(model_path)
        logger.info(f"Model loaded with {len(model.feature_names)} features")
        
        # Load data with missing values
        logger.info(f"Loading data from {input_file}")
        df = pd.read_parquet(input_file)
        
        # Track score-required columns to preserve after imputation
        score_required_columns = set()
        
        # If score config is provided, ensure required columns exist
        # Convert to string and check if non-empty (Luigi parameters need explicit string conversion)
        score_config_str = str(score_config_path).strip() if score_config_path else ''
        logger.info(f"Score config string value: '{score_config_str}'")
        
        if score_config_str and score_config_str != '' and not score_config_str.lower() == 'none':
            logger.info(f"Loading score configuration from {score_config_str}")
            from dpplgngr.utils.score_config_utils import (
                get_required_columns_from_config_file, 
                ensure_columns_exist
            )
            
            try:
                required_columns = get_required_columns_from_config_file(score_config_str)
                logger.info(f"Score requires {len(required_columns)} columns")
                score_required_columns = required_columns  # Track these for later
                
                # Add missing columns with NaN values
                df, added_columns = ensure_columns_exist(df, required_columns)
                
                if added_columns:
                    logger.info(f"Added {len(added_columns)} missing columns for score calculation")
            except Exception as e:
                logger.warning(f"Could not process score config: {e}")
                logger.warning("Continuing without adding score-required columns")
        else:
            logger.info("Score config not provided or empty, skipping column addition")
        
        # Handle Decimal columns
        from decimal import Decimal
        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].apply(lambda x: isinstance(x, Decimal) if pd.notna(x) else False).any():
                    df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle datetime/timestamp columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                logger.info(f"Converting datetime column '{col}' to Unix timestamp")
                df[col] = df[col].astype('int64') / 10**9
        
        # Handle timedelta columns
        for col in df.columns:
            if df[col].dtype.kind == 'm':
                logger.info(f"Converting timedelta column '{col}' to days")
                df[col] = df[col].dt.days
        
        # Drop non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            logger.warning(f"Dropping non-numeric columns: {non_numeric_cols}")
            df = df.select_dtypes(include=[np.number])
        
        # Select columns if specified
        if columns is not None:
            available_cols = [col for col in columns if col in df.columns]
            if len(available_cols) < len(columns):
                missing_cols = set(columns) - set(available_cols)
                logger.warning(f"Columns not found in data: {missing_cols}")
            df = df[available_cols]
        
        # Ensure all model features are present
        missing_features = set(model.feature_names) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features in data: {missing_features}")
            logger.info("Adding missing features with NaN values")
            for feat in missing_features:
                df[feat] = np.nan
        
        # Separate columns into model features and score-only features
        model_feature_set = set(model.feature_names)
        all_columns = set(df.columns)
        score_only_columns = score_required_columns - model_feature_set
        
        if score_only_columns:
            logger.info(f"Found {len(score_only_columns)} score-required columns not in model:")
            for col in sorted(score_only_columns):
                logger.info(f"  - {col}")
            logger.info("These will be imputed separately using median imputation")
            
            # Save score-only columns for later
            df_score_only = df[list(score_only_columns)].copy()
        
        # Extract only model features for graph imputation
        df_model = df[model.feature_names].copy()
        
        # Identify missing values in model features
        missing_mask = df_model.isna().values
        n_missing = np.sum(missing_mask)
        missing_rate = np.mean(missing_mask)
        
        logger.info(f"Data statistics for model features:")
        logger.info(f"  Samples: {len(df_model)}")
        logger.info(f"  Features: {len(df_model.columns)}")
        logger.info(f"  Missing values: {n_missing} ({missing_rate:.2%})")
        
        if n_missing == 0:
            logger.info("No missing values found in model features")
            df_imputed = df_model
        else:
            # Impute missing values using graph model
            logger.info("Imputing model features...")
            X_values = df_model.values
            X_imputed, uncertainty_info = model.impute(X_values, missing_mask, return_uncertainty=True)
            
            logger.info(f"Imputation complete - Mean confidence: {uncertainty_info['mean_confidence']:.4f}")
            
            # Create output DataFrame
            df_imputed = pd.DataFrame(X_imputed, columns=df_model.columns, index=df_model.index)
            
            # Save uncertainty information if requested
            if save_uncertainty:
                uncertainty_file = output_file.replace('.parquet', '_uncertainty.npz')
                output_dir_unc = os.path.dirname(uncertainty_file)
                if output_dir_unc and not os.path.exists(output_dir_unc):
                    os.makedirs(output_dir_unc, exist_ok=True)
                np.savez(
                    uncertainty_file,
                    missing_mask=missing_mask,
                    uncertainty_values=uncertainty_info['uncertainty_values'],
                    confidence_scores=uncertainty_info['confidence_scores'],
                    lower_bounds=uncertainty_info['lower_bounds'],
                    upper_bounds=uncertainty_info['upper_bounds']
                )
                logger.info(f"Uncertainty estimates saved to {uncertainty_file}")
        
        # Impute score-only columns using simple imputation
        if score_only_columns:
            logger.info(f"Imputing {len(score_only_columns)} score-only columns...")
            try:
                from sklearn.impute import SimpleImputer
                
                # Check which columns are entirely NaN
                entirely_nan_cols = [col for col in score_only_columns if df_score_only[col].isna().all()]
                partially_nan_cols = [col for col in score_only_columns if col not in entirely_nan_cols]
                
                if entirely_nan_cols:
                    logger.warning(f"Found {len(entirely_nan_cols)} columns with all NaN values:")
                    for col in entirely_nan_cols:
                        logger.warning(f"  - {col} (will use constant=0)")
                    # For entirely NaN columns, use constant imputation with 0
                    for col in entirely_nan_cols:
                        df_score_only[col] = 0.0
                
                if partially_nan_cols:
                    logger.info(f"Imputing {len(partially_nan_cols)} partially-missing columns with median...")
                    # For partially NaN columns, use median imputation
                    imputer = SimpleImputer(strategy='median')
                    df_score_only[partially_nan_cols] = imputer.fit_transform(df_score_only[partially_nan_cols])
                
                df_score_only_imputed = df_score_only
                logger.info("Score-only columns imputed successfully")
                
                # Combine model features and score-only features
                df_imputed = pd.concat([df_imputed, df_score_only_imputed], axis=1)
                logger.info(f"Combined DataFrame shape: {df_imputed.shape}")
            except Exception as e:
                logger.error(f"Failed to impute score-only columns: {e}")
                logger.warning("Score-only columns will contain NaN values")
                df_imputed = pd.concat([df_imputed, df_score_only], axis=1)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        
        # Save imputed data
        df_imputed.to_parquet(output_file)
        logger.info(f"Imputed data saved to {output_file}")
        
        # Generate t-SNE plot if requested (only works if we have uncertainty_info)
        if generate_tsne and n_missing > 0:
            logger.info("Generating t-SNE visualization...")
            tsne_path = output_file.replace('.parquet', '_tsne.png')
            
            try:
                # Use only complete cases from original data for comparison
                complete_mask = ~np.any(missing_mask, axis=1)
                if np.sum(complete_mask) > 10:
                    model.visualize_tsne(
                        X_real=X_values[complete_mask],
                        X_synthetic=X_imputed[~complete_mask] if np.any(~complete_mask) else X_imputed[:100],
                        save_path=tsne_path
                    )
                    logger.info(f"t-SNE plot saved to {tsne_path}")
                else:
                    logger.warning("Not enough complete cases for t-SNE visualization")
            except Exception as e:
                logger.warning(f"Could not generate t-SNE plot: {e}")
            except Exception as e:
                logger.warning(f"Could not generate t-SNE plot: {e}")
        
        logger.info("="*80)
        logger.info("GRAPH IMPUTATION APPLICATION COMPLETE")
        logger.info("="*80)
