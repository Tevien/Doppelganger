"""
Alternative Imputation Pipeline Step

This module provides a Luigi pipeline task for creating alternative imputations
using different strategies (mean, median, KNN, iterative, etc.) to compare
their impact on downstream score calculations and predictions.

Author: SB
Date: 2025-10-31
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from joblib import dump, load

# Try to import luigi
try:
    import luigi
    _using_luigi_replacement = False
except ImportError:
    from dpplgngr.utils.luigi_replacement import Task, Parameter, LocalTarget, build as luigi_build
    class MockLuigi:
        Task = Task
        Parameter = Parameter
        LocalTarget = LocalTarget
        @staticmethod
        def build(*args, **kwargs):
            return luigi_build(*args, **kwargs)
    luigi = MockLuigi()
    _using_luigi_replacement = True

# Try to import sklearn imputers
try:
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    _sklearn_available = True
except ImportError:
    _sklearn_available = False
    logging.warning("sklearn not available, imputation will be limited")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


AVAILABLE_IMPUTERS = {
    'mean': 'Mean imputation (simple, fast)',
    'median': 'Median imputation (robust to outliers)',
    'most_frequent': 'Mode imputation (for categorical)',
    'knn': 'K-Nearest Neighbors imputation (uses similar patients)',
    'iterative': 'Iterative imputation (multivariate)',
    'mice': 'MICE - Multiple Imputation by Chained Equations (same as iterative)',
}


class AlternativeImputation(luigi.Task):
    """
    Luigi task to create alternative imputations using different strategies.
    
    This task:
    1. Reads preprocessed data with missing values
    2. Applies alternative imputation strategies
    3. Saves multiple versions of the imputed data
    4. Saves imputer models for reproducibility
    
    Parameters
    ----------
    input_file : str
        Path to the preprocessed data file (before imputation)
    output_dir : str
        Directory to save imputed datasets
    imputation_strategies : str
        Comma-separated list of imputation strategies to use
        Options: mean, median, most_frequent, knn, mice (iterative)
        Default: 'mean,median,knn,mice'
    prefix : str
        Prefix for output files
    etl_config : str
        Path to ETL config (optional, for dependency tracking)
    score_config : str
        Path to score configuration file (optional, ensures required columns exist)
    """
    
    input_file = luigi.Parameter()
    output_dir = luigi.Parameter()
    imputation_strategies = luigi.Parameter(default='mean,median,knn,mice')
    prefix = luigi.Parameter(default='imputed')
    etl_config = luigi.Parameter(default='')
    score_config = luigi.Parameter(default='')  # Optional: path to score config file
    alt_impute_config = luigi.Parameter(default='')  # Path to alternative_imputation.json config file
    
    def output(self):
        """Define output targets for each imputation strategy."""
        strategies = [s.strip() for s in self.imputation_strategies.split(',')]
        return {
            strategy: luigi.LocalTarget(
                os.path.join(self.output_dir, f'{self.prefix}_{strategy}.parquet')
            )
            for strategy in strategies
        }
    
    def requires(self):
        """Require TuplesProcess to complete before alternative imputation."""
        if self.etl_config:
            from dpplgngr.etl.prep_dataset_tabular import TuplesProcess
            return TuplesProcess(etl_config=self.etl_config)
        return []
    
    def run(self):
        """Execute the alternative imputation pipeline."""
        # Write debug info to a file that Luigi can't hide
        with open('/tmp/alternative_imputation_debug.log', 'w') as f:
            f.write(f"AlternativeImputation.run() called at {pd.Timestamp.now()}\n")
            f.write(f"score_config parameter: '{self.score_config}'\n")
            f.write(f"score_config type: {type(self.score_config)}\n")
            f.write(f"alt_impute_config parameter: '{self.alt_impute_config}'\n")
            f.write(f"input_file: '{self.input_file}'\n")
            f.write(f"output_dir: '{self.output_dir}'\n")
        
        # WORKAROUND: Luigi is not passing parameters correctly, so infer the config file path
        # from the output directory name or input_file path
        effective_score_config = self.score_config
        effective_alt_impute_config = self.alt_impute_config
        
        # Try multiple ways to get the score_config
        if not effective_score_config or str(effective_score_config).strip() == '':
            # Method 1: Try reading from alt_impute_config if it was passed
            if self.alt_impute_config and str(self.alt_impute_config).strip():
                try:
                    with open(str(self.alt_impute_config).strip(), 'r') as f:
                        alt_cfg = json.load(f)
                        config_score_config = alt_cfg.get('score_config', '')
                        if config_score_config:
                            effective_score_config = config_score_config
                            logger.info(f"Got score_config from alt_impute_config file: {effective_score_config}")
                except Exception as e:
                    logger.warning(f"Could not read alt_impute_config: {e}")
            
            # Method 2: Try to infer config paths from etl_config location
            if not effective_alt_impute_config or str(effective_alt_impute_config).strip() == '':
                if self.etl_config and str(self.etl_config).strip():
                    # Try to find alternative_imputation.json in the same directory as etl_config
                    etl_config_dir = os.path.dirname(str(self.etl_config))
                    inferred_alt_config = os.path.join(etl_config_dir, 'alternative_imputation.json')
                    if os.path.exists(inferred_alt_config):
                        effective_alt_impute_config = inferred_alt_config
                        logger.info(f"Inferred alt_impute_config from etl_config directory: {effective_alt_impute_config}")
                        # Try to read score_config from the inferred config
                        try:
                            with open(effective_alt_impute_config, 'r') as f:
                                alt_cfg = json.load(f)
                                config_score_config = alt_cfg.get('score_config', '')
                                if config_score_config:
                                    effective_score_config = config_score_config
                                    logger.info(f"Got score_config from inferred alt_impute_config: {effective_score_config}")
                        except Exception as e:
                            logger.warning(f"Could not read inferred alt_impute_config: {e}")
            
            # Method 3: Try to find score config in same directory as etl_config
            if not effective_score_config or str(effective_score_config).strip() == '':
                if self.etl_config and str(self.etl_config).strip():
                    etl_config_dir = os.path.dirname(str(self.etl_config))
                    # Look for common score config file names
                    for score_filename in ['maggic.json', 'score_config.json', 'scores.json']:
                        potential_score_config = os.path.join(etl_config_dir, score_filename)
                        if os.path.exists(potential_score_config):
                            effective_score_config = potential_score_config
                            logger.info(f"Inferred score_config from etl_config directory: {effective_score_config}")
                            break
        
        logger.info(f"Starting alternative imputation task")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Score config parameter (from Luigi): '{self.score_config}'")
        logger.info(f"Effective score config (actual): '{effective_score_config}'")

        
        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the input data
        logger.info("Loading input data...")
        data = self._load_data(self.input_file)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # If score config is provided, ensure required columns exist
        # Convert to string and check if non-empty (Luigi parameters need explicit string conversion)
        # Check for None, empty string, or 'None' string
        score_config_str = str(effective_score_config).strip() if effective_score_config is not None else ''
        logger.info(f"Score config string value: '{score_config_str}'")
        logger.info(f"Score config string length: {len(score_config_str)}")
        
        if score_config_str and score_config_str != '' and score_config_str.lower() != 'none':
            logger.info(f"Loading score configuration from {score_config_str}")
            from dpplgngr.utils.score_config_utils import (
                get_required_columns_from_config_file, 
                ensure_columns_exist
            )
            
            try:
                required_columns = get_required_columns_from_config_file(score_config_str)
                logger.info(f"Score requires {len(required_columns)} columns")
                
                # Add missing columns with NaN values
                data, added_columns = ensure_columns_exist(data, required_columns)
                
                if added_columns:
                    logger.info(f"Added {len(added_columns)} missing columns for score calculation")
            except Exception as e:
                logger.warning(f"Could not process score config: {e}")
                logger.warning("Continuing without adding score-required columns")
        else:
            logger.warning(f"Score config not provided or empty, skipping column addition")
        
        # Get imputation strategies
        strategies = [s.strip() for s in self.imputation_strategies.split(',')]
        logger.info(f"Imputation strategies: {strategies}")
        
        # Separate numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Found {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")
        logger.info(f"Numeric columns: {numeric_cols}")
        
        # Log if score-required columns are in numeric columns
        if score_config_str and score_config_str != '' and not score_config_str.lower() == 'none':
            from dpplgngr.utils.score_config_utils import get_required_columns_from_config_file
            try:
                required_columns = get_required_columns_from_config_file(score_config_str)
                score_cols_present = [col for col in required_columns if col in numeric_cols]
                score_cols_missing = [col for col in required_columns if col not in data.columns]
                logger.info(f"Score-required columns in numeric data: {len(score_cols_present)}/{len(required_columns)}")
                if score_cols_missing:
                    logger.error(f"Score-required columns NOT in DataFrame: {score_cols_missing}")
                if len(score_cols_present) < len(required_columns):
                    missing_from_numeric = [col for col in required_columns if col in data.columns and col not in numeric_cols]
                    if missing_from_numeric:
                        logger.error(f"Score-required columns in DataFrame but NOT numeric: {missing_from_numeric}")
                        for col in missing_from_numeric:
                            logger.error(f"  - {col}: dtype={data[col].dtype}")
            except Exception as e:
                logger.error(f"Error checking score columns: {e}")
        
        # Report missing data
        missing_summary = data[numeric_cols].isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        if len(missing_summary) > 0:
            logger.info(f"Missing data in numeric columns:\n{missing_summary}")
        else:
            logger.info("No missing data in numeric columns")
        
        # Apply each imputation strategy
        for strategy in strategies:
            logger.info(f"\n{'='*60}")
            logger.info(f"Applying {strategy} imputation...")
            logger.info(f"{'='*60}")
            
            try:
                # Debug: Check columns before imputation
                logger.info(f"Data has {len(data.columns)} columns before imputation")
                if score_config_str and score_config_str != '' and not score_config_str.lower() == 'none':
                    from dpplgngr.utils.score_config_utils import get_required_columns_from_config_file
                    try:
                        required_columns = get_required_columns_from_config_file(score_config_str)
                        logger.info(f"Checking {len(required_columns)} score-required columns before imputation:")
                        for col in required_columns:
                            if col in data.columns:
                                logger.info(f"  ✓ {col} present")
                            else:
                                logger.error(f"  ✗ {col} MISSING")
                    except Exception as e:
                        logger.error(f"Error checking columns before imputation: {e}")
                
                imputed_data = self._impute_data(data.copy(), strategy, numeric_cols, categorical_cols)
                
                # Debug: Check columns after imputation
                logger.info(f"Imputed data has {len(imputed_data.columns)} columns after imputation")
                if score_config_str and score_config_str != '' and not score_config_str.lower() == 'none':
                    from dpplgngr.utils.score_config_utils import get_required_columns_from_config_file
                    try:
                        required_columns = get_required_columns_from_config_file(score_config_str)
                        missing_after = [col for col in required_columns if col not in imputed_data.columns]
                        if missing_after:
                            logger.error(f"CRITICAL: Score-required columns missing after imputation: {missing_after}")
                            logger.error("Columns in imputed_data:")
                            for col in sorted(imputed_data.columns):
                                logger.error(f"  - {col}")
                    except Exception as e:
                        logger.error(f"Error checking columns after imputation: {e}")
                
                # Debug: Show ALL columns right before save
                logger.info("=" * 80)
                logger.info("COLUMNS IN IMPUTED_DATA RIGHT BEFORE SAVING:")
                logger.info(f"Total columns: {len(imputed_data.columns)}")
                for i, col in enumerate(sorted(imputed_data.columns), 1):
                    logger.info(f"  {i}. {col}")
                logger.info("=" * 80)
                
                # Save the imputed data
                output_path = os.path.join(self.output_dir, f'{self.prefix}_{strategy}.parquet')
                imputed_data.to_parquet(output_path, index=True)
                logger.info(f"Saved imputed data to: {output_path}")
                
                # Debug: Verify what was actually saved
                logger.info("Verifying saved file...")
                saved_df = pd.read_parquet(output_path)
                logger.info(f"Saved file has {len(saved_df.columns)} columns, {len(saved_df)} rows")
                if len(saved_df.columns) != len(imputed_data.columns):
                    logger.error(f"MISMATCH: Expected {len(imputed_data.columns)} columns, but saved file has {len(saved_df.columns)}")
                    missing_in_saved = set(imputed_data.columns) - set(saved_df.columns)
                    if missing_in_saved:
                        logger.error(f"Columns missing from saved file: {sorted(missing_in_saved)}")
                
                # Save metadata
                metadata = {
                    'strategy': strategy,
                    'input_file': self.input_file,
                    'output_file': output_path,
                    'n_rows': len(imputed_data),
                    'n_cols': len(imputed_data.columns),
                    'numeric_cols': numeric_cols,
                    'categorical_cols': categorical_cols,
                }
                metadata_path = os.path.join(self.output_dir, f'{self.prefix}_{strategy}_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Saved metadata to: {metadata_path}")
                
            except Exception as e:
                logger.error(f"Failed to apply {strategy} imputation: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"\n{'='*60}")
        logger.info("Alternative imputation complete!")
        logger.info(f"{'='*60}")
    
    def _load_data(self, file_path):
        """Load data from various file formats."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext == '.parquet':
            return pd.read_parquet(file_path)
        elif file_ext == '.feather':
            return pd.read_feather(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _impute_data(self, data, strategy, numeric_cols, categorical_cols):
        """
        Apply imputation strategy to the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with missing values
        strategy : str
            Imputation strategy name
        numeric_cols : list
            List of numeric column names
        categorical_cols : list
            List of categorical column names
        
        Returns
        -------
        pd.DataFrame
            Imputed data
        """
        if not _sklearn_available:
            raise ImportError("sklearn is required for imputation")
        
        # Identify columns with 100% missing values
        all_missing_numeric = [col for col in numeric_cols if data[col].isnull().all()]
        all_missing_categorical = [col for col in categorical_cols if data[col].isnull().all()]
        
        if all_missing_numeric:
            logger.warning(f"Found {len(all_missing_numeric)} numeric columns with 100% missing values:")
            for col in all_missing_numeric:
                logger.warning(f"  - {col}")
            logger.warning("These columns will be filled with 0 as no imputation is possible.")
        
        if all_missing_categorical:
            logger.warning(f"Found {len(all_missing_categorical)} categorical columns with 100% missing values:")
            for col in all_missing_categorical:
                logger.warning(f"  - {col}")
            logger.warning("These columns will be filled with 'missing' as no imputation is possible.")
        
        # Separate columns that can be imputed from those that are all missing
        numeric_to_impute = [col for col in numeric_cols if col not in all_missing_numeric]
        categorical_to_impute = [col for col in categorical_cols if col not in all_missing_categorical]
        
        # Create imputer based on strategy
        if strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
            logger.info("Using Mean imputation (replaces missing with column mean)")
        elif strategy == 'median':
            imputer = SimpleImputer(strategy='median')
            logger.info("Using Median imputation (robust to outliers)")
        elif strategy == 'most_frequent':
            imputer = SimpleImputer(strategy='most_frequent')
            logger.info("Using Mode imputation (most frequent value)")
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            logger.info("Using KNN imputation (k=5, uses similar patients)")
        elif strategy in ['iterative', 'mice']:
            imputer = IterativeImputer(max_iter=10, random_state=42)
            logger.info("Using MICE (Multiple Imputation by Chained Equations)")
            logger.info("  - Iteratively models each feature as function of others")
            logger.info("  - Maximum 10 iterations, random_state=42 for reproducibility")
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
        
        # Impute numeric columns (only those with some non-missing values)
        if len(numeric_to_impute) > 0 and data[numeric_to_impute].isnull().any().any():
            logger.info(f"Imputing {len(numeric_to_impute)} numeric columns...")
            data[numeric_to_impute] = imputer.fit_transform(data[numeric_to_impute])
            
            # Save the imputer model
            imputer_path = os.path.join(self.output_dir, f'imputer_{strategy}.joblib')
            dump(imputer, imputer_path)
            logger.info(f"Saved imputer model to: {imputer_path}")
        
        # Fill columns with 100% missing values with default value (0 for numeric)
        if all_missing_numeric:
            logger.info(f"Filling {len(all_missing_numeric)} all-NaN numeric columns with 0.0:")
            for col in all_missing_numeric:
                data[col] = 0.0
                logger.info(f"  ✓ Filled '{col}' with 0.0")
        
        # Impute categorical columns (use mode, only those with some non-missing values)
        if len(categorical_to_impute) > 0 and data[categorical_to_impute].isnull().any().any():
            logger.info(f"Imputing {len(categorical_to_impute)} categorical columns with mode...")
            cat_imputer = SimpleImputer(strategy='most_frequent')
            data[categorical_to_impute] = cat_imputer.fit_transform(data[categorical_to_impute])
        
        # Fill columns with 100% missing values with default value ('missing' for categorical)
        if all_missing_categorical:
            logger.info(f"Filling {len(all_missing_categorical)} all-NaN categorical columns with 'missing':")
            for col in all_missing_categorical:
                data[col] = 'missing'
                logger.info(f"  ✓ Filled '{col}' with 'missing'")
        
        # Verify no missing values remain
        remaining_missing = data.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"Warning: {remaining_missing} missing values remain after imputation")
        else:
            logger.info("All missing values successfully imputed")
        
        # Debug: Show columns being returned
        logger.info(f"_impute_data returning dataframe with {len(data.columns)} columns")
        logger.debug(f"Columns: {sorted(data.columns.tolist())}")
        
        return data


def create_alternative_imputations(input_file, output_dir, strategies=None, prefix='imputed'):
    """
    Standalone function to create alternative imputations without Luigi.
    
    Parameters
    ----------
    input_file : str
        Path to the preprocessed data file (before imputation)
    output_dir : str
        Directory to save imputed datasets
    strategies : list or str, optional
        List of imputation strategies or comma-separated string
        Default: ['mean', 'median', 'knn', 'mice']
    prefix : str, optional
        Prefix for output files (default: 'imputed')
    
    Returns
    -------
    dict
        Dictionary mapping strategy names to output file paths
    """
    if strategies is None:
        strategies = ['mean', 'median', 'knn', 'mice']
    elif isinstance(strategies, str):
        strategies = [s.strip() for s in strategies.split(',')]
    
    strategies_str = ','.join(strategies)
    
    task = AlternativeImputation(
        input_file=input_file,
        output_dir=output_dir,
        imputation_strategies=strategies_str,
        prefix=prefix
    )
    
    task.run()
    
    # Return paths to output files
    return {
        strategy: os.path.join(output_dir, f'{prefix}_{strategy}.parquet')
        for strategy in strategies
    }


if __name__ == '__main__':
    luigi.run()
