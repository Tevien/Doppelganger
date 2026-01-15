"""
Missing Value Handler for Metrics Calculation

This module provides utilities for handling missing values before calculating
audit metrics and privacy scores. It preserves the original data while creating
temporary filled versions for metrics that don't handle NaNs.

The philosophy is:
1. Preserve missingness in synthetic data (for realism)
2. Fill NaNs temporarily only for metric calculation
3. Use imputation strategies consistent with the data's characteristics

Author: SB
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class MissingValueHandler:
    """
    Handle missing values for metric calculation without modifying original data.
    
    This class provides methods to create temporary filled versions of data
    for metrics that cannot handle NaNs, while preserving the original data
    with its missing value patterns.
    """
    
    def __init__(self, strategy: str = 'conservative'):
        """
        Initialize the missing value handler.
        
        Parameters
        ----------
        strategy : str, default='conservative'
            Strategy for handling missing values:
            - 'conservative': Use median for numeric, mode for categorical
            - 'mean': Use mean for numeric, mode for categorical
            - 'constant': Use a constant value (requires fill_value)
            - 'forward_fill': Forward fill missing values
            - 'interpolate': Linear interpolation for numeric columns
        """
        self.strategy = strategy
        self.fill_values_ = {}
        
    def fit(self, data: pd.DataFrame) -> 'MissingValueHandler':
        """
        Learn fill values from the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to learn fill values from
            
        Returns
        -------
        self : MissingValueHandler
            Fitted handler
        """
        self.fill_values_ = {}
        
        for col in data.columns:
            if data[col].isna().all():
                # All missing - use 0 for numeric, 'missing' for categorical
                if pd.api.types.is_numeric_dtype(data[col]):
                    self.fill_values_[col] = 0
                else:
                    self.fill_values_[col] = 'missing'
            elif pd.api.types.is_numeric_dtype(data[col]):
                # Numeric column
                if self.strategy == 'conservative' or self.strategy == 'median':
                    self.fill_values_[col] = data[col].median()
                elif self.strategy == 'mean':
                    self.fill_values_[col] = data[col].mean()
                else:
                    self.fill_values_[col] = data[col].median()
            else:
                # Categorical column - use mode
                mode_values = data[col].mode()
                if len(mode_values) > 0:
                    self.fill_values_[col] = mode_values[0]
                else:
                    self.fill_values_[col] = 'missing'
                    
        logger.info(f"Learned fill values using '{self.strategy}' strategy")
        return self
    
    def transform(self, data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Fill missing values in data using learned fill values.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to fill
        inplace : bool, default=False
            If True, modify data in place. If False, return a copy.
            
        Returns
        -------
        pd.DataFrame
            Data with missing values filled
        """
        if not self.fill_values_:
            raise ValueError("Handler must be fitted before transform. Call fit() first.")
        
        if not inplace:
            data = data.copy()
            
        for col, fill_value in self.fill_values_.items():
            if col in data.columns:
                data[col] = data[col].fillna(fill_value)
                
        return data
    
    def fit_transform(self, data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to fit and transform
        inplace : bool, default=False
            If True, modify data in place. If False, return a copy.
            
        Returns
        -------
        pd.DataFrame
            Data with missing values filled
        """
        self.fit(data)
        return self.transform(data, inplace=inplace)
    
    def get_missing_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics about missing values in the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to analyze
            
        Returns
        -------
        pd.DataFrame
            Statistics about missing values per column
        """
        stats = []
        total_rows = len(data)
        
        for col in data.columns:
            missing_count = data[col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            
            stats.append({
                'column': col,
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'dtype': str(data[col].dtype),
                'fill_value': self.fill_values_.get(col, None)
            })
            
        return pd.DataFrame(stats)


def prepare_data_for_metrics(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    strategy: str = 'conservative',
    preserve_original: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Prepare original and synthetic data for metric calculation by handling NaNs.
    
    This function creates filled versions of both datasets using consistent
    fill values learned from the original data. The original data is used
    to learn the fill values to ensure consistency.
    
    Parameters
    ----------
    original_data : pd.DataFrame
        Original/real data
    synthetic_data : pd.DataFrame
        Synthetic data
    strategy : str, default='conservative'
        Strategy for handling missing values ('conservative', 'mean', etc.)
    preserve_original : bool, default=True
        If True, return copies of the data (recommended)
        
    Returns
    -------
    original_filled : pd.DataFrame
        Original data with missing values filled
    synthetic_filled : pd.DataFrame
        Synthetic data with missing values filled
    fill_info : dict
        Information about the filling process
    """
    logger.info("="*60)
    logger.info("PREPARING DATA FOR METRICS (HANDLING NaN VALUES)")
    logger.info("="*60)
    
    # Create handler and fit on original data
    handler = MissingValueHandler(strategy=strategy)
    handler.fit(original_data)
    
    # Get missing stats before filling
    original_stats = handler.get_missing_stats(original_data)
    synthetic_stats = handler.get_missing_stats(synthetic_data)
    
    # Log missing value information
    orig_missing_cols = original_stats[original_stats['missing_count'] > 0]
    synth_missing_cols = synthetic_stats[synthetic_stats['missing_count'] > 0]
    
    logger.info(f"Original data: {len(orig_missing_cols)} columns with missing values")
    logger.info(f"Synthetic data: {len(synth_missing_cols)} columns with missing values")
    
    if len(orig_missing_cols) > 0:
        logger.info("\nColumns with missing values in original data:")
        for _, row in orig_missing_cols.iterrows():
            logger.info(f"  {row['column']}: {row['missing_pct']:.2f}% missing "
                       f"(will fill with {row['fill_value']})")
    
    # Fill missing values
    original_filled = handler.transform(original_data, inplace=False)
    synthetic_filled = handler.transform(synthetic_data, inplace=False)
    
    # Verify no missing values remain
    orig_remaining = original_filled.isna().sum().sum()
    synth_remaining = synthetic_filled.isna().sum().sum()
    
    if orig_remaining > 0:
        logger.warning(f"Warning: {orig_remaining} NaN values remain in original data after filling")
    if synth_remaining > 0:
        logger.warning(f"Warning: {synth_remaining} NaN values remain in synthetic data after filling")
    
    logger.info(f"\nFilling complete. Strategy: '{strategy}'")
    logger.info(f"Original data NaN count: {original_data.isna().sum().sum()} → {orig_remaining}")
    logger.info(f"Synthetic data NaN count: {synthetic_data.isna().sum().sum()} → {synth_remaining}")
    logger.info("="*60 + "\n")
    
    # Compile info about the filling process
    fill_info = {
        'strategy': strategy,
        'fill_values': handler.fill_values_,
        'original_missing_stats': original_stats.to_dict('records'),
        'synthetic_missing_stats': synthetic_stats.to_dict('records'),
        'original_nans_before': int(original_data.isna().sum().sum()),
        'original_nans_after': int(orig_remaining),
        'synthetic_nans_before': int(synthetic_data.isna().sum().sum()),
        'synthetic_nans_after': int(synth_remaining)
    }
    
    return original_filled, synthetic_filled, fill_info


def selective_fill_for_metric(
    data: pd.DataFrame,
    columns: List[str],
    strategy: str = 'conservative'
) -> pd.DataFrame:
    """
    Fill missing values only in specific columns needed for a metric.
    
    Useful when only certain columns need to be filled for a specific metric,
    while keeping other columns with their original missing patterns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to partially fill
    columns : list of str
        Columns to fill (others are left unchanged)
    strategy : str, default='conservative'
        Strategy for filling
        
    Returns
    -------
    pd.DataFrame
        Data with specified columns filled
    """
    data_filled = data.copy()
    subset = data[columns]
    
    handler = MissingValueHandler(strategy=strategy)
    handler.fit(subset)
    filled_subset = handler.transform(subset, inplace=False)
    
    # Replace only the specified columns
    data_filled[columns] = filled_subset
    
    return data_filled


# Convenience function for backward compatibility
def fillna_for_metrics(
    data: pd.DataFrame,
    strategy: str = 'conservative'
) -> pd.DataFrame:
    """
    Simple convenience function to fill NaNs for metric calculation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to fill
    strategy : str, default='conservative'
        Filling strategy
        
    Returns
    -------
    pd.DataFrame
        Filled data (copy)
    """
    handler = MissingValueHandler(strategy=strategy)
    return handler.fit_transform(data, inplace=False)
