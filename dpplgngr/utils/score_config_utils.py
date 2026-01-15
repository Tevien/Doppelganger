"""
Score Configuration Utilities

Helper functions for working with score configuration files.
These utilities help ensure that all columns required for score calculation
are available in the dataset, even if they need to be imputed.

Author: SB
Date: 2025-10-31
"""

import json
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def load_score_config(config_path: str) -> dict:
    """
    Load a score configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the score configuration JSON file
    
    Returns
    -------
    dict
        Score configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def get_required_columns_from_config(config: dict) -> Set[str]:
    """
    Extract the list of required columns from a score configuration.
    
    This handles both single score configurations and multiple score configurations.
    
    Parameters
    ----------
    config : dict
        Score configuration dictionary containing either:
        - 'column_mapping': for single score
        - 'scores': list of score configs for multiple scores
    
    Returns
    -------
    set
        Set of column names required for score calculation
    
    Examples
    --------
    Single score config:
    >>> config = {
    ...     "score_name": "maggic",
    ...     "column_mapping": {
    ...         "sex": "patient_demographics_gender",
    ...         "age": "patient_demographics_age"
    ...     }
    ... }
    >>> get_required_columns_from_config(config)
    {'patient_demographics_gender', 'patient_demographics_age'}
    
    Multiple scores config:
    >>> config = {
    ...     "scores": [
    ...         {
    ...             "score_name": "maggic",
    ...             "column_mapping": {"sex": "gender", "age": "age"}
    ...         },
    ...         {
    ...             "score_name": "audit",
    ...             "column_mapping": {"alcohol": "drinks_per_week"}
    ...         }
    ...     ]
    ... }
    >>> get_required_columns_from_config(config)
    {'gender', 'age', 'drinks_per_week'}
    """
    required_columns = set()
    
    # Check if this is a single score configuration
    if 'column_mapping' in config:
        required_columns.update(config['column_mapping'].values())
    
    # Check if this is a multiple scores configuration
    if 'scores' in config:
        for score_config in config['scores']:
            if 'column_mapping' in score_config:
                required_columns.update(score_config['column_mapping'].values())
    
    return required_columns


def get_required_columns_from_config_file(config_path: str) -> Set[str]:
    """
    Load a score configuration file and extract required columns.
    
    Parameters
    ----------
    config_path : str
        Path to the score configuration JSON file
    
    Returns
    -------
    set
        Set of column names required for score calculation
    """
    config = load_score_config(config_path)
    return get_required_columns_from_config(config)


def ensure_columns_exist(df, required_columns: Set[str], fill_value=None):
    """
    Ensure that all required columns exist in a DataFrame.
    
    Missing columns are added with NaN values (or specified fill_value) as float64 dtype.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    required_columns : set
        Set of column names that must exist
    fill_value : any, optional
        Value to use for missing columns (default: None, which creates NaN)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all required columns present
    list
        List of column names that were added
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'a': [1, 2, 3]})
    >>> required = {'a', 'b', 'c'}
    >>> df_updated, added = ensure_columns_exist(df, required)
    >>> print(df_updated.columns.tolist())
    ['a', 'b', 'c']
    >>> print(added)
    ['b', 'c']
    """
    import pandas as pd
    import numpy as np
    
    existing_columns = set(df.columns)
    missing_columns = required_columns - existing_columns
    
    added_columns = []
    for col in missing_columns:
        # Use float64 dtype to ensure numeric columns work with imputation
        if fill_value is not None:
            df[col] = fill_value
        else:
            df[col] = np.nan
        # Explicitly set dtype to float64 to ensure it's treated as numeric
        df[col] = df[col].astype('float64')
        added_columns.append(col)
    
    if added_columns:
        logger.info(f"Added {len(added_columns)} missing columns with NaN values (dtype: float64):")
        for col in sorted(added_columns):
            logger.info(f"  - {col}")
    
    return df, added_columns


def get_all_score_configs_from_directory(directory: str) -> List[str]:
    """
    Find all score configuration files in a directory.
    
    Parameters
    ----------
    directory : str
        Path to directory containing score configuration files
    
    Returns
    -------
    list
        List of paths to score configuration JSON files
    """
    import os
    import glob
    
    pattern = os.path.join(directory, '*.json')
    config_files = glob.glob(pattern)
    
    # Filter to only include files that look like score configs
    score_configs = []
    for config_file in config_files:
        try:
            config = load_score_config(config_file)
            # Check if it has score-related keys
            if 'score_name' in config or 'scores' in config or 'column_mapping' in config:
                score_configs.append(config_file)
        except Exception as e:
            logger.debug(f"Skipping {config_file}: {e}")
    
    return score_configs


def merge_required_columns_from_multiple_configs(config_paths: List[str]) -> Set[str]:
    """
    Get the union of all required columns from multiple score configuration files.
    
    Parameters
    ----------
    config_paths : list
        List of paths to score configuration JSON files
    
    Returns
    -------
    set
        Union of all required column names across all configurations
    """
    all_required_columns = set()
    
    for config_path in config_paths:
        try:
            required_cols = get_required_columns_from_config_file(config_path)
            all_required_columns.update(required_cols)
            logger.info(f"Found {len(required_cols)} required columns in {config_path}")
        except Exception as e:
            logger.warning(f"Could not process config file {config_path}: {e}")
    
    return all_required_columns
