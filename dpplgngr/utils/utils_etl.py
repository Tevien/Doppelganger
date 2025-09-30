import os
import dask.dataframe as dd
import polars as pl
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger('luigi-interface')

def first_non_nan(x):
    return x[np.isfinite(x)][0]

def convert_bytes_to_mb(num):
    """
    this function will convert bytes to MB
    """
    num /= 1024.0**2
    print(num)
    return num


def file_size(file_path):
    """
    this function will return the file size
    """
    file_info = os.stat(file_path)
    print (file_path)
    return convert_bytes_to_mb(file_info.st_size)

def return_subset(df, cols, index_col=None, blocksize=10000):
    """
    this function will return a subset of the dataframe

    Args:
    df: dask.dataframe.DataFrame
        The input dataframe
    cols: list
        The columns to return
    index_col: str
        The index column
    blocksize: int
        The blocksize to use
    """

    # Restrict to specified columns
    df = df.loc[:, cols+[index_col]]

    if index_col is not None:
        df = df.set_index(index_col)
    return df

def vals_to_cols(df, index_col='pseudo_id', code_col='BepalingCode', value_col='uitslagnumeriek', code_map=None, extra_cols=None, blocksize=10000):

    # Filter and map
    df = df[df[code_col].isin(code_map.keys())].copy()
    df['target_col'] = df[code_col].map(code_map)

    # Build tuple with extra columns
    if extra_cols is None:
        extra_cols = []
    tuple_cols = [value_col] + extra_cols
    df['tuple'] = df[tuple_cols].apply(lambda row: tuple(row), axis=1)#, meta=(None, 'object'))

    # Group and pivot
    grouped = df.groupby([index_col, 'target_col'])['tuple'].agg(list).reset_index()
    grouped['target_col'] = grouped['target_col'].astype('category').cat.set_categories(code_map.values())

    print(f"Grouped dataframe shape: {grouped.shape}")
    print(f"Grouped dataframe columns: {grouped.columns.tolist()}")
    print(f"Grouped dataframe head:\n{grouped.head()}")
    computed_df = grouped.compute()
    result = computed_df.pivot(index=index_col, columns="target_col", values='tuple')#.reset_index()
    print(result.head())
    # Make column names strings
    result.columns = result.columns.astype(str)
    return dd.from_pandas(result, npartitions=3)

def checkpoint(_df, _filename):
    """
    this function will checkpoint the dataframe to a parquet file
    """
    _df.to_parquet(_filename, engine='pyarrow', compression='snappy')
    return _filename

def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    if pd.isnull(date):
        return np.nan
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.utcfromtimestamp(timestamp)

# Function to perform analysis on dask dataframe in terms of missingness, types, and distributions
# Send results to logger
def analyze_dataframe(df, sample_size=10000, prefix="PREPROCESS"):
    """
    Analyzes a dask dataframe and prints out information about missingness, types, and distributions.
    
    Args:
        df (dask.dataframe.DataFrame): The input dask dataframe to analyze.
        sample_size (int): The number of rows to sample for analysis.
    """ 
    logger.info(f"{prefix} - Analyzing dataframe...")
    logger.info(f"{prefix} - Dataframe shape: {df.shape}")
    logger.info(f"{prefix} - Dataframe columns: {df.columns.tolist()}")

    # Compute basic statistics
    desc = df.describe().compute()
    logger.info(f"{prefix} - Basic statistics:")
    logger.info(desc)
    
    # Check for missing values
    missing = df.isnull().sum().compute()
    logger.info(f"{prefix} - Missing values per column:")
    logger.info(missing[missing > 0])
    
    # Sample data for distribution analysis
    sample = df.sample(frac=min(sample_size / len(df), 1.0)).compute()
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            logger.info(f"{prefix} - Distribution for numeric column '{col}':")
            logger.info(sample[col].describe())
        elif pd.api.types.is_categorical_dtype(df[col].dtype) or pd.api.types.is_object_dtype(df[col].dtype):
            logger.info(f"{prefix} - Value counts for categorical column '{col}':")
            logger.info(sample[col].value_counts().head(10))
        else:
            logger.info(f"{prefix} - Column '{col}' has unsupported dtype '{df[col].dtype}' for detailed analysis.")

    logger.info(f"{prefix} - Analysis complete.")


def dask_shape(_df):
    a = _df.shape
    a[0].compute(),a[1]
    return a


# def safe_merge(_df, _df_pp):
#     # remove any of ["Patientcontactid", "PatientContactId"] from _df if it exists
#     if "Patientcontactid" in _df.columns:
#         _df = _df.drop(columns=["Patientcontactid"])
#     if "PatientContactId" in _df.columns:
#         _df = _df.drop(columns=["PatientContactId"])

#     logging.info("df_pp premerge")
#     if _df_pp is not None:
#         logging.info(f"Shape before merge: {dask_shape(_df_pp)}")
#     logging.info(f"New data shape: {dask_shape(_df)}")
#     if _df_pp is None:
#         _df_pp = _df
#     else:
#         _df_pp = _df_pp.merge(_df, how="left")
#     _df_pp_20 = _df.head(20)
#     logging.info("df_pp postmerge")
#     logging.info(_df_pp_20)
#     logging.info(f"Shape after merge: {dask_shape(_df_pp)}")
#     return _df_pp

def safe_merge(_df, _df_pp):
    """
    Safely merge dataframes, handling data type inconsistencies that may occur
    after saving/loading from Parquet files.
    """
    # Remove specific columns that might cause issues
    if "Patientcontactid" in _df.columns:
        _df = _df.drop(columns=["Patientcontactid"])
    if "PatientContactId" in _df.columns:
        _df = _df.drop(columns=["PatientContactId"])

    logging.info("df_pp premerge")
    if _df_pp is not None:
        logging.info(f"Shape before merge: {dask_shape(_df_pp)}")
    logging.info(f"New data shape: {dask_shape(_df)}")
    
    if _df_pp is None:
        _df_pp = _df
    else:
        # Standardize both dataframes before merging
        _df_pp_std = standardize_dataframe_for_merge(_df_pp)
        _df_std = standardize_dataframe_for_merge(_df)
        
        try:
            def get_index_names(df):
                if hasattr(df.index, 'names') and df.index.names[0] is not None:
                    return df.index.names
                elif hasattr(df.index, 'name') and df.index.name is not None:
                    return [df.index.name]
                else:
                    return []
            # Find common columns for merging
            common_cols = list(set(get_index_names(_df_pp_std)) & set(get_index_names(_df_std)))

            if not common_cols and _df_pp_std.index.name and _df_std.index.name:
                # Both have index names, use them
                if _df_pp_std.index.name == _df_std.index.name:
                    _df_pp = _df_pp_std.merge(_df_std, how="left", left_index=True, right_index=True)
                else:
                    # Index names don't match, reset and merge on common columns
                    _df_pp_reset = _df_pp_std.reset_index()
                    _df_reset = _df_std.reset_index()
                    common_cols = list(set(_df_pp_reset.columns) & set(_df_reset.columns))
                    if common_cols:
                        _df_pp = _df_pp_reset.merge(_df_reset, how="left", on=common_cols)
                        # Try to restore index if possible
                        index_col = _df_pp_std.index.name or _df_std.index.name
                        if index_col in _df_pp.columns:
                            _df_pp = _df_pp.set_index(index_col)
                    else:
                        _df_pp = _df_pp_std.merge(_df_std, how="left")
            else:
                # Standard merge
                _df_pp = _df_pp_std.merge(_df_std, how="left")
                
        except Exception as e:
            logging.warning(f"Merge failed with standardized dataframes: {e}")
            logging.info("Attempting merge with additional preprocessing...")
            
            # Try alternative approach: reset indices and find common columns
            try:
                _df_pp_reset = _df_pp.reset_index() if _df_pp.index.name else _df_pp
                _df_reset = _df.reset_index() if _df.index.name else _df
                
                # Find common columns
                common_cols = list(set(_df_pp_reset.columns) & set(_df_reset.columns))
                
                if common_cols:
                    _df_pp = _df_pp_reset.merge(_df_reset, how="left", on=common_cols)
                else:
                    # Last resort: merge without specifying columns (pandas will figure it out)
                    _df_pp = _df_pp.merge(_df, how="left")
                    
            except Exception as e2:
                logging.error(f"All merge attempts failed: {e2}")
                logging.error(f"_df_pp columns: {_df_pp.columns.tolist()}")
                logging.error(f"_df columns: {_df.columns.tolist()}")
                logging.error(f"_df_pp index: {_df_pp.index}")
                logging.error(f"_df index: {_df.index}")
                raise e2
    
    _df_pp_20 = _df_pp.head(20)
    logging.info("df_pp postmerge")
    logging.info(_df_pp_20)
    logging.info(f"Shape after merge: {dask_shape(_df_pp)}")
    return _df_pp

def standardize_dataframe_for_merge(df):
    """
    Standardize a DataFrame for consistent merging by normalizing data types
    and structures that commonly change during Parquet I/O operations.
    """
    # Work with a copy to avoid modifying the original
    df_std = df.copy()
    
    # Handle index normalization
    if df_std.index.name:
        # Ensure index has consistent type
        if df_std.index.dtype == 'object':
            # Try to convert to string for consistency
            df_std.index = df_std.index.astype(str)
        elif pd.api.types.is_categorical_dtype(df_std.index):
            # Convert categorical index to object
            df_std.index = df_std.index.astype(str)
    
    # Handle column data types
    for col in df_std.columns:
        # Handle float columns that might have been integers
        if pd.api.types.is_float_dtype(df_std[col]):
            # Check if all non-null values are actually integers
            non_null_values = df_std[col].dropna()
            if len(non_null_values) > 0:
                # Check if all values are whole numbers
                if all(non_null_values == non_null_values.astype(int)):
                    # Convert to nullable integer type
                    df_std[col] = df_std[col].astype('Int64')
        
        # Handle categorical columns
        elif pd.api.types.is_categorical_dtype(df_std[col]):
            # Convert to object for consistent merging
            df_std[col] = df_std[col].astype('object')
        
        # Handle string columns
        elif df_std[col].dtype == 'string':
            df_std[col] = df_std[col].astype('object')
        
        # Handle object columns that might need string conversion
        elif df_std[col].dtype == 'object':
            # Ensure all values are strings (for consistent comparison)
            df_std[col] = df_std[col].astype(str)
    
    return df_std