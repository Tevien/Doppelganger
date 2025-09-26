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