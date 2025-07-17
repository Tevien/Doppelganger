import os
import dask.dataframe as dd
import polars as pl
import pandas as pd
import numpy as np
import logging

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

def return_subset(filename, cols, index_col=None, blocksize=10000):
    """
    this function will return a subset of the dataframe

    Args:
    filename: str
        The filename of the dataframe
    cols: list
        The columns to return
    index_col: str
        The index column
    blocksize: int
        The blocksize to use
    """
    # Is the file a parquet file?
    if '.parquet' in filename:
        df = dd.read_parquet(filename, columns=cols+[index_col])
    elif '.feather' in filename:
        df = dd.from_pandas(pd.read_feather(filename, columns=cols+[index_col]), npartitions=3)
    else:
        df = dd.read_csv(filename, blocksize=blocksize)
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