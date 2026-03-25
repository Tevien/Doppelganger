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
    # Deduplicate values (multiple codes may map to the same column name)
    unique_categories = list(dict.fromkeys(code_map.values()))
    grouped['target_col'] = grouped['target_col'].astype('category').cat.set_categories(unique_categories)

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
    # Remove PATIENTCONTACTID columns (case-insensitive) that cause many-to-many joins
    for _target_df in [_df, _df_pp]:
        if _target_df is None:
            continue
        cols_to_drop = [c for c in _target_df.columns if c.lower() == 'patientcontactid']
        if cols_to_drop:
            if _target_df is _df:
                _df = _df.drop(columns=cols_to_drop)
            else:
                _df_pp = _df_pp.drop(columns=cols_to_drop)

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

            if common_cols:
                # Both have matching index names - merge on index explicitly
                # This avoids accidentally joining on other common columns (e.g. PATIENTCONTACTID)
                logging.info(f"Merging on common index: {common_cols}")
                _df_pp = _df_pp_std.merge(_df_std, how="left", left_index=True, right_index=True)
            elif _df_pp_std.index.name and _df_std.index.name:
                if _df_pp_std.index.name == _df_std.index.name:
                    _df_pp = _df_pp_std.merge(_df_std, how="left", left_index=True, right_index=True)
                else:
                    # Index names don't match, reset and merge on common columns
                    _df_pp_reset = _df_pp_std.reset_index()
                    _df_reset = _df_std.reset_index()
                    common_reset_cols = list(set(_df_pp_reset.columns) & set(_df_reset.columns))
                    if common_reset_cols:
                        _df_pp = _df_pp_reset.merge(_df_reset, how="left", on=common_reset_cols)
                        # Try to restore index if possible
                        index_col = _df_pp_std.index.name or _df_std.index.name
                        if index_col in _df_pp.columns:
                            _df_pp = _df_pp.set_index(index_col)
                    else:
                        _df_pp = _df_pp_std.merge(_df_std, how="left")
            else:
                # No common index, fall back to standard merge
                logging.warning("No common index found, falling back to standard merge on common columns")
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
    
    logging.info("df_pp postmerge")
    logging.info(f"Columns after merge: {_df_pp.columns.tolist()}")
    logging.info(f"Shape after merge: {dask_shape(_df_pp)}")
    
    # Normalize schema across all partitions to prevent schema mismatch errors
    # This must happen before any computation (including .head())
    _df_pp, pa_schema = normalize_dask_schema(_df_pp)
    
    # Now we can safely sample for logging
    try:
        _df_pp_20 = _df_pp.head(20)
        logging.info("Sample of merged data:")
        logging.info(_df_pp_20)
    except Exception as e:
        logging.warning(f"Could not compute sample after merge: {e}")
    
    return _df_pp, pa_schema

def standardize_dataframe_for_merge(df):
    """
    Standardize a DataFrame for consistent merging by normalizing data types
    and structures that commonly change during Parquet I/O operations.
    
    For Dask DataFrames, this does minimal work to avoid triggering computation.
    The real normalization happens in normalize_dask_schema after merging.
    """
    # For Dask DataFrames, just return as-is to avoid premature computation
    # Schema normalization will happen after merge in normalize_dask_schema
    if isinstance(df, dd.DataFrame):
        return df
    
    # Work with a copy to avoid modifying the original (pandas only)
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
    
    # Handle column data types (pandas only at this point)
    for col in df_std.columns:
        # Handle float columns that might have been integers
        if pd.api.types.is_float_dtype(df_std[col]):
            # Check if all non-null values are actually integers
            non_null_values = df_std[col].dropna()
            if len(non_null_values) > 0:
                # Check if all values are whole numbers
                try:
                    if (non_null_values == non_null_values.astype(int)).all():
                        # Convert to nullable integer type
                        df_std[col] = df_std[col].astype('Int64')
                except:
                    # If conversion fails, keep as float
                    pass
        
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
            try:
                df_std[col] = df_std[col].astype(str)
            except:
                pass
    
    return df_std

def normalize_dask_schema(df):
    """
    Normalize the schema across all partitions of a Dask DataFrame.
    This ensures consistent data types across partitions before saving to Parquet.
    
    Args:
        df: dask.dataframe.DataFrame to normalize
        
    Returns:
        tuple: (dask.dataframe.DataFrame with consistent schema, pyarrow.Schema or None)
    """
    if not isinstance(df, dd.DataFrame):
        # If it's a pandas DataFrame, just return it with no schema
        return df, None
    
    logging.info("Normalizing Dask DataFrame schema across partitions...")
    
    try:
        # Build a dtype mapping that's safe for all partitions
        dtype_map = {}
        for col in df.columns:
            current_dtype = df[col].dtype
            
            # For object/string columns, keep as object
            if current_dtype == 'object' or str(current_dtype) == 'string' or 'string' in str(current_dtype):
                dtype_map[col] = 'object'
            # For datetime/timedelta, keep as-is
            elif pd.api.types.is_datetime64_any_dtype(current_dtype):
                dtype_map[col] = current_dtype
            elif pd.api.types.is_timedelta64_dtype(current_dtype):
                dtype_map[col] = current_dtype
            # For float columns, keep as float64
            elif pd.api.types.is_float_dtype(current_dtype):
                dtype_map[col] = 'float64'
            # For ALL integer columns (including non-nullable), convert to float64
            # This prevents IntCastingNaNError when there are NaN values
            elif pd.api.types.is_integer_dtype(current_dtype):
                dtype_map[col] = 'float64'
            else:
                dtype_map[col] = current_dtype
        
        logging.info(f"Applying dtype conversions: {dtype_map}")
        
        # Define a function to normalize each partition
        def normalize_partition(partition, dtype_map=dtype_map):
            """Normalize a single partition to have consistent types"""
            for col, target_dtype in dtype_map.items():
                if col in partition.columns:
                    try:
                        # Special handling for object/string types
                        if target_dtype == 'object':
                            # Convert to string, handling None/NaN
                            partition[col] = partition[col].astype('object')
                        else:
                            partition[col] = partition[col].astype(target_dtype)
                    except Exception as e:
                        logging.warning(f"Partition-level conversion failed for {col} to {target_dtype}: {e}")
                        # Fallback to object
                        try:
                            partition[col] = partition[col].astype('object')
                        except:
                            pass
            return partition
        
        # Create meta with the correct dtypes
        meta_dict = {}
        for col, dtype in dtype_map.items():
            if dtype == 'object':
                meta_dict[col] = pd.Series([], dtype='object')
            elif isinstance(dtype, str):
                meta_dict[col] = pd.Series([], dtype=dtype)
            else:
                meta_dict[col] = pd.Series([], dtype=dtype)
        
        meta = pd.DataFrame(meta_dict, index=df._meta.index[:0])
        
        # Apply normalization using map_partitions
        df = df.map_partitions(normalize_partition, dtype_map=dtype_map, meta=meta)
        
        # Repartition to clear task graph
        df = df.repartition(npartitions=max(df.npartitions, 1))
        
        # Create PyArrow schema from the meta
        try:
            import pyarrow as pa
            # Convert meta to pyarrow to get the schema
            pa_schema = pa.Schema.from_pandas(meta, preserve_index=True)
            logging.info(f"Created PyArrow schema: {pa_schema}")
        except Exception as e:
            logging.warning(f"Could not create PyArrow schema: {e}")
            pa_schema = None
        
        logging.info("Schema normalization complete.")
        return df, pa_schema
        
    except Exception as e:
        logging.error(f"Error during schema normalization: {e}")
        logging.info("Attempting fallback: compute and recreate...")
        try:
            # Last resort: compute entire dataframe and recreate
            df_pandas = df.compute()
            df = dd.from_pandas(df_pandas, npartitions=max(len(df_pandas) // 10000, 1))
            logging.info("Fallback successful - dataframe recomputed.")
            
            # Create schema from the computed dataframe
            try:
                import pyarrow as pa
                pa_schema = pa.Schema.from_pandas(df_pandas, preserve_index=True)
            except:
                pa_schema = None
            
            return df, pa_schema
        except Exception as e2:
            logging.error(f"Fallback also failed: {e2}")
            raise