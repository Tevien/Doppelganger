import dask.dataframe as dd
import pandas as pd
import logging
from dpplgngr.utils.definitions import ace_atc, beta_atc, make_classification_map
import inspect

logger = logging.getLogger('luigi-interface')

def datetime_keepfirst(_df, **kwargs):
    col_to_date = kwargs.get('col_to_date', None)
    sort_col = kwargs.get('sort_col', None)
    drop_col = kwargs.get('drop_col', None)

    _df[col_to_date] = dd.to_datetime(_df[col_to_date])
    _df = _df.sort_values(by=[sort_col])
    
    # If drop_col is the index then reset and set again
    reset = False

    if drop_col == _df.index.name:
        _df = _df.reset_index(drop=False)
        logging.info("Drop col is the index")
        reset = True

    _df = _df.drop_duplicates(subset=[drop_col], keep='first')
    
    if reset:
        _df = _df.set_index(drop_col)
    return _df

def keepfirst(_df, **kwargs):
    sort_col = kwargs.get('sort_col', None)
    drop_col = kwargs.get('drop_col', None)

    _df = _df.sort_values(by=[sort_col])
    
    # If drop_col is the index then reset and set again
    reset = False

    if drop_col == _df.index.name:
        _df = _df.reset_index(drop=False)
        logging.info("Drop col is the index")
        reset = True

    _df = _df.drop_duplicates(subset=[drop_col], keep='first')
    
    if reset:
        _df = _df.set_index(drop_col)
    return _df

def datetime(_df, **kwargs):
    col_to_date = kwargs.get('col_to_date', None)
    _df[col_to_date] = dd.to_datetime(_df[col_to_date])
    return _df

def diagnoseprocess_simple(_df, **kwargs):
    disease = kwargs.get('disease', None)
    search_col = kwargs.get('search_col', None)
    id_col = kwargs.get('id_col', None)
    _df = _df.fillna(value={search_col: "0"})
    # Make binary column for diabetes
    _df[disease] = _df.apply(lambda x: 1 if disease in x[search_col].lower() else 0, axis=1,
                             meta=pd.Series(dtype='int', name=disease))
    reset=False
    # Group by pseudo_id and get max
    if id_col == _df.index.name:
        _df = _df.reset_index(drop=False)
        logging.info("Id col is the index")
        reset = True
    _df = _df[[disease, id_col]].groupby(id_col).max().reset_index()
    if reset:
        _df = _df.set_index(id_col)
    return _df

def smokerprocess(_df, **kwargs):
    smoker = kwargs.get('smoking', None)
    id_col = kwargs.get('id_col', None)
    # Fill IsHuidigeRoker with 'missing' if nan
    _df = _df.fillna(value={smoker: "missing"})
    # Map to binary
    map_roken = {'missing': 0, 'Nee': 0, 'N.b.': 0, 'Ja': 1}
    _df[smoker] = _df[smoker].map(map_roken)
    
    # Group by pseudo_id and get max
    reset = False
    if id_col == _df.index.name:
        _df = _df.reset_index(drop=False)
        logging.info("Id col is the index")
        reset = True
    #_df = _df[[smoker, id_col]].groupby(id_col).max().reset_index()
    if reset:
        _df = _df.set_index(id_col)
    return _df

def ace(_df, **kwargs):
    meds = kwargs.get('meds', None)
    out_col = kwargs.get('out_col', None)
    id_col = kwargs.get('id_col', None)

    logger.info("Looking for ACEi with codes:", ace_atc)

    _df = _df.fillna(value={meds: "missing"})
    _df = _df.mask(_df == 'NA', "missing")
    _df[out_col] = _df.apply(lambda x: 1 if x[meds] in ace_atc else 0, axis=1,
                             meta=pd.Series(dtype='int', name=out_col))
    # If id_col is the index then reset and set again
    reset = False
    if id_col == _df.index.name:
        _df = _df.reset_index(drop=False)
        logging.info("Id col is the index")
        reset = True
    _df_ace = _df[[out_col, id_col]].groupby(id_col).max().reset_index()
    if reset:
        _df_ace = _df.set_index(id_col)

    return _df_ace

def beta(_df, **kwargs):
    meds = kwargs.get('meds', None)
    out_col = kwargs.get('out_col', None)
    id_col = kwargs.get('id_col', None)

    # Beta blockers
    beta_atc = list(beta_atc.values())
    logging.info("Looking for beta blockers with codes:", beta_atc)

    _df = _df.fillna(value={meds: "missing"})
    _df = _df.mask(_df == 'NA', "missing")
    _df[out_col] = _df.apply(lambda x: 1 if x[meds] in beta_atc else 0, axis=1,
                             meta=pd.Series(dtype='int', name=out_col))
    # If id_col is the index then reset and set again
    reset = False
    if id_col == _df.index.name:
        _df = _df.reset_index(drop=False)
        logging.info("Id col is the index")
        reset = True
    _df_beta = _df[[out_col, id_col]].groupby(id_col).max().reset_index()
    if reset:
        _df_beta = _df_beta.set_index(id_col)

    return _df_beta


# TODO: There is an issue in which if you try to do multiple classifications in the same chain
# it takes only the first one if there are for example overlapping ICD 10 codes then 
# the second classification will never make it to the final columns
def classify(_df, **kwargs):
    classification_map = kwargs.get('classification_map', None) # dictionary of {col: {class1: [values], class2: [values]}}
    out_col = kwargs.get('out_col', None)
    input_col = kwargs.get('input_col', None)
    id_col = kwargs.get('id_col', None)
    """
    Classify values in a column based on a classification map.
    The classification map is a dictionary where keys are the classification names
    and values are lists of values that belong to that classification.
    """
    if classification_map is None:
        raise ValueError("classification_map must be provided")
    if out_col is None:
        raise ValueError("out_col must be provided")
    if input_col is None:
        raise ValueError("input_col must be provided")
    if id_col is None:
        raise ValueError("id_col must be provided")

    classification_map = make_classification_map(classification_map)
    # Create a new column for classification
    _df[out_col] = _df.apply(lambda x: next((k for k, v in classification_map.items() if x[input_col] in v), 'Other'),
                             axis=1, meta=pd.Series(dtype='object', name=out_col))

    return _df


def chain(_df, **kwargs):
    # Get the functions
    funcs = kwargs.get('funcs', None)
    merge_id = kwargs.get('merge_id', None)

    df_pandas = pd.DataFrame()
    final = dd.from_pandas(df_pandas, npartitions=1)
    
    for func in funcs:
        # Get func kwargs
        f_kwargs = kwargs['kwargs'].get(func, None)
        i_df = function_to_execute(func)(_df, **f_kwargs)
        if len(final)==0:
            final = i_df
        else:
            final = dd.merge(final, i_df, on=merge_id, how='outer')
    return final

def diff(_df, **kwargs):
    out_col = kwargs.get('out_col', None)
    start_col = kwargs.get('start', None)
    end_col = kwargs.get('end', None)
    level = kwargs.get('level', None)

    series_start = getattr(_df, start_col)
    series_end = getattr(_df,end_col)
    if level == "year":
        # Check if any series is datetime
        if series_start.dtype == 'datetime64[ns]':
            series_start = series_start.apply(lambda x: x.year)
        if series_end.dtype == 'datetime64[ns]':
            series_end = series_end.apply(lambda x: x.year)
    _df[out_col] = series_end - series_start

    return _df

def dg_map(_df, **kwargs):
    out_col = kwargs.get('out_col', None)
    map_dict = kwargs.get('map', None)
    _df[out_col] = _df[out_col].map(map_dict)
    return _df

def fillna(_df, **kwargs):
    out_col = kwargs.get('out_col', None)
    values = kwargs.get('values', None)
    _df = _df.fillna(value=values)
    return _df

def dropna(_df, **kwargs):
    out_col = kwargs.get('out_col', None)
    subset = kwargs.get('subset', None)
    _df = _df.dropna(subset=subset)
    return _df

def pattern_match(_df, **kwargs):
    """
    Create multiple binary columns based on pattern matching in a search column.
    
    Parameters:
    - id_col: Column to group by
    - search_col: Column to search for patterns
    - pattern_dict: Dictionary where keys are output column names and values are patterns to match
    - case_sensitive: Whether matching should be case sensitive (default: False)
    - group_by_id: Whether to group by id_col and take max (default: True)
    """
    id_col = kwargs.get('id_col', None)
    search_col = kwargs.get('search_col', None)
    pattern_dict = kwargs.get('pattern_dict', None)
    case_sensitive = kwargs.get('case_sensitive', False)
    group_by_id = kwargs.get('group_by_id', True)
    
    if id_col is None:
        raise ValueError("id_col must be provided")
    if search_col is None:
        raise ValueError("search_col must be provided")
    if pattern_dict is None:
        raise ValueError("pattern_dict must be provided")
    
    # Fill NaN values in search column
    _df = _df.fillna(value={search_col: ""})
    
    # Create binary columns for each pattern
    for out_col, pattern in pattern_dict.items():
        if case_sensitive:
            _df[out_col] = _df[search_col].str.contains(pattern, na=False, regex=True).astype(int)
        else:
            _df[out_col] = _df[search_col].str.contains(pattern, case=False, na=False, regex=True).astype(int)
    
    # Handle index reset if needed
    reset = False
    if id_col == _df.index.name:
        _df = _df.reset_index(drop=False)
        logging.info("Id col is the index")
        reset = True
    
    # Optionally group by id_col and take max
    if group_by_id:
        pattern_cols = list(pattern_dict.keys())
        
        # Simple approach: aggregate all needed columns at once
        agg_dict = {search_col: 'first'}  # Keep first occurrence of search column
        agg_dict.update({col: 'max' for col in pattern_cols})  # Max for pattern columns
        
        # Get all other columns that aren't pattern columns or search/id columns
        other_cols = [col for col in _df.columns if col not in pattern_cols + [id_col, search_col]]
        if other_cols:
            agg_dict.update({col: 'first' for col in other_cols})  # Keep first for other columns
        
        _df_result = _df.groupby(id_col).agg(agg_dict).reset_index()
        
        if reset:
            _df_result = _df_result.set_index(id_col)
        
        # Verify new columns by printing first few rows
        print(f"Pattern match result sample:\n{_df_result.head()}")
        return _df_result
    else:
        # Return the dataframe without grouping
        if reset:
            _df = _df.set_index(id_col)

        print(f"Pattern match result sample:\n{_df.head()}")
        return _df

# Create a dictionary that maps strings to functions
function_dict = {
    "datetime_keepfirst": datetime_keepfirst,
    "keepfirst": keepfirst,
    "datetime": datetime,
    "smokerprocess": smokerprocess,
    "diagnoseprocess_simple": diagnoseprocess_simple,
    "ace": ace,
    "beta": beta,
    "diff": diff,
    "map": dg_map,
    "fillna": fillna,
    "dropna": dropna,
    "chain": chain,
    "classify": classify,
    "pattern_match": pattern_match,
}
# Functions that don't use out_col parameter - derive from function signatures
no_output_col_funcs = []
for func_name, func in function_dict.items():
    sig = inspect.signature(func)
    # Check if the function uses out_col in its implementation
    source = inspect.getsource(func)
    if 'out_col' not in source or 'kwargs.get(\'out_col\'' not in source:
        no_output_col_funcs.append(func_name)

def function_to_execute(config_parameter):
    if config_parameter not in function_dict:
        logger.info(f"Function {config_parameter} not found")
        # Print the available options
        logger.info(f"Available options: {function_dict.keys()}")
        raise ValueError(f"Function {config_parameter} not found")
    return function_dict.get(config_parameter)

def transform_aggregations(_df, aggs, cols_for_aggs):
    for col in cols_for_aggs:
        func = function_to_execute(aggs[col]['func'])
        _kwargs = aggs[col]['kwargs']
        _df = func(_df, **_kwargs)
    return _df

def merged_transforms(_df, _tfs):
    
    for col in _tfs:
        func_name = _tfs[col]['func']
        func = function_to_execute(func_name)
        _kwargs = _tfs[col]['kwargs'].copy()  # Copy to avoid modifying original
        
        # Only add out_col for functions that actually use it
        if func_name not in no_output_col_funcs:
            _kwargs['out_col'] = col
        
        _df = func(_df, **_kwargs)
        
        # Check if dataframe became empty
        if len(_df) == 0:
            raise ValueError(f"DataFrame became empty after {func_name} transform! Transform config: {_tfs[col]}")
            
    return _df

