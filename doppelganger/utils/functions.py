import dask.dataframe as dd
import pandas as pd
import logging

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
    _df = _df.fillna(value={search_col: 0})
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
    if id_col == _df.index.name:
        _df = _df.reset_index(drop=False)
        logging.info("Id col is the index")
        reset = True
    _df = _df[[smoker, id_col]].groupby(id_col).max().reset_index()
    if reset:
        _df = _df.set_index(id_col)
    return _df

def ace(_df, **kwargs):
    meds = kwargs.get('meds', None)
    out_col = kwargs.get('out_col', None)
    id_col = kwargs.get('id_col', None)

    # ACE inhibitors
    ace_atc = [f"C09AA{n:02d}" for n in range(1,17)]
    ace_atc.extend([f"C09BA{n:02d}" for n in range(1,16)])
    ace_atc.extend([f"C09BB{n:02d}" for n in range(2,13)])
    ace_atc.extend([f"C09BX{n:02d}" for n in range(1,6)])
    logging.info("Looking for ACEi with codes:", ace_atc)

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
        _df = _df.set_index(id_col)

    return _df_ace

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

# Create a dictionary that maps strings to functions
function_dict = {
    "datetime_keepfirst": datetime_keepfirst,
    "keepfirst": keepfirst,
    "datetime": datetime,
    "smokerprocess": smokerprocess,
    "diagnoseprocess_simple": diagnoseprocess_simple,
    "ace": ace,
    "diff": diff,
    "map": dg_map,
    "fillna": fillna,
    "dropna": dropna
}

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
        func = function_to_execute(_tfs[col]['func'])
        _kwargs = _tfs[col]['kwargs']
        # Make col a kwarg
        _kwargs['out_col'] = col
        print(_kwargs)
        _df = func(_df, **_kwargs)
    return _df