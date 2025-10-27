from dpplgngr.utils.utils_etl import file_size, return_subset, vals_to_cols, to_datetime, analyze_dataframe, safe_merge
from dpplgngr.utils.functions import transform_aggregations, merged_transforms
from dpplgngr.etl.convert_to_parquet import convert
import dask.dataframe as dd
from dateutil import parser
from dask_ml.impute import SimpleImputer
from dask_ml.preprocessing import StandardScaler
from joblib import load, dump
import pandas as pd
import numpy as np
import logging
import json
import os
import ast

# Try to import snowpark
try:
    from snowflake.snowpark import Session
    from snowflake.snowpark.functions import col
    _snowpark_available = True
except ImportError:
    _snowpark_available = False
    Session = None

# Try to import luigi, fallback to replacement if not available
try:
    import luigi
    _using_luigi_replacement = False
except ImportError:
    from dpplgngr.utils.luigi_replacement import Task, Parameter, IntParameter, LocalTarget, build as luigi_build
    # Create a mock luigi module for compatibility
    class MockLuigi:
        Task = Task
        Parameter = Parameter
        IntParameter = IntParameter
        LocalTarget = LocalTarget
        
        @staticmethod
        def build(*args, **kwargs):
            return luigi_build(*args, **kwargs)
    
    luigi = MockLuigi()
    _using_luigi_replacement = True

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('luigi-interface')

# meta data
__author__ = 'SB'
__date__ = '2023-09-25'

def create_snowflake_session(connection_params):
    """
    Create a Snowflake session with the given connection parameters.
    
    Args:
        connection_params (dict): Dictionary containing Snowflake connection parameters
            - account: Snowflake account identifier
            - user: Username
            - password: Password
            - role: Role to use
            - warehouse: Warehouse to use
            - database: Database to use
            - schema: Default schema (optional)
    
    Returns:
        snowflake.snowpark.Session: Configured Snowflake session
    """
    if not _snowpark_available:
        raise ImportError("snowflake-snowpark-python is required for Snowflake integration")
    
    return Session.builder.configs(connection_params).create()

class ConvertLargeFiles(luigi.Task):
    lu_output_path = luigi.Parameter(default='converted.json')
    lu_size_limit = luigi.IntParameter(default=500) # Limit in MB
    etl_config = luigi.Parameter(default="config/etl.json")

    def output(self):
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        name = input_json.get('name', None)
        outdir = input_json.get('preprocessing', None)
        if not outdir:
            name = input_json.get('name', None)
            outdir = f"data/{name}/preprocessing"
        return luigi.LocalTarget(os.path.join(outdir, self.lu_output_path))
    
    def run(self):
        # Load input json
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        name = input_json.get('name', None)
        outdir = input_json.get('preprocessing', None)
        if not outdir:
            outdir = f"data/{name}/preprocessing"

        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        
        filenames = input_json.keys()
        # Keep only names with [.csv and .csv.gz] extensions
        filenames = [f for f in filenames if any([f.endswith(ext) for ext in ['.csv', '.csv.gz']])]
        
        path = input_json.get('absolute_path', None)
        filenames = [os.path.join(path, f) for f in filenames]
        logging.info(f"Filenames: {filenames}")
        if filenames:
            assert all([os.path.exists(f) for f in filenames]), "Some files do not exist"
        logger.info(f"*** Found {len(filenames)} files to convert ***")

        # Check if larger than size limit
        filenames = [f for f in filenames if file_size(f) > self.lu_size_limit]

        # Convert the remaining files
        filenames_out = [f.replace('.csv', '.parquet') for f in filenames]
        filenames_out = [f.split("/")[-1] for f in filenames_out]
        filenames_out = [os.path.join(outdir, f) for f in filenames_out]

        for i, o in zip(filenames, filenames_out):
            convert(i, o)

        # Write output mapping
        with self.output().open('w') as f:
            json.dump(dict(zip(filenames, filenames_out)), f)


class PreProcess(luigi.Task):
    lu_output_path = luigi.Parameter(default='preprocessed.parquet')
    etl_config = luigi.Parameter(default="config/etl.json")
    snowpark_session = luigi.Parameter(default=None)

    def requires(self):
        # Only require ConvertLargeFiles if not using Snowflake
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        source = input_json.get('SOURCE', 'FILE')
        
        if source == 'SNOWFLAKE':
            return []
        else:
            return ConvertLargeFiles(etl_config=self.etl_config)
    
    def output(self):
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        outdir = input_json.get('preprocessing', None)
        if not outdir:
            name = input_json.get('name', None)
            outdir = f"data/{name}/preprocessing"
        logging.info(f"Output directory: {outdir}")
        logging.info(f"Output path: {self.lu_output_path}")
        return luigi.LocalTarget(os.path.join(outdir, self.lu_output_path))
    
    def check_snowflake_table_exists(self, session, schema, table_name):
        """Check if a table exists in Snowflake schema"""
        try:
            # Try to describe the table - if it doesn't exist, this will raise an exception
            session.table(f"{schema}.{table_name}").schema
            return True
        except Exception:
            return False
    
    def load_individual_preprocessed_table(self, session, schema, table_name):
        """Load an individual preprocessed table from Snowflake"""
        if not _snowpark_available:
            raise ImportError("snowflake-snowpark-python is required for Snowflake integration")
        
        # Build the table reference
        table_ref = f"{schema}.{table_name}"
        
        # Load the table
        df_snow = session.table(table_ref)
        
        # Convert to pandas DataFrame for compatibility with existing processing
        df_pandas = df_snow.to_pandas()
        
        # The index was reset when saved, so we need to identify the pseudo_id column
        # It could be named 'index' (if original index was unnamed) or have the original name
        logging.info(f"Loaded table columns: {df_pandas.columns.tolist()}")
        
        # Convert to Dask DataFrame
        df_dask = dd.from_pandas(df_pandas, npartitions=3)
        
        return df_dask

    def write_individual_preprocessed_table(self, session, df, schema, table_name):
        """Write an individual preprocessed table to Snowflake"""
        if not _snowpark_available:
            raise ImportError("snowflake-snowpark-python is required for Snowflake integration")
        
        # Convert to pandas for Snowflake write
        df_pandas = df.compute()
        
        # Reset index to ensure it's a column, but preserve the index name for later retrieval
        index_name = df_pandas.index.name
        df_pandas = df_pandas.reset_index(drop=False)
        
        # If the index was unnamed but we know it should be the first column, 
        # rename it to ensure we can identify it later
        if index_name is None and len(df_pandas.columns) > 0:
            # The reset_index will create a column named 'index' if the index was unnamed
            if 'index' in df_pandas.columns:
                # This is likely our pseudo_id column
                pass  # Keep it as 'index' - we'll handle this in load
        
        # Write to Snowflake
        snow_df = session.create_dataframe(df_pandas)
        
        # Write to table (overwrite mode)
        snow_df.write.mode("overwrite").save_as_table(f"{schema}.{table_name}")
        logging.info(f"Individual preprocessed table written to Snowflake: {schema}.{table_name}")
        logging.info(f"Table columns: {df_pandas.columns.tolist()}")
        logging.info(f"Original index name: {index_name}")
    
    def load_snowflake_data(self, session, schema, table_name, columns):
        """Load data from Snowflake table"""
        if not _snowpark_available:
            raise ImportError("snowflake-snowpark-python is required for Snowflake integration")
        
        # Build the table reference
        table_ref = f"{schema}.{table_name}"
        
        # Load the table
        df_snow = session.table(table_ref)
        
        # Select only the required columns if specified
        if isinstance(columns, list) and len(columns) > 0:
            # Handle simple column selection
            if all(isinstance(c, str) for c in columns):
                # Get existing columns in the table
                existing_columns = [field.name for field in df_snow.schema.fields]
                # Filter to only include columns that exist
                available_columns = [c for c in columns if c in existing_columns]
                if available_columns:
                    df_snow = df_snow.select([col(c) for c in available_columns])
                else:
                    logger.warning(f"None of the specified columns {columns} exist in table {table_ref}")
        
        # Convert to pandas DataFrame for compatibility with existing processing
        df_pandas = df_snow.to_pandas()
        
        # Convert to Dask DataFrame
        df_dask = dd.from_pandas(df_pandas, npartitions=3)
        
        return df_dask

    def get_data_source_config(self, input_json):
        """Get the appropriate data source configuration based on SOURCE setting"""
        source = input_json.get('SOURCE', 'FILE')
        
        if source == 'SNOWFLAKE':
            # Use the new Snowflake table configuration
            data_configs = {}
            snowflake_config = input_json.get('snowflake_config', {})
            
            for key, value in input_json.items():
                if key.startswith('Datatools4heart_') and isinstance(value, dict):
                    # This is a Snowflake table configuration
                    data_configs[key] = {
                        'table_name': value['table_name'],
                        'columns': value['columns']
                    }
            return data_configs, snowflake_config
        else:
            # Extract file-based configurations from the original structure
            data_configs = {}
            for key, value in input_json.items():
                if key.endswith('.feather') or key.endswith('.csv') or key.endswith('.parquet'):
                    data_configs[key] = value
            return data_configs, None
    
    def apply_transformations(self, s_df, _config, total_cols, transform_type="PreTransforms"):
        # See if any column names specify aggregations
        aggs = _config.get(transform_type, None)
        if not aggs:
            logging.info("No aggregations or transforms specified")
        
        # Separate cols into dictionaries and non-dictionaries
        # Check if there's any dictionaries in cols
        d_entry = any([isinstance(c, dict) for c in total_cols])

        l_cols = []
        if not d_entry:
            # If no dictionaries, just use cols directly
            l_cols = total_cols
        else:
            # Find the key of the innermost nested dictionary
            d_cols = total_cols[1]
            for k, v in d_cols.items():
                l_cols.extend(list(v.values()))
            # Add the outer key too
            l_cols.append(total_cols[0]) 

        # Find intersection of cols and aggs keys
        cols_for_aggregations = list(set(l_cols).intersection(aggs.keys()))

        logging.info("Aggregation columns: %s", cols_for_aggregations)
        # Apply aggregations
        if cols_for_aggregations:
            s_df = transform_aggregations(s_df, aggs, cols_for_aggregations)
        return s_df

    def run(self):
        # Load input json
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)

        source = input_json.get('SOURCE', 'FILE')
        data_configs, snowflake_config = self.get_data_source_config(input_json)
        
        # Open empty dask dataframe
        df_pp = None

        if source == 'SNOWFLAKE':
            if not self.snowpark_session:
                raise ValueError("Snowpark session is required when SOURCE is SNOWFLAKE")
            
            session = self.snowpark_session
            input_schema = snowflake_config.get('input_schema', 'RAW_DATA')
            output_schema = snowflake_config.get('output_schema', 'PROCESSED_DATA')
            
            logging.info(f"Processing Snowflake tables from schema: {input_schema}")
            
            # Process each Snowflake table
            for table_key, table_config in data_configs.items():
                table_name = table_config['table_name']
                vals = table_config['columns']
                
                current_name = table_key.replace('Datatools4heart_', '')
                individual_table_name = f"{current_name}_preprocessed"
                
                # Check if we have this data already preprocessed in Snowflake
                if self.check_snowflake_table_exists(session, output_schema, individual_table_name):
                    logging.info(f"*** Loading existing preprocessed table: {output_schema}.{individual_table_name} ***")
                    df = self.load_individual_preprocessed_table(session, output_schema, individual_table_name)
                    
                    # Ensure the pseudo_id (index) is properly set for consistent merging
                    index = vals[0]  # First entry is always the pseudo_id
                    if index in df.columns and df.index.name != index:
                        df = df.set_index(index)
                    elif df.index.name is None and index in df.columns:
                        df = df.set_index(index)
                    
                    # Log merge information for debugging
                    logging.info(f"Before merge (existing table) - df index name: {df.index.name}, df_pp index name: {df_pp.index.name if df_pp is not None else 'None'}")
                    logging.info(f"df shape: {df.shape}, df_pp shape: {df_pp.shape if df_pp is not None else 'None'}")
                    
                    if len(df) == 0:
                        raise ValueError(f"Loaded preprocessed table {output_schema}.{individual_table_name} is empty")
                    
                    df_pp = safe_merge(df, df_pp)

                    if len(df_pp) == 0:
                        raise ValueError("Merging Snowflake tables resulted in empty dataframe - likely index mismatch")
                    continue

                logging.info(f"*** Processing Snowflake table {table_name} ***")
                
                # Load data from Snowflake
                df = self.load_snowflake_data(session, input_schema, table_name, vals)
                
                # Set index if specified
                # index = None
                # if isinstance(vals, list) and len(vals) > 0:
                #     index = cols[0]  # First column as index
                #     if index in df.columns:
                #         df = df.set_index(index)

                index = vals[0]
                cols = vals[1:]

                # Apply transformations
                df = self.apply_transformations(df, input_json, cols, transform_type="InitTransforms")

                # are any items in the list dictionaries?
                col_extract = not any([isinstance(v, dict) for v in vals])
                logger.info(f"*** Column extraction: {col_extract} ***")
                if col_extract:
                    df = return_subset(df, cols, index_col=index)
                else:
                    """ Assume form of cols is:
                    ["col_name", {"val_name": {"type1": name1, "type2": name2}}, ["optional_extra_col"]]
                    """
                    
                    col_name = cols[0]
                    val_name = list(cols[1].keys())[0]
                    col_map = cols[1][val_name]
                    extra_cols = None
                    if len(cols) == 3:
                        extra_cols = cols[2]
                    if len(cols) > 3:
                        raise ValueError("Too many columns specified")
                    df = vals_to_cols(df, index_col=index, code_col=col_name, value_col=val_name,
                    code_map=col_map, extra_cols=extra_cols)
                
                # Ensure the pseudo_id (index) is properly set for consistent merging
                if index in df.columns and df.index.name != index:
                    df = df.set_index(index)
                elif df.index.name is None and index in df.columns:
                    df = df.set_index(index)
                
                assert df.index.unique, "Index is not unique"
                
                df = self.apply_transformations(df, input_json, cols, transform_type="PreTransforms")
                
                # Write individual preprocessed table to Snowflake for future use
                self.write_individual_preprocessed_table(session, df, output_schema, individual_table_name)
                
                # Log merge information for debugging
                logging.info(f"Before merge - df index name: {df.index.name}, df_pp index name: {df_pp.index.name if df_pp is not None else 'None'}")
                logging.info(f"df shape: {df.shape}, df_pp shape: {df_pp.shape if df_pp is not None else 'None'}")
                
                df_pp = safe_merge(df, df_pp)
                # Throw error if merge fails
                if len(df_pp) == 0:
                    raise ValueError("Merging Snowflake tables resulted in empty dataframe - likely index mismatch")
            
            # Debug: Check final merged result for Snowflake processing
            logging.info(f"After processing all Snowflake tables:")
            logging.info(f"df_pp shape: {df_pp.shape if df_pp is not None else 'None'}")
            logging.info(f"df_pp index name: {df_pp.index.name if df_pp is not None else 'None'}")
            logging.info(f"df_pp columns: {df_pp.columns.tolist() if df_pp is not None else 'None'}")
                
        else:
            # Original file-based processing
            filenames = list(data_configs.keys())
            # Keep only names with normal extensions
            filenames = [f for f in filenames if any([f.endswith(ext) for ext in ['.csv', '.csv.gz', '.parquet', '.feather']])]
            path = input_json['absolute_path']
            filenames = [os.path.join(path, f) for f in filenames]
            
            # Load converted json if available
            converted_json = {}
            if hasattr(self, 'input') and self.input():
                with self.input().open('r') as f:
                    converted_json = json.load(f)
            
            filenames_to_convert = converted_json.keys()
            # Replace with converted filenames
            new_filenames = [converted_json[f] if f in filenames_to_convert else f for f in filenames]

            logging.info(f"Filenames: {filenames}")
            # Create the requested columns from each filename
            for o, f in zip(filenames, new_filenames):
                # First check if we have this file checkpointed
                current_name = f.split("/")[-1].split(".")[0]
                saved_loc = f"{input_json['preprocessing']}/{current_name}_preprocessed.parquet"
                # If file exists then load it
                if os.path.exists(saved_loc):
                    logging.info(f"*** Loading {saved_loc} ***")
                    df = dd.read_parquet(saved_loc, npartitions=3)
                    df_pp = safe_merge(df, df_pp)
                    continue

                logging.info(f"*** Processing {f} ***")
                vals = data_configs[o.split("/")[-1]]
                index = vals[0]
                cols = vals[1:]

                logging.info(f"Index: {index}")
                logging.info(f"Columns: {cols}")

                if '.parquet' in f:
                    df = dd.read_parquet(f)
                elif '.feather' in f:
                    df = dd.from_pandas(pd.read_feather(f), npartitions=3)
                else:
                    df = dd.read_csv(f, blocksize='64MB')  # Fixed blocksize

                # Apply initial transformations
                df = self.apply_transformations(df, input_json, cols, transform_type="InitTransforms")

                # are any items in the list dictionaries?
                col_extract = not any([isinstance(v, dict) for v in vals])
                logger.info(f"*** Column extraction: {col_extract} ***")
                if col_extract:
                    df = return_subset(df, cols, index_col=index)
                else:
                    """ Assume form of cols is:
                    ["col_name", {"val_name": {"type1": name1, "type2": name2}}, ["optional_extra_col"]]
                    """
                    
                    col_name = cols[0]
                    val_name = list(cols[1].keys())[0]
                    col_map = cols[1][val_name]
                    extra_cols = None
                    if len(cols) == 3:
                        extra_cols = cols[2]
                    if len(cols) > 3:
                        raise ValueError("Too many columns specified")
                    df = vals_to_cols(df, index_col=index, code_col=col_name, value_col=val_name,
                    code_map=col_map, extra_cols=extra_cols)
                
                assert df.index.unique, "Index is not unique"

                df = self.apply_transformations(df, input_json, cols, transform_type="PreTransforms")

                # Checkpoint pre-concat only if not using SNOWFLAKE
                if input_json.get("SOURCE", "FILE") != "SNOWFLAKE":
                    df.to_parquet(saved_loc)
                df_pp = safe_merge(df, df_pp)

                if len(df_pp) == 0:
                    raise ValueError("Merging files resulted in empty dataframe - likely index mismatch")
        
        # Merged transforms
        if len(df_pp) == 0:
            raise ValueError("Unexpected empty dataframe before merged transforms")
        end_transforms = input_json.get('MergedTransforms', None)
        if end_transforms:
            df_pp = merged_transforms(df_pp, end_transforms)
        if len(df_pp) == 0:
            raise ValueError("Merged transforms resulted in empty dataframe")

        # Debug: Check dataframe state before final column selection
        logging.info(f"Before final column selection:")
        logging.info(f"df_pp shape: {df_pp.shape if df_pp is not None else 'None'}")
        logging.info(f"df_pp index name: {df_pp.index.name if df_pp is not None else 'None'}")
        logging.info(f"Available columns: {df_pp.columns.tolist() if df_pp is not None else 'None'}")
        logging.info(f"Requested final_cols: {input_json['final_cols']}")
        
        if df_pp is not None:
            missing_cols = [col for col in input_json['final_cols'] if col not in df_pp.columns]
            if missing_cols:
                logging.warning(f"Missing columns in final selection: {missing_cols}")
                # Try to include the index in columns if it's one of the missing columns
                if df_pp.index.name in missing_cols:
                    df_pp = df_pp.reset_index()
                    logging.info(f"Reset index, new columns: {df_pp.columns.tolist()}")

        # Reduce to final specified columns
        df_pp = df_pp[input_json["final_cols"]]
        logging.info(df_pp.head(20))
        logging.info(f"Final shape: {df_pp.shape}")

        # Handle output based on source
        if source == 'SNOWFLAKE' and self.snowpark_session:
            # Write directly to Snowflake without local backup
            snowflake_config = input_json.get('snowflake_config', {})
            output_schema = snowflake_config.get('output_schema', 'PROCESSED_DATA')
            output_table = f"{input_json['name']}_preprocessed"
            
            # Convert to pandas for Snowflake write
            df_pandas = df_pp.compute()

            # Reset index to ensure it's a column
            df_pandas = df_pandas.reset_index(drop=False)
            
            # Write to Snowflake
            session = self.snowpark_session
            snow_df = session.create_dataframe(df_pandas)
            
            # Write to table (overwrite mode)
            snow_df.write.mode("overwrite").save_as_table(f"{output_schema}.{output_table}")
            logging.info(f"Preprocessed data written to Snowflake table: {output_schema}.{output_table}")
            
            # Create a dummy local target for Luigi compatibility but don't write actual data
            #with open(self.output().path, 'w') as f:
            #os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
            #    f.write(f"Data written to Snowflake: {output_schema}.{output_table}")
        else:
            # Save to local parquet file for file-based processing
            df_pp.to_parquet(self.output().path)
        
        logging.info("Success")


# Task to take preprocessed data and for columns of the format: [(measurement1, date1), (measurement2, date2) ...]
# find the first measurement after a configurable date, e. g. OpnameDatum
class TuplesProcess(luigi.Task):
    lu_output_path = luigi.Parameter(default='preprocessed_tupleprocess.parquet')
    etl_config = luigi.Parameter(default="config/etl.json")
    snowpark_session = luigi.Parameter(default=None)

    def requires(self):
        # Only require PreProcess if not using Snowflake independently
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        source = input_json.get('SOURCE', 'FILE')
        
        if source == 'SNOWFLAKE' and self.snowpark_session:
            return []  # Run independently for Snowflake
        else:
            return PreProcess(etl_config=self.etl_config)
    
    def output(self):
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        outdir = input_json.get('preprocessing', None)
        if not outdir:
            name = input_json.get('name', None)
            outdir = f"data/{name}/preprocessing"
        return luigi.LocalTarget(os.path.join(outdir, self.lu_output_path))
    
    def load_snowflake_preprocessed_data(self, session, input_json):
        """Load preprocessed data from Snowflake table"""
        if not _snowpark_available:
            raise ImportError("snowflake-snowpark-python is required for Snowflake integration")
        
        snowflake_config = input_json.get('snowflake_config', {})
        output_schema = snowflake_config.get('output_schema', 'PROCESSED_DATA')
        input_table = f"{input_json['name']}_preprocessed"
        
        # Build the table reference
        table_ref = f"{output_schema}.{input_table}"
        
        # Load the table
        df_snow = session.table(table_ref)
        
        # Convert to pandas DataFrame for compatibility with existing processing
        df_pandas = df_snow.to_pandas()
        
        # Convert to Dask DataFrame
        df_dask = dd.from_pandas(df_pandas, npartitions=3)
        
        return df_dask
    
    def run(self):
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)

        source = input_json.get('SOURCE', 'FILE')
        
        if source == 'SNOWFLAKE' and self.snowpark_session:
            # Load data from Snowflake
            ddf = self.load_snowflake_preprocessed_data(self.snowpark_session, input_json)
        else:
            # Load from local parquet file
            ddf = dd.read_parquet(self.input().path)

        # Get reference date column from config
        ref_date_col = input_json.get('ref_date', 'OpnameDatum')
        # Tuple columns to process
        tuple_cols_after = input_json.get('tuple_vals_after', None)
        tuple_cols_anybefore = input_json.get('tuple_vals_anybefore', None)
        if not tuple_cols_anybefore and not tuple_cols_after:
            logging.info("No tuple columns specified, skipping")
            if source == 'SNOWFLAKE' and self.snowpark_session:
                # For Snowflake, just copy the input table to output table
                snowflake_config = input_json.get('snowflake_config', {})
                output_schema = snowflake_config.get('output_schema', 'PROCESSED_DATA')
                output_table = f"{input_json['name']}_tupleprocessed"
                
                # Convert to pandas for Snowflake write
                df_pandas = ddf.compute()
                
                # Write to Snowflake
                session = self.snowpark_session
                snow_df = session.create_dataframe(df_pandas)
                
                # Write to table (overwrite mode)
                snow_df.write.mode("overwrite").save_as_table(f"{output_schema}.{output_table}")
                logging.info(f"Tuple processed data written to Snowflake table: {output_schema}.{output_table}")
                
                # Create a dummy local target for Luigi compatibility
                os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
                with open(self.output().path, 'w') as f:
                    f.write(f"Data written to Snowflake: {output_schema}.{output_table}")
            else:
                ddf.to_parquet(self.output().path)
            return
        if not tuple_cols_anybefore:
            tuple_cols_anybefore = []
        if not tuple_cols_after:
            tuple_cols_after = []

        def process_tuple(row, _col_tuple, ref_date_col, after=True):
            # First find the reference date
            ref_date = row[ref_date_col]
            # Then get the tuple column
            r_col_tuple = row[_col_tuple]

            # Handle <NA>
            if pd.isnull(r_col_tuple):
                return np.nan
            
            # TODO: Understand why sometimes r_col_tuple is a float with no date
            # In that case return the float if it's a number
            if isinstance(r_col_tuple, float):
                return r_col_tuple
            
            # Sort format
            logger.info(f"_col_tuple (pre-literal eval): {r_col_tuple}")
            r_col_tuple = ast.literal_eval(r_col_tuple)
            # Do literal eval on entries in r_col_tuple
            r_col_tuple = [ast.literal_eval(str(t)) for t in r_col_tuple if "(nan" not in str(t).lower()]

            if len(r_col_tuple) == 0:
                return np.nan

            measurements, dates = zip(*r_col_tuple)
            logger.info(f"Measurements: {measurements}, Dates: {dates}, Ref date: {ref_date}")
            try:
                # Sort the measurements by date
                measurements, dates = zip(*r_col_tuple)
                logger.info(f"Measurements: {measurements}, Dates: {dates}, Ref date: {ref_date}")
            except ValueError as e:
                logger.info(f"Error unpacking _col_tuple: {r_col_tuple}")
                logger.info(f"Error: {e}")
                return np.nan
            
            # Parse dates using dateutil parser for automatic format detection
            parsed_dates = []
            for date in dates:
                try:
                    parsed_dates.append(parser.parse(str(date)))
                except (ValueError, TypeError):
                    parsed_dates.append(pd.NaT)
            dates = pd.Series(parsed_dates)
            
            try:
                ref_date_parsed = parser.parse(str(ref_date))
            except (ValueError, TypeError):
                ref_date_parsed = pd.NaT
            
            # Handle case where no valid dates
            if all(pd.isnull(dates)):
                return np.nan
            
            sorted_indices = np.argsort(dates)
            measurements = np.array(measurements)[sorted_indices]
            # If measurements is an array of strings make the values all 1.0
            if all(isinstance(m, str) for m in measurements):
                measurements = np.array([1.0 for _ in measurements])
            dates = np.array(dates)[sorted_indices]
            # Make dates datetime rather than datetime64
            dates = [to_datetime(d) for d in dates]

            # If the reference date is NaN/NaT, return first valid measurement
            if pd.isnull(ref_date_parsed):
                return measurements[0] if len(measurements) > 0 else np.nan
            
            if after:
                # Find the first measurement after the reference date
                for meas, date in zip(measurements, dates):
                    if pd.isnull(date):
                        continue
                    try:
                        if date >= ref_date_parsed:
                            return meas
                    except TypeError:
                        print(f"TypeError comparing dates: {date} and {ref_date_parsed}")
                        print(f"Types: {type(date)} and {type(ref_date_parsed)}")
                return np.nan  # If none found, return NaN
            else:
                # Find the first measurement before the reference date
                for meas, date in zip(measurements, dates):
                    if pd.isnull(date):
                        continue
                    if date < ref_date_parsed:
                        return meas
            return np.nan  # If none found, return NaN

        for t in tuple_cols_after:
            ddf[t + '_first_after'] = ddf.apply(process_tuple, axis=1, args=(t, ref_date_col, True), meta=(t + '_first_after', 'float32'))
        for t in tuple_cols_anybefore:
            ddf[t + '_any_before'] = ddf.apply(process_tuple, axis=1, args=(t, ref_date_col, False), meta=(t + '_any_before', 'float32'))

        # Make analysis of dataframe (only for non-Snowflake sources)
        if source != 'SNOWFLAKE':
            analyze_dataframe(ddf, prefix="TUPLEPROCESS")

        # Handle output based on source
        if source == 'SNOWFLAKE' and self.snowpark_session:
            # Write to Snowflake
            snowflake_config = input_json.get('snowflake_config', {})
            output_schema = snowflake_config.get('output_schema', 'PROCESSED_DATA')
            output_table = f"{input_json['name']}_tupleprocessed"
            
            # Convert to pandas for Snowflake write
            df_pandas = ddf.compute()
            
            # Write to Snowflake
            session = self.snowpark_session
            snow_df = session.create_dataframe(df_pandas)
            
            # Write to table (overwrite mode)
            snow_df.write.mode("overwrite").save_as_table(f"{output_schema}.{output_table}")
            logging.info(f"Tuple processed data written to Snowflake table: {output_schema}.{output_table}")
        else:
            # Save to local parquet file for file-based processing
            ddf.to_parquet(self.output().path)
        
        logging.info("Success")



class ImputeScaleCategorize(luigi.Task):
    lu_output_path = luigi.Parameter(default='preprocessed_imputed.parquet')
    etl_config = luigi.Parameter(default="config/etl.json")
    snowpark_session = luigi.Parameter(default=None)

    def requires(self):
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        
        # Check if tuple processing is needed
        tuple_cols_after = input_json.get('tuple_vals_after', None)
        tuple_cols_anybefore = input_json.get('tuple_vals_anybefore', None)
        needs_tuple_processing = (tuple_cols_after and len(tuple_cols_after) > 0) or (tuple_cols_anybefore and len(tuple_cols_anybefore) > 0)
        
        source = input_json.get('SOURCE', 'FILE')
        
        if source == 'SNOWFLAKE' and self.snowpark_session:
            if needs_tuple_processing:
                return []  # Run independently, will load from TuplesProcess output
            else:
                return []  # Run independently, will load from PreProcess output
        else:
            if needs_tuple_processing:
                return TuplesProcess(etl_config=self.etl_config)
            else:
                return PreProcess(etl_config=self.etl_config, snowpark_session=self.snowpark_session)
    
    def load_snowflake_data(self, session, input_json):
        """Load data from appropriate Snowflake table based on configuration"""
        if not _snowpark_available:
            raise ImportError("snowflake-snowpark-python is required for Snowflake integration")
        
        snowflake_config = input_json.get('snowflake_config', {})
        output_schema = snowflake_config.get('output_schema', 'PROCESSED_DATA')
        
        # Check if tuple processing is needed
        tuple_cols_after = input_json.get('tuple_vals_after', None)
        tuple_cols_anybefore = input_json.get('tuple_vals_anybefore', None)
        needs_tuple_processing = (tuple_cols_after and len(tuple_cols_after) > 0) or (tuple_cols_anybefore and len(tuple_cols_anybefore) > 0)
        
        if needs_tuple_processing:
            # Load from tuple processed table
            input_table = f"{input_json['name']}_tupleprocessed"
        else:
            # Load from preprocessed table
            input_table = f"{input_json['name']}_preprocessed"
        
        # Build the table reference
        table_ref = f"{output_schema}.{input_table}"
        
        # Load the table
        df_snow = session.table(table_ref)
        
        # Convert to pandas DataFrame for compatibility with existing processing
        df_pandas = df_snow.to_pandas()
        
        # Convert to Dask DataFrame
        df_dask = dd.from_pandas(df_pandas, npartitions=3)
        
        return df_dask
    
    def output(self):
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        outdir = input_json.get('preprocessing', None)
        if not outdir:
            name = input_json.get('name', None)
            outdir = f"data/{name}/preprocessing"
        return luigi.LocalTarget(os.path.join(outdir, self.lu_output_path))
    
    def run(self):
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        name = input_json['name']
        source = input_json.get('SOURCE', 'FILE')

        # TODO: Fix for preprocessing locs
        assert all(x in list(input_json.keys()) for x in ["scaler", "imputer"]), "Scaler and imputer not specified"
        scaler_path = f"data/{name}/preprocessing/{input_json['scaler']}"
        imputer_path = f"data/{name}/preprocessing/{input_json['imputer']}"
        load_sc, load_imp = False, False
        if os.path.exists(scaler_path):
            load_sc = True
            scaler = load(scaler_path)
        if os.path.exists(imputer_path):
            load_imp = True
            imputer = load(imputer_path)
        
        # Load data based on source
        if source == 'SNOWFLAKE' and self.snowpark_session:
            # Load data from Snowflake
            ddf = self.load_snowflake_data(self.snowpark_session, input_json)
        else:
            # Load from local parquet file
            ddf = dd.read_parquet(self.input().path)

        # Make cells with underscores nan
        to_nan = ["", "__", "_", "___"]
        ddf = ddf.mask(ddf.isin(to_nan), other=np.nan)

        # Find categorical columns
        total_cols = ddf.columns
        # Remove categorical columns from list
        categories = input_json['categories']
        num_cols = [c for c in total_cols if c not in categories.keys()]
        for n in num_cols:
            ddf[n] = ddf[n].astype('float32')

        # Scale numerical columns
        if not load_sc:
            scaler = StandardScaler()
            scaler.fit(ddf[num_cols])
        ddf[num_cols] = scaler.transform(ddf[num_cols])

        # Map categorical columns to binary if only 2
        for c in categories:
            cats = categories[c]
            if len(cats) == 2:
                map_c = {cats[0]: 0, cats[1]: 1}
                ddf[c] = ddf[c].map(map_c)
            else:
                # One hot encode categorical columns
                ddf = dd.get_dummies(ddf, columns=[c])
        
        # Impute missing values
        if not load_imp:
            imputer = SimpleImputer(strategy='median')
            imputer.fit(ddf)
        ddf = imputer.transform(ddf)

        # Save scaler/imputer to h5 file
        if not load_sc:
            dump(scaler, scaler_path)
        if not load_imp:
            dump(imputer, imputer_path)
        
        # Handle output based on source
        if source == 'SNOWFLAKE' and self.snowpark_session:
            # Write only to Snowflake, no local backup
            snowflake_config = input_json.get('snowflake_config', {})
            output_schema = snowflake_config.get('output_schema', 'PROCESSED_DATA')
            output_table = f"{name}_preprocessed_imputed"
            
            # Convert to pandas for Snowflake write
            df_pandas = ddf.compute()
            
            # Write to Snowflake
            session = self.snowpark_session
            snow_df = session.create_dataframe(df_pandas)
            
            # Write to table (overwrite mode)
            snow_df.write.mode("overwrite").save_as_table(f"{output_schema}.{output_table}")
            logging.info(f"Final processed data written to Snowflake table: {output_schema}.{output_table}")
            
            # Create a dummy local target for Luigi compatibility but don't write actual data
            os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
            with open(self.output().path, 'w') as f:
                f.write(f"Data written to Snowflake: {output_schema}.{output_table}")
        else:
            # Save to local parquet file for file-based processing
            ddf.to_parquet(self.output().path)
        
        logging.info("Success")
        

if __name__ == '__main__':
    # Example usage with Snowflake
    # session = Session.builder.configs({
    #     "account": "your_account",
    #     "user": "your_user", 
    #     "password": "your_password",
    #     "role": "your_role",
    #     "warehouse": "your_warehouse",
    #     "database": "your_database"
    # }).create()
    
    # For file-based processing (default)
    luigi.build([ConvertLargeFiles(), PreProcess(), TuplesProcess(), ImputeScaleCategorize()], workers=2, local_scheduler=True)
    
    # For Snowflake processing, pass the session:
    # luigi.build([PreProcess(snowpark_session=session), TuplesProcess(snowpark_session=session), ImputeScaleCategorize(snowpark_session=session)], workers=2, local_scheduler=True)
    
    # For independent Snowflake processing (when steps run in separate worksheets):
    # Step 1: luigi.build([PreProcess(snowpark_session=session)], workers=1, local_scheduler=True)
    # Step 2: luigi.build([TuplesProcess(snowpark_session=session)], workers=1, local_scheduler=True)  
    # Step 3: luigi.build([ImputeScaleCategorize(snowpark_session=session)], workers=1, local_scheduler=True)
