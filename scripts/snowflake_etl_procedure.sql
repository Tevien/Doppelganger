-- Snowflake Stored Procedure for ETL Pipeline
-- This procedure runs PreProcess, TuplesProcess, and FillNaN steps
-- It reads configuration from the ETL_CONFIGS table

-- First, create the config table if it doesn't exist
CREATE TABLE IF NOT EXISTS ETL_CONFIGS (
    config_name VARCHAR PRIMARY KEY,
    config_json VARIANT NOT NULL,
    description VARCHAR,
    config_hash VARCHAR,
    created_date TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_date TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Add config_hash column if table already exists without it
ALTER TABLE ETL_CONFIGS ADD COLUMN IF NOT EXISTS config_hash VARCHAR;

CREATE OR REPLACE PROCEDURE dpplgngr_etl_pipeline(CONFIG_NAME VARCHAR)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = 3.9
ARTIFACT_REPOSITORY = snowflake.snowpark.pypi_shared_repository
PACKAGES = ('dpplgngr==0.4.29', 'dask==2023.10.0', 'dask-ml==2023.3.24', 'dateutils==0.6.12', 'joblib==1.3.1', 'matplotlib==3.7.2', 'numpy==1.25.2', 'pandas==2.2.2', 'polars==1.33.1', 'pyarrow==13.0.0', 'scikit-learn==1.3.0', 'scipy==1.11.1', 'snowflake-snowpark-python')
HANDLER = 'run_etl_pipeline'
AS
$$
def run_etl_pipeline(session, config_name):
    """
    Run the complete ETL pipeline (PreProcess -> TuplesProcess -> FillNaN)
    
    Args:
        session: Snowflake Snowpark session
        config_name: Name of the configuration stored in ETL_CONFIGS table
    
    Returns:
        String with status message
    """
    try:
        # Import required modules
        from dpplgngr.etl.prep_dataset_tabular import PreProcess, TuplesProcess, FillNaN
        import json
        import tempfile
        import os
       
        # Read configuration from ETL_CONFIGS table
        config_df = session.sql(
            f"SELECT config_json FROM ETL_CONFIGS WHERE config_name = '{config_name}'"
        ).collect()
        
        if len(config_df) == 0:
            # List available configs
            available_configs = session.sql(
                "SELECT config_name FROM ETL_CONFIGS ORDER BY config_name"
            ).collect()
            config_list = [row['CONFIG_NAME'] for row in available_configs]
            return f"Error: Configuration '{config_name}' not found in ETL_CONFIGS table. Available configs: {config_list}"
        
        # Extract the JSON config (Snowflake VARIANT is converted to Python dict)
        etl_config = json.loads(config_df[0]['CONFIG_JSON'])
        
        # Validate required fields
        if 'name' not in etl_config:
            return "Error: Configuration must include 'name' field"
        
        # Ensure SOURCE is set to SNOWFLAKE
        if 'SOURCE' not in etl_config:
            etl_config['SOURCE'] = 'SNOWFLAKE'
        
        # Set default snowflake_config if not present
        if 'snowflake_config' not in etl_config:
            etl_config['snowflake_config'] = {}
        
        # Set default schemas if not provided
        if 'input_schema' not in etl_config['snowflake_config']:
            etl_config['snowflake_config']['input_schema'] = session.get_current_schema()
        if 'output_schema' not in etl_config['snowflake_config']:
            etl_config['snowflake_config']['output_schema'] = session.get_current_schema()
        if 'database' not in etl_config['snowflake_config']:
            etl_config['snowflake_config']['database'] = session.get_current_database()
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(etl_config, f)
            config_path = f.name
        
        try:
            # Get schema information for logging
            current_db = session.get_current_database()
            current_schema = session.get_current_schema()
            input_schema = etl_config['snowflake_config']['input_schema']
            output_schema = etl_config['snowflake_config']['output_schema']
            
            # Log the table mappings from the configuration
            table_info = []
            for key, value in etl_config.items():
                if key.startswith('Datatools4heart_') and isinstance(value, dict):
                    table_name = value['table_name']
                    table_info.append(f"{input_schema}.{table_name}")
            
            # Step 1: PreProcess
            preprocessor = PreProcess(
                etl_config=config_path,
                snowpark_session=session
            )
            preprocessor.run()
            preprocessed_table = f"{output_schema}.{etl_config['name']}_preprocessed"
            
            # Step 2: TuplesProcess
            tuple_processor = TuplesProcess(
                etl_config=config_path,
                snowpark_session=session
            )
            tuple_processor.run()
            tupleprocessed_table = f"{output_schema}.{etl_config['name']}_tupleprocessed"
            
            # Step 3: FillNaN (Impute and Scale)
            imputer = FillNaN(
                etl_config=config_path,
                snowpark_session=session           
            )
            imputer.run()
            final_table = f"{output_schema}.{etl_config['name']}_preprocessed_imputed"
            
            return f"SUCCESS: ETL pipeline completed for configuration '{config_name}'! Database: {current_db}, Schema: {current_schema}. Reading from: {', '.join(table_info)}. Output tables: {preprocessed_table}, {tupleprocessed_table}, {final_table}"
        
        finally:
            # Clean up temporary file
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    except Exception as e:
        import traceback
        return f"ERROR: {str(e)}\n{traceback.format_exc()}"
$$;

-- Example usage:
-- First, load a config (usually done by the shell script):
-- MERGE INTO ETL_CONFIGS AS target
-- USING (SELECT 'my_config' AS config_name, PARSE_JSON('{...}') AS config_json, 'Description' AS description) AS source
-- ON target.config_name = source.config_name
-- WHEN MATCHED THEN UPDATE SET config_json = source.config_json, updated_date = CURRENT_TIMESTAMP()
-- WHEN NOT MATCHED THEN INSERT (config_name, config_json, description) VALUES (source.config_name, source.config_json, source.description);
--
-- Then call the procedure:
-- CALL dpplgngr_etl_pipeline('my_config');
--
-- To list available configs:
-- SELECT config_name, description, updated_date FROM ETL_CONFIGS ORDER BY config_name;
