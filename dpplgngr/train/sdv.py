import dask.dataframe as dd
import pandas as pd
import polars as pl
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.lite import SingleTablePreset
from sdv.metadata import SingleTableMetadata
from realtabformer import REaLTabFormer
import logging
import json
import luigi
import os
# Import last ETL step for requirements
from dpplgngr.etl.prep_dataset_tabular import ImputeScaleCategorize, TuplesProcess
# Import missing value handler
from dpplgngr.utils.missing_value_handler import MissingValueHandler

logger = logging.getLogger('luigi-interface')

# meta data
__author__ = 'SB'
__date__ = '2024-01-28'

# Create a dictionary that maps strings to functions
function_dict = {
    "GC": GaussianCopulaSynthesizer,
    "CTGAN": CTGANSynthesizer,
    "TVAE": TVAESynthesizer,
    "RTF": [REaLTabFormer, {
        "model_type": "tabular", 
        "gradient_accumulation_steps": 4,
        "save_strategy": "no",
        "train_size": 1.0  # Use all data for training, disable validation/evaluation split
    }] # TODO: Make the options configurable
}

# Synthesizers that require NaN handling
SYNTHESIZERS_REQUIRING_NAN_FILL = {
    "RTF": -1000  # REaLTabFormer uses -1000 as missing value placeholder
}

# Synthesizers that can handle NaNs natively
SYNTHESIZERS_WITH_NAN_SUPPORT = ["GC", "CTGAN", "TVAE"]

class SDVGen(luigi.Task):
    gen_config = luigi.Parameter(default="config/synth.json")
    etl_config = luigi.Parameter(default="config/etl.json")
    override_etl = luigi.BoolParameter(default=False)

    def output(self):
        with open(self.gen_config, 'r') as f:
            input_json = json.load(f)
        outdir = input_json.get('working_dir', None)
        synth_type = input_json.get('synth_type', None)
        synth_out = f"synth_{synth_type}.pkl"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        return luigi.LocalTarget(os.path.join(outdir, synth_out))
    
    def requires(self):
        if self.override_etl:
            # Skip ETL requirement
            return []
        else:
            return TuplesProcess(etl_config=self.etl_config)

    def run(self):
        # Load input json
        with open(self.gen_config, 'r') as f:
            input_json = json.load(f)

        outdir = input_json.get('working_dir', None)
        synth_type = input_json.get('synth_type', None)
        num_points = int(input_json.get('num_points', None))
        cols = input_json.get('columns', None)
        synth_out = f"synth_{synth_type}.pkl"
        synth_out = os.path.join(outdir, synth_out)
        data_out = f"synthdata_{synth_type}_{num_points}.parquet"
        data_out = os.path.join(outdir, data_out)

        if not outdir:
            os.makedirs(outdir)
        
        # Load data
        df = pd.read_parquet(input_json['input_file'])

        # Convert Decimal columns to float
        from decimal import Decimal
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].apply(lambda x: isinstance(x, Decimal)).any():
                df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Make BMI physical
        # TODO: MOVE THIS TO ETL
        # Check for BMI column (could be "BMI" or "vital_signs_BMI_value_pET_first")
        # bmi_col = None
        # for col in df.columns:
        #     if 'BMI' in col or col == 'BMI':
        #         bmi_col = col
        #         break
        # if bmi_col is not None:
        #     df = df[pd.to_numeric(df[bmi_col], errors='coerce')<100]
        
        df = df[cols]

        # Check if a col is timedelta and convert
        for col in df.columns:
            if df[col].dtype.kind == 'm':  # 'm' indicates timedelta
                df[col] = df[col].dt.days  # Convert to days as float

        # Clean mixed-type columns before processing (especially for REaLTabFormer)
        logger.info("Cleaning data types for synthesizer compatibility...")
        for col in df.columns:
            # Check for mixed types in object columns
            if df[col].dtype == 'object':
                # Get non-null values
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    # Check if we have mixed types
                    types = non_null.apply(type).unique()
                    if len(types) > 1:
                        logger.warning(f"Column '{col}' has mixed types: {[t.__name__ for t in types]}. Attempting conversion...")
                        
                        # Try to convert to numeric first (most common case for categorical codes)
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            # Check if conversion was successful (not too many NaNs created)
                            new_nan_count = df[col].isna().sum()
                            original_nan_count_col = non_null.apply(lambda x: pd.isna(x)).sum()
                            if new_nan_count <= original_nan_count_col + 1:  # Allow 1 error
                                logger.info(f"  Column '{col}': successfully converted to numeric")
                            else:
                                # Too many conversion errors, fall back to string
                                df[col] = non_null.astype(str)
                                logger.info(f"  Column '{col}': converted to string (numeric conversion failed)")
                        except:
                            # Numeric conversion failed, use string
                            df[col] = df[col].astype(str)
                            df[col] = df[col].replace('nan', np.nan)
                            logger.info(f"  Column '{col}': converted to string")
                    else:
                        # Single type but object - try numeric first
                        try:
                            numeric_col = pd.to_numeric(df[col], errors='coerce')
                            # If most values convert successfully, use numeric
                            if numeric_col.notna().sum() >= non_null.shape[0] * 0.95:
                                df[col] = numeric_col
                                logger.info(f"  Column '{col}': converted to numeric (was object)")
                            else:
                                # Keep as string
                                df[col] = df[col].astype(str)
                                df[col] = df[col].replace('nan', np.nan)
                                logger.info(f"  Column '{col}': kept as string")
                        except:
                            df[col] = df[col].astype(str)
                            df[col] = df[col].replace('nan', np.nan)
                            logger.info(f"  Column '{col}': kept as string")

        # Store original data info
        original_nan_count = df.isna().sum().sum()
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Original data NaN count: {original_nan_count}")
        
        # For REaLTabFormer, ensure all object columns are purely string (no mixed int/str)
        # This prevents sklearn encoder errors during REaLTabFormer's internal evaluation
        if synth_type == "RTF":
            logger.info("Applying REaLTabFormer-specific data cleaning...")
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Convert all non-null values to string to ensure homogeneity
                    non_null_mask = df[col].notna()
                    if non_null_mask.any():
                        df.loc[non_null_mask, col] = df.loc[non_null_mask, col].astype(str)
                        logger.info(f"  Column '{col}': enforced string type for REaLTabFormer")
        
        # Handle missing values based on synthesizer type
        df_for_training = df.copy()
        fill_info = {'strategy': 'none', 'original_nans': int(original_nan_count)}
        
        if synth_type in SYNTHESIZERS_REQUIRING_NAN_FILL:
            if original_nan_count > 0:
                fill_value = SYNTHESIZERS_REQUIRING_NAN_FILL[synth_type]
                logger.info(f"="*60)
                logger.info(f"HANDLING MISSING VALUES FOR {synth_type}")
                logger.info(f"="*60)
                logger.info(f"Synthesizer '{synth_type}' requires NaN filling")
                logger.info(f"Filling {original_nan_count} NaN values with placeholder: {fill_value}")
                
                # Fill NaNs with type-appropriate placeholders
                for col in df_for_training.columns:
                    if df_for_training[col].isna().any():
                        if df_for_training[col].dtype == 'object':
                            # For categorical/object columns, use string placeholder
                            df_for_training[col] = df_for_training[col].fillna(str(fill_value))
                            logger.info(f"  Column '{col}' (categorical): filled with '{fill_value}'")
                        else:
                            # For numeric columns, use numeric placeholder
                            df_for_training[col] = df_for_training[col].fillna(fill_value)
                            logger.info(f"  Column '{col}' (numeric): filled with {fill_value}")
                
                fill_info.update({
                    'strategy': 'placeholder',
                    'fill_value': fill_value,
                    'synthesizer': synth_type,
                    'message': f'NaNs filled with {fill_value} for {synth_type}',
                    'nans_after_fill': int(df_for_training.isna().sum().sum())
                })
                logger.info(f"NaN count after filling: {fill_info['nans_after_fill']}")
                logger.info(f"="*60)
            else:
                logger.info(f"No NaN values to fill for {synth_type}")
        else:
            if original_nan_count > 0:
                logger.info(f"Synthesizer '{synth_type}' can handle NaN values natively ({original_nan_count} NaNs present)")
            fill_info.update({
                'strategy': 'native',
                'synthesizer': synth_type,
                'message': f'{synth_type} handles NaNs natively'
            })

        # Detect metadata from the prepared data
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_for_training)

        synth_fn = function_dict.get(synth_type)
        
        logger.info(f"Metadata: {metadata.to_dict()}")

        if type(synth_fn)==list:
            # REaLTabFormer - disable internal evaluation to avoid mixed-type issues
            synth_params = synth_fn[1].copy()
            synth_params['epochs'] = synth_params.get('epochs', 100)
            # Disable internal benchmarking/sensitivity analysis that causes mixed-type errors
            synthesizer = synth_fn[0](**synth_params)
        else:
            synthesizer = synth_fn(metadata=metadata)
        
        # Save df_for_training for debugging
        debug_path = os.path.join(outdir, f"df_for_training_{synth_type}.parquet")
        df_for_training.to_parquet(debug_path)
        logger.info(f"Prepared training data saved to: {debug_path}")
        
        # Train synthesizer on prepared data
        logger.info(f"Training {synth_type} synthesizer...")
        synthesizer.fit(df_for_training)

        # Sample synthetic data
        logger.info(f"Generating {num_points} synthetic samples...")
        if type(synth_fn)==list:
            synthetic_data = synthesizer.sample(n_samples=num_points)
            synthesizer.save(self.output().path+"/")
        else:
            synthetic_data = synthesizer.sample(num_rows=num_points)
            synthesizer.save(filepath=self.output().path)
        
        # Restore NaN values if they were filled with placeholder
        if synth_type in SYNTHESIZERS_REQUIRING_NAN_FILL and original_nan_count > 0:
            fill_value = SYNTHESIZERS_REQUIRING_NAN_FILL[synth_type]
            logger.info(f"Restoring NaN values in synthetic data...")
            
            # Restore NaNs based on column type
            for col in synthetic_data.columns:
                if synthetic_data[col].dtype == 'object':
                    # For categorical, replace string placeholder
                    placeholder_count = (synthetic_data[col] == str(fill_value)).sum()
                    if placeholder_count > 0:
                        synthetic_data[col] = synthetic_data[col].replace(str(fill_value), np.nan)
                        logger.info(f"  Column '{col}': converted {placeholder_count} placeholder values to NaN")
                else:
                    # For numeric, replace numeric placeholder
                    placeholder_count = (synthetic_data[col] == fill_value).sum()
                    if placeholder_count > 0:
                        synthetic_data[col] = synthetic_data[col].replace(fill_value, np.nan)
                        logger.info(f"  Column '{col}': converted {placeholder_count} placeholder values to NaN")
            
            final_nan_count = synthetic_data.isna().sum().sum()
            logger.info(f"Final synthetic data NaN count: {final_nan_count}")
            
            fill_info.update({
                'synthetic_nans_restored': int(final_nan_count)
            })
        
        # Save metadata and fill info
        metadata.save_to_json(self.output().path.replace('.pkl', '_metadata.json'))
        
        # Save fill info for transparency
        fill_info_path = self.output().path.replace('.pkl', '_fill_info.json')
        with open(fill_info_path, 'w') as f:
            json.dump(fill_info, f, indent=2)
        logger.info(f"Missing value handling info saved to: {fill_info_path}")

        # Save synthetic data
        synthetic_data.to_parquet(data_out)
        logger.info(f"Synthetic data saved to: {data_out}")
        logger.info(f"Synthetic data shape: {synthetic_data.shape}")
        logger.info(f"Synthetic data NaN count: {synthetic_data.isna().sum().sum()}")

