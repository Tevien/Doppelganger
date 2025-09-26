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

logger = logging.getLogger('luigi-interface')

# meta data
__author__ = 'SB'
__date__ = '2024-01-28'

# Create a dictionary that maps strings to functions
function_dict = {
    "GC": GaussianCopulaSynthesizer,
    "CTGAN": CTGANSynthesizer,
    "TVAE": TVAESynthesizer,
    "RTF": [REaLTabFormer, {"model_type": "tabular", "gradient_accumulation_steps":4}] # TODO: Make the options configurable
}

class SDVGen(luigi.Task):
    gen_config = luigi.Parameter(default="config/synth.json")
    etl_config = luigi.Parameter(default="config/etl.json")

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

        # Make BMI physical
        # TODO: MOVE THIS TO ETL
        df = df[df["BMI"]<100]
        
        df = df[cols]

        # Check if a col is timedelta and convert
        for col in df.columns:
            if df[col].dtype.kind == 'm':  # 'm' indicates timedelta
                df[col] = df[col].dt.days  # Convert to days as float

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        synth_fn = function_dict.get(synth_type)
        
        print(metadata.to_dict())

        if type(synth_fn)==list:
            # Replace NaN with -1000
            df = df.fillna(-1000)
            synthesizer = synth_fn[0](**synth_fn[1])
        else:
            synthesizer = synth_fn(metadata=metadata)
        
        synthesizer.fit(df)

        if type(synth_fn)==list:
            synthetic_data = synthesizer.sample(n_samples=num_points)
            # Replace -1000 with NaN
            synthetic_data = synthetic_data.replace(-1000, np.nan)
            synthesizer.save(self.output().path+"/")
        else:
            synthetic_data = synthesizer.sample(num_rows=num_points)
            synthesizer.save(filepath=self.output().path)
        metadata.save_to_json(self.output().path.replace('.pkl', '_metadata.json')) 

        synthetic_data.to_parquet(data_out)

