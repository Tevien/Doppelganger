from dpplgngr.utils.utils_etl import file_size, return_subset, vals_to_cols
from dpplgngr.utils.functions import transform_aggregations, merged_transforms
from dpplgngr.etl.convert_to_parquet import convert
import dask.dataframe as dd
from dask_ml.impute import SimpleImputer
from dask_ml.preprocessing import StandardScaler
from joblib import load, dump
import pandas as pd
import polars as pl
import numpy as np
import logging
import json
import luigi
import os

logger = logging.getLogger('luigi-interface')

# meta data
__author__ = 'SB'
__date__ = '2023-09-25'

def dask_shape(_df):
    a = _df.shape
    a[0].compute(),a[1]
    return a

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

    def requires(self):
        return ConvertLargeFiles(etl_config=self.etl_config)
    
    def output(self):
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        outdir = input_json.get('preprocessing', None)
        if not outdir:
            name = input_json.get('name', None)
            outdir = f"data/{name}/preprocessing"
        print(f"Output directory: {outdir}")
        print(f"Output path: {self.lu_output_path}")
        return luigi.LocalTarget(os.path.join(outdir, self.lu_output_path))
    
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

        print("Aggregation columns: ", cols_for_aggregations)
        # Apply aggregations
        if cols_for_aggregations:
            s_df = transform_aggregations(s_df, aggs, cols_for_aggregations)
        return s_df
    
    def safe_merge(self, _df, _df_pp):
        # remove any of ["Patientcontactid", "PatientContactId"] from _df if it exists
        if "Patientcontactid" in _df.columns:
            _df = _df.drop(columns=["Patientcontactid"])
        if "PatientContactId" in _df.columns:
            _df = _df.drop(columns=["PatientContactId"])

        print("df_pp premerge")
        if _df_pp is not None:
            logging.info(f"Shape before merge: {dask_shape(_df_pp)}")
        logging.info(f"New data shape: {dask_shape(_df)}")
        if _df_pp is None:
            _df_pp = _df
        else:
            _df_pp = _df_pp.merge(_df, how="left")
        _df_pp_20 = _df.head(20)
        logging.info("df_pp postmerge")
        logging.info(_df_pp_20)
        print(f"Shape after merge: {dask_shape(_df_pp)}")
        return _df_pp

    def run(self):
        # Load input json
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)

        filenames = input_json.keys()

        # Keep only names with normal extensions
        filenames = [f for f in filenames if any([f.endswith(ext) for ext in ['.csv', '.csv.gz', '.parquet', '.feather']])]
        path = input_json['absolute_path']
        filenames = [os.path.join(path, f) for f in filenames]
        
        # Load converted json
        with self.input().open('r') as f:
            converted_json = json.load(f)
        filenames_to_convert = converted_json.keys()
        
        # Replace with converted filenames
        new_filenames = [converted_json[f] if f in filenames_to_convert else f for f in filenames]

        # Open empty dask dataframe
        df_pp = None

        logging.info(f"Filenames: {filenames}")
        # Create the requested columns from each filename
        for o, f in zip(filenames, new_filenames):
            # First check if we have this file checkpointed
            current_name = f.split("/")[-1].split(".")[0]
            saved_loc = f"{input_json['preprocessing']}/{current_name}_preprocessed.parquet"
            # If file exists then load it
            if os.path.exists(saved_loc):
                print(f"*** Loading {saved_loc} ***")
                df = dd.read_parquet(saved_loc, npartitions=3) #TODO: Make this configurable
                print(df.head())
                print("Index: ", df.index.name)
                print("Columns: ", df.columns)
                df_pp_20 = df.head(20)
                df_pp = self.safe_merge(df, df_pp)
                continue

            print(f"*** Processing {f} ***")
            vals = input_json[o.split("/")[-1]]
            index = vals[0]
            cols = vals[1:]

            print(f"Index: {index}")
            print(f"Columns: {cols}")

            # are any items in the list dictionaries?
            col_extract = not any([isinstance(v, dict) for v in vals])
            logger.info(f"*** Column extraction: {col_extract} ***")
            if col_extract:
                df = return_subset(f, cols, index_col=index)
            else:
                """ Assume form of cols is:
                ["col_name", {"val_name": {"type1": name1, "type2": name2}}, ["optional_extra_col"]]
                """

                if '.parquet' in f:
                    df = dd.read_parquet(f)
                elif '.feather' in f:
                    df = dd.from_pandas(pd.read_feather(f), npartitions=3)
                else:
                    df = dd.read_csv(f, blocksize=blocksize)

                # Apply initial transformations
                print(df.head(20))
                print(f"Columns: {df.columns.tolist()}")
                df = self.apply_transformations(df, input_json, cols, transform_type="InitTransforms")
                print(df.head(20))
                print(f"Columns: {df.columns.tolist()}")
                
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


            df_20 = df.head(20)
            logging.info("df pre transform:")
            logging.info(df_20)

            df = self.apply_transformations(df, input_json, cols, transform_type="PreTransforms")

            # Add new cols to the dataframe and join with subject_id
            #df_100 = df.head(100)
            #print(df_100)
            df_20 = df.head(20)
            print(df_20)
            logging.info("df post transform:")
            logging.info(df_20)
            # Checkpoint pre-concat
            # Save current transformed dataframe
            df.to_parquet(saved_loc)

            df_pp_20 = df.head(20)
            df_pp = self.safe_merge(df, df_pp)
        
        # Merged transforms
        end_transforms = input_json.get('MergedTransforms', None)
        if end_transforms:
            df_pp = merged_transforms(df_pp, end_transforms)

        
        # Reduce to final specified columns
        df_pp = df_pp[input_json["final_cols"]]
        print(df_pp.head(20))
        print(f"Final shape: {df_pp.shape}")
        
        df_pp.to_parquet(self.output())
        print("Success")


class ImputeScaleCategorize(luigi.Task):
    lu_output_path = luigi.Parameter(default='preprocessed_imputed.parquet')
    etl_config = luigi.Parameter(default="config/etl.json")

    def requires(self):
        return PreProcess(etl_config=self.etl_config)
    
    def output(self):
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        outdir = input_json.get('preprocessing', None)
        if not outdir:
            name = input_json.get('name', None)
            outdir = f"data/{name}/preprocessing"
        return luigi.LocalTarget(os.path.join(prefix, self.lu_output_path))
    
    def run(self):
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        name = input_json['name']

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
        
        ddf = dd.read_parquet(self.input())

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
        
        ddf.to_parquet(self.output().path)
        

if __name__ == '__main__':    
    luigi.build([ConvertLargeFiles(), PreProcess(), ImputeScaleCategorize()], workers=2, local_scheduler=True)
