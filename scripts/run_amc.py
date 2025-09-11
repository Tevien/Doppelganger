from dpplgngr.etl.prep_dataset_tabular import ConvertLargeFiles, PreProcess
import luigi
import argparse

# Set up the command line arguments
parser = argparse.ArgumentParser(description='Run the AMC ETL pipeline.')
parser.add_argument('--etlconfig', type=str, help='Path to the ETL configuration file.',
                    default='/home/sbenson/sw/Doppelganger/config/etl_amc_v3.json')
args = parser.parse_args()
etl_config = args.etlconfig

luigi.build([ConvertLargeFiles(etl_config=etl_config), PreProcess(etl_config=etl_config)], 
workers=2, local_scheduler=True, no_lock=False, log_level='INFO')
