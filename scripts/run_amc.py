from dpplgngr.etl.prep_dataset_tabular import ConvertLargeFiles, PreProcess, TuplesProcess
from dpplgngr.train.sdv import SDVGen

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

import argparse

# Set up the command line arguments
parser = argparse.ArgumentParser(description='Run the AMC ETL pipeline.')
parser.add_argument('--etlconfig', type=str, help='Path to the ETL configuration file.',
                    default='/home/sbenson/sw/Doppelganger/config/etl_amc_v3.json')
parser.add_argument('--genconfig', type=str, help='Path to the generation configuration file.',
                    default='/home/sbenson/sw/Doppelganger/config/synth.json')
args = parser.parse_args()
etl_config = args.etlconfig
gen_config = args.genconfig

luigi.build([ConvertLargeFiles(etl_config=etl_config), 
             PreProcess(etl_config=etl_config),
             TuplesProcess(etl_config=etl_config),
             SDVGen(gen_config=gen_config, etl_config=etl_config)
             ], 
workers=2, local_scheduler=True, no_lock=False, log_level='INFO')
