"""
ETL (Extract, Transform, Load) Module

This module provides pipeline tasks for data preprocessing and transformation.

Available Tasks
---------------
- ConvertLargeFiles: Convert large CSV files to Parquet format
- PreProcess: Main data preprocessing task
- TuplesProcess: Process tuple-based data structures
- ImputeScaleCategorize: Impute missing values, scale features, and categorize variables
- AlternativeImputation: Create alternative imputations using different strategies
- CalculateScores: Calculate clinical risk scores and add them to datasets
- EvaluateScorePerformance: Evaluate and compare score performance across datasets

Example Usage
-------------
>>> import luigi
>>> from dpplgngr.etl import PreProcess, AlternativeImputation, CalculateScores, EvaluateScorePerformance
>>> 
>>> # Run complete pipeline with multiple imputations
>>> from dpplgngr.utils.study_config import StudyConfig
>>> config = StudyConfig('/path/to/study')
>>> 
>>> # Create alternative imputations
>>> luigi.build([
>>>     AlternativeImputation(
>>>         input_file=config.get_preprocessed_file(),
>>>         output_dir=str(config.imputed_dir),
>>>         imputation_strategies='mean,median,knn,iterative'
>>>     )
>>> ], local_scheduler=True)
"""

from dpplgngr.etl.prep_dataset_tabular import (
    ConvertLargeFiles,
    PreProcess,
    TuplesProcess,
    ImputeScaleCategorize,
)

# Note: Score calculation and evaluation moved to dpplgngr.scores
# Imputation moved to dpplgngr.train
# Import them from there instead:
# from dpplgngr.scores.calculate_scores import CalculateScores
# from dpplgngr.scores.evaluate_scores import EvaluateScorePerformance
# from dpplgngr.train.alternative_imputation import AlternativeImputation

__all__ = [
    # Pipeline tasks
    'ConvertLargeFiles',
    'PreProcess',
    'TuplesProcess',
    'ImputeScaleCategorize',
]

__version__ = '2.0.0'
__author__ = 'SB'
