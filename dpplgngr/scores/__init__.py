"""
Clinical Scores Module

This module provides functions to calculate various clinical risk scores
and prediction models, as well as tools to evaluate their performance.

Available Scores
----------------
- MAGGIC: Heart failure mortality risk score
- AUDIT: Alcohol Use Disorders Identification Test
- Cox Prediction: Cox proportional hazards model predictions
- Privacy: Privacy metrics and scores
- PyMarker: Biomarker analysis tools

Pipeline Tasks
--------------
- CalculateScores: Calculate clinical scores and add them to datasets
- EvaluateScorePerformance: Evaluate and compare score performance

Example Usage
-------------
>>> from dpplgngr.scores import calculateMAGGIC
>>> import pandas as pd
>>> 
>>> # Prepare your data
>>> data = pd.DataFrame({
...     'Female': [1, 0, 1],
...     'Age_years': [65, 70, 75],
...     'LVEF_percent': [35, 40, 25],
...     # ... other columns
... })
>>> 
>>> # Define column mapping
>>> column_mapping = {
...     'sex': 'Female',
...     'age': 'Age_years',
...     'lvef': 'LVEF_percent',
...     # ... other mappings
... }
>>> 
>>> # Calculate the score
>>> scores = calculateMAGGIC(data, column_mapping)
>>> print(scores)
"""

from dpplgngr.scores.maggic import calculateMAGGIC
from dpplgngr.scores.calculate_scores import (
    CalculateScores,
    calculate_scores_standalone,
    get_score_calculator,
)
from dpplgngr.scores.evaluate_scores import (
    EvaluateScorePerformance,
    evaluate_score_performance,
)

__all__ = [
    # Score calculation functions
    'calculateMAGGIC',
    'calculateAUDIT',
    # Pipeline tasks
    'CalculateScores',
    'EvaluateScorePerformance',
    # Standalone functions
    'calculate_scores_standalone',
    'get_score_calculator',
    'evaluate_score_performance',
]

# Version information
__version__ = '2.0.0'
__author__ = 'SB'
