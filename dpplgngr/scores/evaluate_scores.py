"""
Score Performance Evaluation Pipeline Step

This module provides a Luigi pipeline task for evaluating and comparing
the performance of calculated clinical scores (e.g., MAGGIC) across different
imputation strategies or datasets.

The evaluation includes:
- Predictive performance for mortality (AUROC, AUPRC, F1, PPV, NPV, Sensitivity, Specificity)
- Calibration and Brier score
- Score distribution analysis
- Missing data impact assessment
- Comparison across multiple datasets
- Bootstrap confidence intervals

For programmatic use, prefer ``dpplgngr.scores.pipeline.ScoreAnalysisPipeline``
which wraps everything (feature resolution, scoring, evaluation) in a single
call.  This module remains for Luigi-based workflow compatibility.

Author: SB
Date: 2025-10-31
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from pathlib import Path


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Parameters
    ----------
    obj : any
        Object to convert (can be dict, list, numpy type, etc.)
    
    Returns
    -------
    any
        Object with numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

# Try to import luigi
try:
    import luigi
    _using_luigi_replacement = False
except ImportError:
    from dpplgngr.utils.luigi_replacement import Task, Parameter, LocalTarget, build as luigi_build
    class MockLuigi:
        Task = Task
        Parameter = Parameter
        LocalTarget = LocalTarget
        @staticmethod
        def build(*args, **kwargs):
            return luigi_build(*args, **kwargs)
    luigi = MockLuigi()
    _using_luigi_replacement = True

# Try to import sklearn metrics
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.calibration import calibration_curve
    _sklearn_available = True
except ImportError:
    _sklearn_available = False
    logging.warning("sklearn not available, evaluation metrics will be limited")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluateScorePerformance(luigi.Task):
    """
    Luigi task to evaluate and compare MAGGIC score performance across datasets.
    
    This task:
    1. Reads multiple datasets with calculated MAGGIC scores
    2. Evaluates predictive performance for 1-year mortality
    3. Compares score distributions
    4. Generates comprehensive comparison report
    
    Parameters
    ----------
    input_files : str
        Comma-separated list of file paths with calculated scores
    input_labels : str
        Comma-separated list of labels for each input file
    outcome_column : str
        Name of the binary outcome column (1-year mortality)
    score_column : str
        Name of the score column to evaluate (default: 'maggic (1-year risk of death)')
    output_file : str
        Path to save the evaluation report (JSON)
    output_plots_dir : str, optional
        Directory to save evaluation plots
    score_config : str, optional
        Path to score config file (for dependency tracking)
    graph_config : str, optional
        Path to graph imputation config file (for dependency tracking)
    alt_impute_config : str, optional
        Path to alternative imputation config file (for dependency tracking)
    """
    
    input_files = luigi.Parameter()
    input_labels = luigi.Parameter()
    outcome_column = luigi.Parameter()
    score_column = luigi.Parameter(default='maggic (1-year risk of death)')
    output_file = luigi.Parameter()
    output_plots_dir = luigi.Parameter(default='')
    score_config = luigi.Parameter(default='')
    graph_config = luigi.Parameter(default='')
    alt_impute_config = luigi.Parameter(default='')
    
    def output(self):
        """Define the output target for this task."""
        return luigi.LocalTarget(self.output_file)
    
    def requires(self):
        """
        Declare dependencies on all CalculateScores tasks.
        This ensures evaluation only runs after all scoring is complete.
        """
        if not self.score_config:
            # No dependency tracking configured
            return []
        
        from dpplgngr.scores.calculate_scores import CalculateScores
        from dpplgngr.train.graph_imputation import ApplyGraphImputer
        from dpplgngr.train.alternative_imputation import AlternativeImputation
        
        dependencies = []
        files = [f.strip() for f in self.input_files.split(',')]
        labels = [l.strip() for l in self.input_labels.split(',')]
        
        # Create CalculateScores dependencies for each input file
        for file_path, label in zip(files, labels):
            if label == 'graph_imputation' and self.graph_config:
                # This is a graph imputed file
                input_file = file_path.replace('_scored.parquet', '.parquet')
                dependencies.append(CalculateScores(
                    input_file=input_file,
                    output_file=file_path,
                    score_config=self.score_config,
                    depends_on_task='ApplyGraphImputer',
                    depends_on_config=self.graph_config
                ))
            elif self.alt_impute_config:
                # This is an alternative imputed file
                input_file = file_path.replace('_scored.parquet', '.parquet')
                dependencies.append(CalculateScores(
                    input_file=input_file,
                    output_file=file_path,
                    score_config=self.score_config,
                    depends_on_task='AlternativeImputation',
                    depends_on_config=self.alt_impute_config
                ))
        
        return dependencies
    
    def run(self):
        """Execute the evaluation pipeline."""
        logger.info(f"Starting score performance evaluation")
        logger.info(f"Output file: {self.output_file}")
        
        # Parse input files and labels
        files = [f.strip() for f in self.input_files.split(',')]
        labels = [l.strip() for l in self.input_labels.split(',')]
        
        if len(files) != len(labels):
            raise ValueError(f"Number of files ({len(files)}) must match number of labels ({len(labels)})")
        
        logger.info(f"Evaluating {len(files)} datasets:")
        for label, file in zip(labels, files):
            logger.info(f"  - {label}: {file}")
        
        # Load all datasets
        datasets = {}
        for label, file in zip(labels, files):
            logger.info(f"\nLoading {label}...")
            data = self._load_data(file)
            logger.info(f"  Shape: {data.shape}")
            logger.info(f"  Columns: {data.columns.tolist()[:10]}..." if len(data.columns) > 10 else f"  Columns: {data.columns.tolist()}")
            datasets[label] = data
        
        # Check for outcome column
        logger.info(f"\nChecking for outcome column: {self.outcome_column}")
        for label, data in datasets.items():
            if self.outcome_column not in data.columns:
                logger.warning(f"  WARNING: Outcome column '{self.outcome_column}' not found in {label}")
                logger.warning(f"  Available columns: {data.columns.tolist()}")
            else:
                n_events = data[self.outcome_column].sum()
                n_total = len(data)
                logger.info(f"  {label}: {n_events}/{n_total} events ({n_events/n_total*100:.1f}%)")
        
        # Evaluate each dataset
        results = {}
        for label, data in datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {label}")
            logger.info(f"{'='*60}")
            
            try:
                eval_results = self._evaluate_dataset(data, label)
                results[label] = eval_results
            except Exception as e:
                logger.error(f"Failed to evaluate {label}: {e}")
                import traceback
                traceback.print_exc()
                results[label] = {'error': str(e)}
        
        # Compare results
        logger.info(f"\n{'='*60}")
        logger.info(f"Comparison Summary")
        logger.info(f"{'='*60}")
        comparison = self._compare_results(results)
        
        # Create output directory
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        full_results = {
            'individual_results': results,
            'comparison': comparison,
            'metadata': {
                'outcome_column': self.outcome_column,
                'score_column': self.score_column,
                'n_datasets': len(files),
                'datasets': dict(zip(labels, files))
            }
        }
        
        # Convert numpy types to Python types for JSON serialization
        full_results = convert_numpy_types(full_results)
        
        with open(self.output_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        logger.info(f"\nEvaluation results saved to: {self.output_file}")
        
        # Create plots if directory specified
        if self.output_plots_dir:
            self._create_plots(datasets, results, self.output_plots_dir)
    
    def _load_data(self, file_path):
        """Load data from various file formats."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext == '.parquet':
            return pd.read_parquet(file_path)
        elif file_ext == '.feather':
            return pd.read_feather(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _evaluate_dataset(self, data, label):
        """
        Evaluate a single dataset with calculated scores.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset with scores and outcomes
        label : str
            Label for this dataset
        
        Returns
        -------
        dict
            Evaluation metrics
        """
        results = {
            'n_samples': len(data),
            'n_features': len(data.columns),
        }
        
        # Check if score column exists
        if self.score_column not in data.columns:
            logger.warning(f"Score column '{self.score_column}' not found in {label}")
            logger.warning(f"Available columns with 'maggic': {[c for c in data.columns if 'maggic' in c.lower()]}")
            results['score_available'] = False
            return results
        
        results['score_available'] = True
        
        # Score distribution statistics
        score_data = data[self.score_column].dropna()
        results['score_stats'] = {
            'n_valid': len(score_data),
            'n_missing': data[self.score_column].isnull().sum(),
            'mean': float(score_data.mean()),
            'std': float(score_data.std()),
            'min': float(score_data.min()),
            'q25': float(score_data.quantile(0.25)),
            'median': float(score_data.quantile(0.50)),
            'q75': float(score_data.quantile(0.75)),
            'max': float(score_data.max()),
        }
        
        logger.info(f"Score statistics:")
        logger.info(f"  Mean: {results['score_stats']['mean']:.4f}")
        logger.info(f"  Median: {results['score_stats']['median']:.4f}")
        logger.info(f"  Range: [{results['score_stats']['min']:.4f}, {results['score_stats']['max']:.4f}]")
        logger.info(f"  Missing: {results['score_stats']['n_missing']}")
        
        # Check if outcome column exists
        if self.outcome_column not in data.columns:
            logger.warning(f"Outcome column '{self.outcome_column}' not found in {label}")
            results['outcome_available'] = False
            return results
        
        results['outcome_available'] = True
        
        # Outcome statistics
        outcome_data = data[self.outcome_column].dropna()
        results['outcome_stats'] = {
            'n_valid': len(outcome_data),
            'n_missing': data[self.outcome_column].isnull().sum(),
            'n_events': int(outcome_data.sum()),
            'event_rate': float(outcome_data.mean()),
        }
        
        logger.info(f"Outcome statistics:")
        logger.info(f"  Events: {results['outcome_stats']['n_events']}/{results['outcome_stats']['n_valid']}")
        logger.info(f"  Event rate: {results['outcome_stats']['event_rate']:.2%}")
        
        # Predictive performance metrics
        if _sklearn_available:
            # Get complete cases
            valid_mask = data[self.score_column].notna() & data[self.outcome_column].notna()
            y_true = data.loc[valid_mask, self.outcome_column].values
            y_pred = data.loc[valid_mask, self.score_column].values
            
            if len(y_true) > 0 and len(np.unique(y_true)) > 1:
                try:
                    # Use the new comprehensive metrics module
                    from dpplgngr.scores.metrics import evaluate_score as _eval
                    eval_result = _eval(
                        y_true=y_true,
                        y_score=y_pred,
                        score_name=label,
                        outcome_column=self.outcome_column,
                        score_column=self.score_column,
                        n_bootstrap=500,  # fewer for speed in Luigi context
                    )
                    results['performance'] = {
                        'n_evaluated': len(y_true),
                        'auroc': eval_result.auroc,
                        'auprc': eval_result.auprc,
                        'brier_score': eval_result.brier_score,
                        'f1': eval_result.f1,
                        'ppv': eval_result.ppv,
                        'npv': eval_result.npv,
                        'sensitivity': eval_result.sensitivity,
                        'specificity': eval_result.specificity,
                        'accuracy': eval_result.accuracy,
                        'threshold': eval_result.threshold,
                    }

                    # Add CIs if available
                    for m in ('auroc', 'auprc', 'f1', 'ppv', 'npv', 'sensitivity', 'specificity'):
                        ci = getattr(eval_result, f'{m}_ci', None)
                        if ci is not None:
                            results['performance'][f'{m}_ci'] = list(ci)

                    logger.info(f"Predictive performance:")
                    logger.info(f"  AUROC:       {eval_result.auroc:.4f}" if eval_result.auroc else "  AUROC:       N/A")
                    logger.info(f"  AUPRC:       {eval_result.auprc:.4f}" if eval_result.auprc else "  AUPRC:       N/A")
                    logger.info(f"  F1:          {eval_result.f1:.4f}" if eval_result.f1 else "  F1:          N/A")
                    logger.info(f"  PPV:         {eval_result.ppv:.4f}" if eval_result.ppv else "  PPV:         N/A")
                    logger.info(f"  NPV:         {eval_result.npv:.4f}" if eval_result.npv else "  NPV:         N/A")
                    logger.info(f"  Sensitivity: {eval_result.sensitivity:.4f}" if eval_result.sensitivity else "  Sensitivity: N/A")
                    logger.info(f"  Specificity: {eval_result.specificity:.4f}" if eval_result.specificity else "  Specificity: N/A")
                    logger.info(f"  Brier Score: {eval_result.brier_score:.4f}" if eval_result.brier_score else "  Brier Score: N/A")
                    logger.info(f"  Threshold:   {eval_result.threshold:.4f}" if eval_result.threshold else "  Threshold:   N/A")

                    # Calibration
                    if eval_result.calibration_prob_true is not None:
                        results['calibration'] = {
                            'prob_true': eval_result.calibration_prob_true,
                            'prob_pred': eval_result.calibration_prob_pred,
                        }

                except Exception as e:
                    logger.error(f"Error computing performance metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    results['performance_error'] = str(e)
            else:
                logger.warning(f"Cannot compute performance metrics: insufficient data or no variation in outcome")
                results['performance_error'] = 'Insufficient data or no outcome variation'
        else:
            logger.warning("sklearn not available, skipping performance metrics")
            results['performance_error'] = 'sklearn not available'
        
        return results
    
    def _compare_results(self, results):
        """
        Compare results across datasets.
        
        Parameters
        ----------
        results : dict
            Dictionary mapping labels to evaluation results
        
        Returns
        -------
        dict
            Comparison summary
        """
        comparison = {
            'n_datasets': len(results),
            'datasets': list(results.keys()),
        }
        
        # Compare score distributions
        score_means = {}
        score_medians = {}
        for label, res in results.items():
            if 'score_stats' in res:
                score_means[label] = res['score_stats']['mean']
                score_medians[label] = res['score_stats']['median']
        
        if score_means:
            comparison['score_means'] = score_means
            comparison['score_medians'] = score_medians
            logger.info("\nScore means by dataset:")
            for label, mean in sorted(score_means.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {label}: {mean:.4f}")
        
        # Compare all performance metrics
        metric_keys = ['auroc', 'auprc', 'brier_score', 'f1', 'ppv', 'npv',
                       'sensitivity', 'specificity', 'accuracy', 'threshold']
        perf_data = {}  # label -> {metric: value}
        for label, res in results.items():
            if 'performance' in res:
                perf_data[label] = {m: res['performance'].get(m) for m in metric_keys}

        if perf_data:
            # Store per-metric lookups for backward compat
            comparison['aurocs'] = {l: d['auroc'] for l, d in perf_data.items() if d.get('auroc') is not None}
            comparison['auprcs'] = {l: d['auprc'] for l, d in perf_data.items() if d.get('auprc') is not None}
            comparison['brier_scores'] = {l: d['brier_score'] for l, d in perf_data.items() if d.get('brier_score') is not None}
            comparison['f1_scores'] = {l: d['f1'] for l, d in perf_data.items() if d.get('f1') is not None}
            comparison['ppv_scores'] = {l: d['ppv'] for l, d in perf_data.items() if d.get('ppv') is not None}
            comparison['npv_scores'] = {l: d['npv'] for l, d in perf_data.items() if d.get('npv') is not None}
            comparison['sensitivity_scores'] = {l: d['sensitivity'] for l, d in perf_data.items() if d.get('sensitivity') is not None}
            comparison['specificity_scores'] = {l: d['specificity'] for l, d in perf_data.items() if d.get('specificity') is not None}

            # Log key metrics
            for metric_name, display_name in [('auroc', 'AUROC'), ('auprc', 'AUPRC'),
                                               ('f1', 'F1'), ('ppv', 'PPV'), ('npv', 'NPV'),
                                               ('sensitivity', 'Sensitivity'), ('specificity', 'Specificity')]:
                vals = {l: d[metric_name] for l, d in perf_data.items() if d.get(metric_name) is not None}
                if vals:
                    logger.info(f"\n{display_name} by dataset:")
                    for label, val in sorted(vals.items(), key=lambda x: x[1], reverse=True):
                        logger.info(f"  {label}: {val:.4f}")

            # Rank datasets (using metrics where available)
            aurocs = comparison.get('aurocs', {})
            auprcs = comparison.get('auprcs', {})
            briers = comparison.get('brier_scores', {})
            comparison['rankings'] = {}
            if aurocs:
                comparison['rankings']['by_auroc'] = sorted(aurocs.items(), key=lambda x: x[1], reverse=True)
            if auprcs:
                comparison['rankings']['by_auprc'] = sorted(auprcs.items(), key=lambda x: x[1], reverse=True)
            if briers:
                comparison['rankings']['by_brier'] = sorted(briers.items(), key=lambda x: x[1])  # Lower is better
            f1s = comparison.get('f1_scores', {})
            if f1s:
                comparison['rankings']['by_f1'] = sorted(f1s.items(), key=lambda x: x[1], reverse=True)

            if aurocs:
                logger.info("\nBest performing dataset (by AUROC):")
                best_label, best_auroc = comparison['rankings']['by_auroc'][0]
                logger.info(f"  {best_label}: {best_auroc:.4f}")
        
        return comparison
    
    def _find_common_numeric_features(self, datasets, max_features=20):
        """
        Find common numeric features across datasets for distribution comparison.
        
        Parameters
        ----------
        datasets : dict
            Dictionary mapping labels to dataframes
        max_features : int
            Maximum number of features to return
        
        Returns
        -------
        list
            List of common numeric feature names
        """
        # Get intersection of columns across all datasets
        common_cols = set(next(iter(datasets.values())).columns)
        for data in datasets.values():
            common_cols = common_cols.intersection(set(data.columns))
        
        logger.info(f"    Total common columns across datasets: {len(common_cols)}")
        
        # Filter to numeric columns only (exclude datetime types)
        numeric_cols = []
        first_data = next(iter(datasets.values()))
        for col in common_cols:
            # Check if numeric but NOT datetime
            if pd.api.types.is_numeric_dtype(first_data[col]) and \
               not pd.api.types.is_datetime64_any_dtype(first_data[col]):
                # Exclude outcome, score, and ID columns
                if col not in [self.outcome_column, self.score_column] and \
                   not col.lower().endswith('_id') and \
                   not col.lower().startswith('id_') and \
                   'date' not in col.lower() and \
                   'time' not in col.lower():
                    numeric_cols.append(col)
        
        logger.info(f"    Numeric columns after filtering: {len(numeric_cols)}")
        
        # Sort by variance (features with more variance are more interesting)
        if numeric_cols:
            variances = {}
            for col in numeric_cols:
                try:
                    all_values = pd.concat([data[col].dropna() for data in datasets.values()])
                    # Try to convert to numeric in case there are any edge cases
                    all_values = pd.to_numeric(all_values, errors='coerce')
                    var = all_values.var()
                    if pd.notna(var) and var > 0:
                        variances[col] = var
                except Exception as e:
                    # Skip columns that cause issues
                    logger.debug(f"    Skipping column {col} due to error: {e}")
                    continue
            
            logger.info(f"    Columns with valid variance: {len(variances)}")
            
            # Sort by variance and take top features (only those with valid variance)
            numeric_cols = [col for col in numeric_cols if col in variances]
            numeric_cols = sorted(numeric_cols, key=lambda x: variances[x], reverse=True)[:max_features]
            
            logger.info(f"    Returning top {len(numeric_cols)} features by variance")
        
        return numeric_cols
    
    def _create_auc_scan_plot(self, datasets, output_dir):
        """
        Create AUC scan plot showing performance across MAGGIC thresholds for predicting high/low survival time.
        
        Parameters
        ----------
        datasets : dict
            Dictionary mapping labels to dataframes
        output_dir : str
            Directory to save the plot
        """
        # Try to find time variable (common names)
        time_col_candidates = ['time_to_event', 'survival_time', 'followup_time', 'time', 
                               'duration', 'time_to_death', 'fu_time', 'follow_up_time']
        
        time_col = None
        for candidate in time_col_candidates:
            if any(candidate in data.columns for data in datasets.values()):
                time_col = candidate
                break
        
        # Also check case-insensitive
        if time_col is None:
            first_data = next(iter(datasets.values()))
            for col in first_data.columns:
                col_lower = col.lower()
                if 'time' in col_lower and 'event' not in col_lower and col != self.outcome_column:
                    time_col = col
                    break
        
        if time_col is None:
            logger.warning("Could not find time variable for AUC scan plot")
            return
        
        logger.info(f"  Creating AUC scan plot using time variable: {time_col}")
        
        try:
            from sklearn.metrics import roc_auc_score
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Define threshold range for MAGGIC scores (typically 0-1 range)
            thresholds = np.linspace(0.05, 0.95, 50)
            
            for label, data in datasets.items():
                # Check if required columns exist
                if self.score_column not in data.columns or time_col not in data.columns:
                    logger.warning(f"  Skipping {label}: missing score or time column")
                    continue
                
                # Get valid data
                valid_mask = data[self.score_column].notna() & data[time_col].notna()
                scores = data.loc[valid_mask, self.score_column].values
                times_raw = data.loc[valid_mask, time_col].values
                
                if len(scores) < 10:
                    logger.warning(f"  Skipping {label}: insufficient data ({len(scores)} samples)")
                    continue
                
                # Convert times to numeric if they are datetime
                if pd.api.types.is_datetime64_any_dtype(times_raw):
                    logger.info(f"    Converting datetime column {time_col} to numeric (days)")
                    # Convert to timedelta from a reference point (first observation)
                    reference_time = pd.Timestamp(times_raw[0])
                    times_delta = pd.to_datetime(times_raw) - reference_time
                    # Convert to days (total_seconds() / 86400)
                    times = times_delta.total_seconds() / 86400.0
                    times = times if isinstance(times, np.ndarray) else times.values
                else:
                    # Already numeric
                    times = pd.to_numeric(times_raw, errors='coerce')
                    times = times.values if hasattr(times, 'values') else times
                
                # Remove any NaN values that might have been introduced
                valid_times_mask = ~np.isnan(times)
                if not valid_times_mask.all():
                    logger.warning(f"  Removing {(~valid_times_mask).sum()} samples with invalid time values")
                    times = times[valid_times_mask]
                    scores = scores[valid_times_mask]
                
                if len(scores) < 10:
                    logger.warning(f"  Skipping {label}: insufficient valid data after time conversion ({len(scores)} samples)")
                    continue
                
                # Create binary outcome: high vs low survival time (split at median)
                median_time = np.median(times)
                high_time = (times >= median_time).astype(int)
                
                logger.info(f"    {label}: median time = {median_time:.2f}, {np.sum(high_time)} high / {len(high_time)} total")
                
                # Compute AUC at each threshold
                aucs = []
                for threshold in thresholds:
                    # Predict high risk if MAGGIC score >= threshold
                    predictions = (scores >= threshold).astype(int)
                    
                    # Check if we have both classes
                    if len(np.unique(predictions)) > 1 and len(np.unique(high_time)) > 1:
                        try:
                            # AUC for predicting LOW time (high risk) from high MAGGIC score
                            auc = roc_auc_score(1 - high_time, predictions)
                            aucs.append(auc)
                        except:
                            aucs.append(np.nan)
                    else:
                        aucs.append(np.nan)
                
                # Plot
                ax.plot(thresholds, aucs, marker='o', markersize=3, label=label, linewidth=2)
            
            ax.set_xlabel('MAGGIC Score Threshold', fontsize=12)
            ax.set_ylabel('AUC (Predicting Low Survival Time)', fontsize=12)
            ax.set_title(f'AUC Scan: Predicting Below-Median Survival Time\n(Median split on {time_col})', fontsize=13)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            ax.set_ylim([0.3, 1.0])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'auc_scan_survival_time.png'), dpi=150)
            plt.close()
            logger.info("  Created: auc_scan_survival_time.png")
            
        except ImportError:
            logger.warning("sklearn not available for AUC scan plot")
        except Exception as e:
            logger.error(f"Error creating AUC scan plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_plots(self, datasets, results, output_dir):
        """Create visualization plots for comparison."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"\nCreating plots in {output_dir}...")
            
            # Plot 1: Score distributions
            fig, ax = plt.subplots(figsize=(10, 6))
            for label, data in datasets.items():
                if self.score_column in data.columns:
                    data[self.score_column].dropna().hist(alpha=0.5, label=label, bins=30, ax=ax)
            ax.set_xlabel('MAGGIC Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Score Distribution Comparison')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=150)
            plt.close()
            logger.info("  Created: score_distributions.png")
            
            # Plot 2: Performance comparison
            if any('performance' in r for r in results.values()):
                metrics = ['auroc', 'auprc']
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                for idx, metric in enumerate(metrics):
                    labels = []
                    values = []
                    for label, res in results.items():
                        if 'performance' in res:
                            labels.append(label)
                            values.append(res['performance'][metric])
                    
                    axes[idx].bar(range(len(labels)), values)
                    axes[idx].set_xticks(range(len(labels)))
                    axes[idx].set_xticklabels(labels, rotation=45, ha='right')
                    axes[idx].set_ylabel(metric.upper())
                    axes[idx].set_title(f'{metric.upper()} Comparison')
                    axes[idx].set_ylim([0, 1])
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=150)
                plt.close()
                logger.info("  Created: performance_comparison.png")
            
            # Plot 3: Variable distribution comparisons
            logger.info("  Creating variable distribution comparison plots...")
            common_features = self._find_common_numeric_features(datasets, max_features=20)
            
            logger.info(f"  Number of common features found: {len(common_features)}")
            if common_features:
                logger.info(f"  Features: {common_features[:10]}..." if len(common_features) > 10 else f"  Features: {common_features}")
            
            if common_features:
                logger.info(f"  Found {len(common_features)} common numeric features for comparison")
                
                # Create subplot grid
                n_features = len(common_features)
                n_cols = 4
                n_rows = max(1, (n_features + n_cols - 1) // n_cols)
                
                logger.info(f"  Creating grid: {n_rows} rows x {n_cols} cols for {n_features} features")
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
                
                # Handle different axes shapes
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes.flatten()
                elif n_cols == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()
                
                for idx, feature in enumerate(common_features):
                    ax = axes[idx]
                    
                    # Plot distributions for each dataset
                    for label, data in datasets.items():
                        if feature in data.columns:
                            feature_data = data[feature].dropna()
                            if len(feature_data) > 0:
                                try:
                                    ax.hist(feature_data, alpha=0.5, label=label, bins=30, density=True)
                                except Exception as e:
                                    logger.warning(f"    Could not plot {feature} for {label}: {e}")
                    
                    ax.set_xlabel(feature, fontsize=9)
                    ax.set_ylabel('Density', fontsize=9)
                    ax.set_title(f'{feature}', fontsize=10)
                    ax.legend(fontsize=8)
                    ax.tick_params(labelsize=8)
                    ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for idx in range(len(common_features), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                output_path = os.path.join(output_dir, 'variable_distributions.png')
                plt.savefig(output_path, dpi=150)
                plt.close()
                logger.info(f"  Created: variable_distributions.png at {output_path}")
                plt.close()
                logger.info("  Created: variable_distributions.png")
            else:
                logger.warning("  No common numeric features found for distribution comparison")
            
            # Plot 4: AUC scan across MAGGIC thresholds for survival time prediction
            logger.info("  Creating AUC scan plot...")
            self._create_auc_scan_plot(datasets, output_dir)
            
        except ImportError as ie:
            logger.warning(f"Required library not available, skipping plot generation: {ie}")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            import traceback
            traceback.print_exc()


def evaluate_score_performance(input_files, input_labels, outcome_column, 
                               output_file, score_column='maggic (1-year risk of death)',
                               output_plots_dir=None):
    """
    Standalone function to evaluate score performance without Luigi.
    
    Parameters
    ----------
    input_files : list of str
        List of file paths with calculated scores
    input_labels : list of str
        List of labels for each input file
    outcome_column : str
        Name of the binary outcome column
    output_file : str
        Path to save the evaluation report
    score_column : str, optional
        Name of the score column to evaluate
    output_plots_dir : str, optional
        Directory to save evaluation plots
    
    Returns
    -------
    dict
        Evaluation results
    """
    if isinstance(input_files, list):
        input_files = ','.join(input_files)
    if isinstance(input_labels, list):
        input_labels = ','.join(input_labels)
    
    task = EvaluateScorePerformance(
        input_files=input_files,
        input_labels=input_labels,
        outcome_column=outcome_column,
        score_column=score_column,
        output_file=output_file,
        output_plots_dir=output_plots_dir or ''
    )
    
    task.run()
    
    # Load and return results
    with open(output_file, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    luigi.run()
