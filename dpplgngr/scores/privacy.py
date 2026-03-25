import pandas as pd
import numpy as np
import warnings
import json
import luigi
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Import the SDV generation step as a dependency
from dpplgngr.train.sdv import SDVGen

# Import missing value handler
from dpplgngr.utils.missing_value_handler import prepare_data_for_metrics

import logging
logger = logging.getLogger('luigi-interface')

# meta data
__author__ = 'SB'
__date__ = '2025-10-28'


class PrivacyEvaluation(luigi.Task):
    """Luigi task to evaluate privacy risks of synthetic data using multiple frameworks."""
    
    gen_config = luigi.Parameter(default="config/synth.json")
    etl_config = luigi.Parameter(default="config/etl.json")
    
    def output(self):
        """Define output files for the privacy evaluation task."""
        try:
            with open(self.gen_config, 'r') as f:
                input_json = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load generation config: {e}")
            raise
            
        outdir = input_json.get('working_dir', None)
        synth_type = input_json.get('synth_type', input_json.get('synthesizer', 'unknown'))
        
        if outdir is None:
            raise KeyError("'working_dir' not found in generation config")
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            
        return {
            'privacy_report': luigi.LocalTarget(os.path.join(outdir, f'privacy_report_{synth_type}.json')),
            'privacy_plots': luigi.LocalTarget(os.path.join(outdir, f'privacy_plots_{synth_type}'))
        }
    
    def requires(self):
        """Depend on the synthetic data generation task."""
        return SDVGen(gen_config=self.gen_config, etl_config=self.etl_config)
    
    def run(self):
        """Execute the privacy evaluation process."""
        logger.info("Starting privacy evaluation...")
        
        # Load configuration
        with open(self.gen_config, 'r') as f:
            input_json = json.load(f)
            
        outdir = input_json.get('working_dir', None)
        synth_type = input_json.get('synth_type', None)
        num_points = int(input_json.get('num_points', None))
        cols = input_json.get('columns', None)
        
        # Load original data
        original_data = pd.read_parquet(input_json['input_file'])
        
        # Apply same preprocessing as in SDVGen
        bmi_col = None
        for col in original_data.columns:
            if 'BMI' in col or col == 'BMI':
                bmi_col = col
                break
        if bmi_col is not None:
            original_data = original_data[pd.to_numeric(original_data[bmi_col], errors='coerce')<100]
        
        original_data = original_data[cols]
        original_data = original_data.reset_index(drop=True)
        
        # Handle timedelta columns — convert to integer days so both datasets share
        # the same scale. Synthetic parquets store timedeltas as raw int64 nanoseconds,
        # so we also rescale any int64 columns whose original counterpart was timedelta.
        timedelta_cols = [col for col in original_data.columns
                          if original_data[col].dtype.kind == 'm']
        for col in timedelta_cols:
            original_data[col] = original_data[col].dt.days

        # Load synthetic data
        synthetic_data_path = os.path.join(outdir, f"synthdata_{synth_type}_{num_points}.parquet")
        synthetic_data = pd.read_parquet(synthetic_data_path)
        synthetic_data = synthetic_data.reset_index(drop=True)

        # Synthetic timedelta columns are stored as int64 nanoseconds by the synthesiser.
        # Convert to days to match original_data.
        NS_PER_DAY = 86_400 * 1_000_000_000
        for col in timedelta_cols:
            if col in synthetic_data.columns and synthetic_data[col].dtype.kind in ('i', 'u', 'f'):
                synthetic_data[col] = synthetic_data[col] / NS_PER_DAY

        # Apply the same BMI filter to synthetic data to remove physiologically
        # implausible outliers that would otherwise dominate t-closeness.
        if bmi_col is not None and bmi_col in synthetic_data.columns:
            synthetic_data = synthetic_data[
                pd.to_numeric(synthetic_data[bmi_col], errors='coerce') < 100
            ]
            synthetic_data = synthetic_data.reset_index(drop=True)
        
        # Create plots directory
        plots_dir = self.output()['privacy_plots'].path
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Run privacy evaluation
        privacy_results = evaluate_privacy(
            original_data, 
            synthetic_data,
            plots_dir=plots_dir
        )
        
        # Save privacy report
        with open(self.output()['privacy_report'].path, 'w') as f:
            # Convert numpy objects to Python types for JSON serialization
            privacy_results_serializable = {}
            for key, value in privacy_results.items():
                if isinstance(value, pd.DataFrame):
                    privacy_results_serializable[key] = value.to_dict()
                elif hasattr(value, 'to_dict'):
                    privacy_results_serializable[key] = value.to_dict()
                elif isinstance(value, np.ndarray):
                    privacy_results_serializable[key] = value.tolist()
                elif isinstance(value, dict):
                    privacy_results_serializable[key] = _make_serializable(value)
                else:
                    try:
                        json.dumps(value)  # Test if serializable
                        privacy_results_serializable[key] = value
                    except (TypeError, ValueError):
                        privacy_results_serializable[key] = str(value)  # Convert to string as fallback
            
            json.dump(privacy_results_serializable, f, indent=2)
        
        logger.info(f"Privacy evaluation completed.")
        logger.info(f"Privacy report saved to: {self.output()['privacy_report'].path}")
        logger.info(f"Privacy plots saved to: {plots_dir}")


def _make_serializable(obj):
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def evaluate_privacy(original_data, synthetic_data, plots_dir=None, handle_missing='conservative'):

    """
    Comprehensive privacy evaluation using multiple frameworks.
    
    Args:
        original_data (pd.DataFrame): Original dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        plots_dir (str, optional): Directory to save plots
        handle_missing (str, optional): Strategy for handling missing values in metrics.
            Options: 'conservative' (median/mode), 'mean', 'none' (no filling).
            Default: 'conservative'
    
    Returns:
        dict: Privacy metrics and analysis results
    """
    
    results = {}
    
    # Store original data shapes and missing value info
    logger.info(f"Original data shape: {original_data.shape}")
    logger.info(f"Synthetic data shape: {synthetic_data.shape}")
    logger.info(f"Original data NaN count: {original_data.isna().sum().sum()}")
    logger.info(f"Synthetic data NaN count: {synthetic_data.isna().sum().sum()}")
    
    # Prepare data for metrics if there are missing values
    if handle_missing and handle_missing != 'none':
        if original_data.isna().any().any() or synthetic_data.isna().any().any():
            logger.info("Handling missing values for privacy metric calculation...")
            original_filled, synthetic_filled, fill_info = prepare_data_for_metrics(
                original_data, synthetic_data, strategy=handle_missing
            )
            results['missing_value_handling'] = fill_info
        else:
            logger.info("No missing values detected, using original data")
            original_filled = original_data
            synthetic_filled = synthetic_data
            fill_info = {'strategy': 'none', 'message': 'No missing values'}
            results['missing_value_handling'] = fill_info
    else:
        logger.info("Missing value handling disabled, using original data")
        original_filled = original_data
        synthetic_filled = synthetic_data
        results['missing_value_handling'] = {'strategy': 'none', 'message': 'Disabled by user'}
    

    # ==========================================
    # 1. SDMetrics Privacy Metrics
    # ==========================================
    logger.info("="*60)
    logger.info("SDMETRICS PRIVACY EVALUATION")
    logger.info("="*60)
    
    try:
        from sdmetrics.reports.single_table import QualityReport
        from sdmetrics.single_table import NewRowSynthesis, CategoricalCAP, NumericalLR
        
        logger.info("Calculating SDMetrics privacy metrics...")
        
        # New Row Synthesis (measures if synthetic rows are novel)
        try:
            nrs_score = NewRowSynthesis.compute(
                real_data=original_filled,
                synthetic_data=synthetic_filled,
                metadata=None  # Can be enhanced with metadata
            )
            results['sdmetrics_new_row_synthesis'] = float(nrs_score)
            logger.info(f"New Row Synthesis Score: {nrs_score:.4f} (higher is better, indicates novelty)")
        except Exception as e:
            logger.warning(f"New Row Synthesis calculation failed: {e}")
            results['sdmetrics_new_row_synthesis'] = None
        
        # Categorical CAP (privacy risk for categorical columns)
        try:
            categorical_cols = original_filled.select_dtypes(include=['object', 'category']).columns

            if len(categorical_cols) > 0:
                cap_scores = {}
                for col in categorical_cols:
                    try:
                        cap_score = CategoricalCAP.compute(
                            real_data=original_filled,
                            synthetic_data=synthetic_filled,
                            key_fields=[col],
                            sensitive_fields=[col]
                        )
                        cap_scores[col] = float(cap_score)
                        logger.info(f"Categorical CAP ({col}): {cap_score:.4f}")
                    except Exception as e:
                        logger.warning(f"CAP calculation failed for {col}: {e}")
                
                results['sdmetrics_categorical_cap'] = cap_scores
                if cap_scores:
                    avg_cap = np.mean(list(cap_scores.values()))
                    results['sdmetrics_categorical_cap_avg'] = float(avg_cap)
                    logger.info(f"Average Categorical CAP: {avg_cap:.4f} (lower is better)")
            else:
                results['sdmetrics_categorical_cap'] = {}
                logger.info("No categorical columns found for CAP evaluation")
        except Exception as e:
            logger.warning(f"Categorical CAP evaluation failed: {e}")
            results['sdmetrics_categorical_cap'] = {}
        
        # Numerical LR (privacy risk for numerical columns)
        try:
            numeric_cols = original_filled.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                lr_scores = {}
                for col in numeric_cols:
                    try:
                        lr_score = NumericalLR.compute(
                            real_data=original_filled,
                            synthetic_data=synthetic_filled,
                            key_fields=[col],
                            sensitive_fields=[col]
                        )
                        lr_scores[col] = float(lr_score)
                        logger.info(f"Numerical LR ({col}): {lr_score:.4f}")
                    except Exception as e:
                        logger.warning(f"LR calculation failed for {col}: {e}")
                
                results['sdmetrics_numerical_lr'] = lr_scores
                if lr_scores:
                    avg_lr = np.mean(list(lr_scores.values()))
                    results['sdmetrics_numerical_lr_avg'] = float(avg_lr)
                    logger.info(f"Average Numerical LR: {avg_lr:.4f} (lower is better)")
            else:
                results['sdmetrics_numerical_lr'] = {}
                logger.info("No numerical columns found for LR evaluation")
        except Exception as e:
            logger.warning(f"Numerical LR evaluation failed: {e}")
            results['sdmetrics_numerical_lr'] = {}
            
    except ImportError as e:
        logger.warning(f"SDMetrics not available: {e}")
        results['sdmetrics_error'] = str(e)
    
    # ==========================================
    # 2. Privacy Metrics (k-anonymity, l-diversity, t-closeness)
    #    Self-contained implementation using pandas/numpy/scipy
    # ==========================================
    logger.info("="*60)
    logger.info("PRIVACY EVALUATION (k-anonymity, l-diversity, t-closeness)")
    logger.info("="*60)
    
    try:
        from scipy.stats import wasserstein_distance as _emd
        
        logger.info("Calculating privacy metrics...")
        
        # Identify quasi-identifiers (QI) and sensitive attributes.
        # Only use low-cardinality (categorical / binary / integer) columns as QIs —
        # continuous float columns like AGEATOPNAME create singleton equivalence classes
        # which make k-anonymity trivially 1 and inflate t-closeness.
        def _is_low_cardinality(series, max_unique=50):
            if series.dtype.kind in ('O', 'b'):
                return True
            if series.dtype.kind in ('i', 'u'):
                return series.nunique() <= max_unique
            return False  # float columns excluded

        candidate_qi = [col for col in original_data.columns if _is_low_cardinality(original_data[col])]
        n_qi = min(4, len(candidate_qi))
        quasi_identifiers = candidate_qi[:n_qi]

        # Fall back to first columns if no low-cardinality candidates found
        if not quasi_identifiers:
            quasi_identifiers = list(original_data.columns[:min(4, len(original_data.columns) - 1)])

        sensitive_attrs = [col for col in original_data.columns if col not in quasi_identifiers]
        if not sensitive_attrs:
            sensitive_attrs = [quasi_identifiers.pop()]

        logger.info(f"Quasi-identifiers: {quasi_identifiers}")
        logger.info(f"Sensitive attributes: {sensitive_attrs}")
        
        # --- k-Anonymity ---
        # k = minimum equivalence class size (grouped by QI columns)
        def _k_anonymity(df, qi_cols):
            return int(df.groupby(qi_cols, dropna=False).size().min())
        
        for label, df in [('original', original_data), ('synthetic', synthetic_data)]:
            try:
                k = _k_anonymity(df, quasi_identifiers)
                results[f'k_anonymity_{label}'] = k
                logger.info(f"{label.title()} Data k-anonymity: {k}")
            except Exception as e:
                logger.warning(f"k-anonymity for {label} failed: {e}")
                results[f'k_anonymity_{label}'] = {'error': str(e)}
        
        # --- l-Diversity ---
        # l = minimum number of distinct sensitive values across all equivalence classes
        def _l_diversity(df, qi_cols, sensitive_col):
            grouped = df.groupby(qi_cols, dropna=False)[sensitive_col]
            return int(grouped.nunique().min())
        
        if sensitive_attrs:
            for label, df in [('original', original_data), ('synthetic', synthetic_data)]:
                try:
                    l_val = _l_diversity(df, quasi_identifiers, sensitive_attrs[0])
                    results[f'l_diversity_{label}'] = l_val
                    logger.info(f"{label.title()} Data l-diversity: {l_val}")
                except Exception as e:
                    logger.warning(f"l-diversity for {label} failed: {e}")
                    results[f'l_diversity_{label}'] = {'error': str(e)}
        
        # --- t-Closeness ---
        # t = max normalised Earth Mover's Distance between the sensitive attribute
        # distribution within any equivalence class and the overall distribution.
        # Following Li et al. (2007): for numerical attributes the EMD is normalised
        # by the range of the sensitive attribute so t is dimensionless in [0, 1].
        # For categorical attributes the EMD is normalised by the number of distinct
        # values (max possible EMD between two integer-coded distributions).
        def _t_closeness(df, qi_cols, sensitive_col):
            overall = df[sensitive_col]
            # For categorical data, convert to numeric codes for EMD
            if overall.dtype == 'object' or str(overall.dtype) == 'category':
                codes_map = {v: i for i, v in enumerate(overall.unique())}
                overall_vals = overall.map(codes_map).values.astype(float)
                n_vals = max(len(codes_map) - 1, 1)  # normalisation denominator
                grouped = df.groupby(qi_cols, dropna=False)
                max_t = 0.0
                for _, group in grouped:
                    group_vals = group[sensitive_col].map(codes_map).values.astype(float)
                    if len(group_vals) > 0:
                        max_t = max(max_t, _emd(overall_vals, group_vals) / n_vals)
                return max_t
            else:
                overall_vals = overall.dropna().values.astype(float)
                attr_range = float(overall_vals.max() - overall_vals.min())
                normaliser = attr_range if attr_range > 0 else 1.0
                grouped = df.groupby(qi_cols, dropna=False)
                max_t = 0.0
                for _, group in grouped:
                    group_vals = group[sensitive_col].dropna().values.astype(float)
                    if len(group_vals) > 0:
                        max_t = max(max_t, _emd(overall_vals, group_vals) / normaliser)
                return max_t
        
        if sensitive_attrs:
            for label, df in [('original', original_data), ('synthetic', synthetic_data)]:
                try:
                    t_val = _t_closeness(df, quasi_identifiers, sensitive_attrs[0])
                    results[f't_closeness_{label}'] = float(t_val)
                    logger.info(f"{label.title()} Data t-closeness: {t_val:.4f}")
                except Exception as e:
                    logger.warning(f"t-closeness for {label} failed: {e}")
                    results[f't_closeness_{label}'] = {'error': str(e)}
        
        # --- Equivalence class statistics ---
        try:
            original_ec_sizes = original_data.groupby(quasi_identifiers, dropna=False).size()
            synthetic_ec_sizes = synthetic_data.groupby(quasi_identifiers, dropna=False).size()
            
            results['ec_stats'] = {
                'original_mean_ec_size': float(original_ec_sizes.mean()),
                'original_min_ec_size': int(original_ec_sizes.min()),
                'original_max_ec_size': int(original_ec_sizes.max()),
                'synthetic_mean_ec_size': float(synthetic_ec_sizes.mean()),
                'synthetic_min_ec_size': int(synthetic_ec_sizes.min()),
                'synthetic_max_ec_size': int(synthetic_ec_sizes.max()),
                'n_equivalence_classes_original': int(len(original_ec_sizes)),
                'n_equivalence_classes_synthetic': int(len(synthetic_ec_sizes))
            }
            
            logger.info(f"Original: {len(original_ec_sizes)} equivalence classes, "
                       f"mean size: {original_ec_sizes.mean():.2f}")
            logger.info(f"Synthetic: {len(synthetic_ec_sizes)} equivalence classes, "
                       f"mean size: {synthetic_ec_sizes.mean():.2f}")
        except Exception as e:
            logger.warning(f"Equivalence class calculation failed: {e}")
            results['ec_stats'] = {'error': str(e)}
        
        # --- Summary Privacy Score ---
        try:
            privacy_score = 0
            n_metrics = 0
            
            if isinstance(results.get('k_anonymity_synthetic'), int):
                privacy_score += min(results['k_anonymity_synthetic'] / 10.0, 1.0)
                n_metrics += 1
            
            if isinstance(results.get('l_diversity_synthetic'), int):
                privacy_score += min(results['l_diversity_synthetic'] / 5.0, 1.0)
                n_metrics += 1
            
            if isinstance(results.get('t_closeness_synthetic'), float):
                privacy_score += max(1.0 - results['t_closeness_synthetic'] / 0.2, 0.0)
                n_metrics += 1
            
            if n_metrics > 0:
                avg_privacy_score = privacy_score / n_metrics
                results['privacy_score'] = float(avg_privacy_score)
                logger.info(f"Overall Privacy Score: {avg_privacy_score:.4f} (0=low, 1=high)")
            
        except Exception as e:
            logger.warning(f"Privacy score calculation failed: {e}")
            
    except ImportError as e:
        logger.warning(f"scipy not available for privacy metrics: {e}")
        results['privacy_metrics_error'] = str(e)
    except Exception as e:
        logger.warning(f"Privacy metrics calculation failed: {e}")
        results['privacy_metrics_error'] = str(e)
    
    # ==========================================
    # 3. Distance to Closest Record (DCR)
    #    Self-contained implementation using sklearn NearestNeighbors
    # ==========================================
    logger.info("="*60)
    logger.info("DCR (DISTANCE TO CLOSEST RECORD)")
    logger.info("="*60)
    
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        
        logger.info("Calculating DCR metric...")
        
        numeric_cols = original_filled.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            original_numeric = original_filled[numeric_cols]
            synthetic_numeric = synthetic_filled[numeric_cols]
            
            # Normalize the data
            scaler = StandardScaler()
            original_scaled = scaler.fit_transform(original_numeric)
            synthetic_scaled = scaler.transform(synthetic_numeric)
            
            # For each synthetic record, find distance to its nearest original record
            nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean')
            nn.fit(original_scaled)
            distances, _ = nn.kneighbors(synthetic_scaled)
            dcr_distances = distances.flatten()
            
            results['dcr'] = {
                'mean_distance': float(np.mean(dcr_distances)),
                'median_distance': float(np.median(dcr_distances)),
                'min_distance': float(np.min(dcr_distances)),
                'max_distance': float(np.max(dcr_distances)),
                'std_distance': float(np.std(dcr_distances)),
                'pct_below_5th_percentile': float(
                    np.mean(dcr_distances < np.percentile(dcr_distances, 5)) * 100
                )
            }
            
            logger.info(f"DCR Mean Distance: {results['dcr']['mean_distance']:.4f}")
            logger.info(f"DCR Median Distance: {results['dcr']['median_distance']:.4f}")
            logger.info(f"DCR Min Distance: {results['dcr']['min_distance']:.4f}")
            
            # Plot DCR distribution
            if plots_dir:
                plt.figure(figsize=(10, 6))
                plt.hist(dcr_distances, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                plt.xlabel('Distance to Closest Record')
                plt.ylabel('Frequency')
                plt.title('DCR: Distribution of Distances to Closest Original Record')
                plt.axvline(results['dcr']['mean_distance'], color='red', linestyle='--', 
                           label=f"Mean: {results['dcr']['mean_distance']:.4f}")
                plt.axvline(results['dcr']['median_distance'], color='orange', linestyle='--',
                           label=f"Median: {results['dcr']['median_distance']:.4f}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'dcr_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
        else:
            logger.info("No numeric columns available for DCR calculation")
            results['dcr'] = {'error': 'No numeric columns'}
            
    except ImportError as e:
        logger.warning(f"sklearn not available for DCR: {e}")
        results['dcr_error'] = str(e)
    except Exception as e:
        logger.warning(f"DCR calculation failed: {e}")
        results['dcr'] = {'error': str(e)}
    
    # ==========================================
    # 4. RepU (Representativeness/Utility)
    # ==========================================
    logger.info("="*60)
    logger.info("REPU (REPRESENTATIVENESS)")
    logger.info("="*60)
    
    try:
        # Calculate RepU as distribution similarity measure
        logger.info("Calculating RepU metric...")
        
        numeric_cols = original_filled.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            repu_scores = {}
            
            from scipy.stats import wasserstein_distance, ks_2samp
            
            for col in numeric_cols:
                # Data is already filled, no need to dropna
                original_col = original_filled[col]
                synthetic_col = synthetic_filled[col]

                if len(original_col) > 0 and len(synthetic_col) > 0:
                    # Wasserstein distance (Earth Mover's Distance) — raw units
                    wd = wasserstein_distance(original_col, synthetic_col)

                    # Normalised Wasserstein: divide both series by the original std so
                    # the distance is dimensionless and comparable across columns with
                    # different scales (e.g., days vs mmHg vs mg/dL).
                    std = float(original_col.std())
                    if std > 0:
                        wd_norm = wasserstein_distance(
                            original_col / std, synthetic_col / std
                        )
                    else:
                        wd_norm = 0.0

                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pval = ks_2samp(original_col, synthetic_col)

                    repu_scores[col] = {
                        'wasserstein_distance': float(wd),
                        'wasserstein_distance_normalised': float(wd_norm),
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_pval)
                    }
                    logger.info(f"RepU ({col}) - Wasserstein: {wd:.4f} (norm: {wd_norm:.4f}), KS: {ks_stat:.4f}")

            results['repu'] = repu_scores

            # Calculate average RepU score
            if repu_scores:
                avg_wd = np.mean([s['wasserstein_distance'] for s in repu_scores.values()])
                avg_wd_norm = np.mean([s['wasserstein_distance_normalised'] for s in repu_scores.values()])
                avg_ks = np.mean([s['ks_statistic'] for s in repu_scores.values()])
                results['repu_summary'] = {
                    'avg_wasserstein_distance': float(avg_wd),
                    'avg_wasserstein_distance_normalised': float(avg_wd_norm),
                    'avg_ks_statistic': float(avg_ks)
                }
                logger.info(f"Average RepU Wasserstein Distance: {avg_wd:.4f} (norm: {avg_wd_norm:.4f})")
                logger.info(f"Average RepU KS Statistic: {avg_ks:.4f}")
        else:
            logger.info("No numeric columns available for RepU calculation")
            results['repu'] = {'error': 'No numeric columns'}
            
    except Exception as e:
        logger.warning(f"RepU calculation failed: {e}")
        results['repu'] = {'error': str(e)}
    
    # ==========================================
    # 5. Membership Inference Attack (MIA)
    # ==========================================
    logger.info("="*60)
    logger.info("MEMBERSHIP INFERENCE ATTACK")
    logger.info("="*60)
    
    try:
        logger.info("Running Membership Inference Attack...")
        
        # Simple MIA using nearest neighbor approach
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        
        numeric_cols = original_filled.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Data is already filled, no need to fillna again
            original_numeric = original_filled[numeric_cols]
            synthetic_numeric = synthetic_filled[numeric_cols]
            
            # Normalize
            scaler = StandardScaler()
            original_scaled = scaler.fit_transform(original_numeric)
            synthetic_scaled = scaler.transform(synthetic_numeric)
            
            # For each synthetic record, find nearest neighbor in original data
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(original_scaled)
            distances, indices = nbrs.kneighbors(synthetic_scaled)

            # Threshold: 5th percentile of real-to-real leave-one-out NN distances.
            # A synthetic record is a potential member only if it is closer to a real
            # record than 95% of real records are to their own nearest neighbour.
            # This scales automatically with dimensionality and dataset size.
            nbrs_real = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(original_scaled)
            real_nn_dists, _ = nbrs_real.kneighbors(original_scaled)
            real_nn_dists = real_nn_dists[:, 1]  # skip self (distance=0)
            threshold = float(np.percentile(real_nn_dists, 5))
            logger.info(f"MIA threshold (5th pct real-to-real NN dist): {threshold:.4f}")

            # Calculate MIA metrics
            distances_flat = distances.flatten()
            mia_positive = np.sum(distances_flat < threshold)
            mia_rate = mia_positive / len(distances_flat)

            results['membership_inference'] = {
                'mean_distance': float(np.mean(distances_flat)),
                'median_distance': float(np.median(distances_flat)),
                'min_distance': float(np.min(distances_flat)),
                'threshold': threshold,
                'potential_members': int(mia_positive),
                'membership_rate': float(mia_rate)
            }

            logger.info(f"MIA Mean Distance: {results['membership_inference']['mean_distance']:.4f}")
            logger.info(f"MIA Membership Rate: {mia_rate:.4f}")
            logger.info(f"Potential Members: {mia_positive}/{len(distances_flat)}")

            # Plot MIA results
            if plots_dir:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                axes[0].hist(distances_flat, bins=50, alpha=0.7, color='coral', edgecolor='black')
                axes[0].axvline(threshold, color='red', linestyle='--',
                                label=f'Threshold (5th pct real-NN): {threshold:.4f}')
                axes[0].set_xlabel('Distance to Nearest Original Record')
                axes[0].set_ylabel('Frequency')
                axes[0].set_title('Synthetic → Real NN distances')
                axes[0].legend()

                axes[1].hist(real_nn_dists, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                axes[1].axvline(threshold, color='red', linestyle='--',
                                label=f'Threshold: {threshold:.4f}')
                axes[1].set_xlabel('Distance to Nearest Original Record (leave-one-out)')
                axes[1].set_ylabel('Frequency')
                axes[1].set_title('Real → Real NN distances (baseline)')
                axes[1].legend()

                fig.tight_layout()
                fig.savefig(os.path.join(plots_dir, 'mia_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close(fig)
        else:
            logger.info("No numeric columns available for MIA")
            results['membership_inference'] = {'error': 'No numeric columns'}
            
    except Exception as e:
        logger.warning(f"Membership Inference Attack failed: {e}")
        results['membership_inference'] = {'error': str(e)}
    
    # ==========================================
    # 6. Attribute Disclosure Risk
    # ==========================================
    logger.info("="*60)
    logger.info("ATTRIBUTE DISCLOSURE RISK")
    logger.info("="*60)
    
    try:
        logger.info("Calculating Attribute Disclosure Risk...")
        
        # For each record in synthetic data, find closest match in original
        # Then check how many attributes can be inferred
        numeric_cols = list(original_filled.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(original_filled.select_dtypes(include=['object', 'category']).columns)
        
        if len(numeric_cols) > 0:
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler
            
            # Use subset of columns as quasi-identifiers
            n_qi = min(3, len(numeric_cols))  # Use up to 3 columns as quasi-identifiers
            quasi_identifiers = numeric_cols[:n_qi]
            
            # Data is already filled, no need to fillna again
            original_qi = original_filled[quasi_identifiers]
            synthetic_qi = synthetic_filled[quasi_identifiers]
            
            scaler = StandardScaler()
            original_qi_scaled = scaler.fit_transform(original_qi)
            synthetic_qi_scaled = scaler.transform(synthetic_qi)
            
            # Find nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(original_qi_scaled)
            distances, indices = nbrs.kneighbors(synthetic_qi_scaled)
            
            # For sensitive attributes (other columns), check how well they match
            sensitive_cols = [col for col in numeric_cols if col not in quasi_identifiers][:3]
            
            if len(sensitive_cols) > 0:
                disclosure_risks = {}
                
                for col in sensitive_cols:
                    # Use filled data
                    original_sensitive = original_filled[col].values
                    synthetic_sensitive = synthetic_filled[col].values
                    
                    matched_values = original_sensitive[indices.flatten()]
                    
                    # Calculate mean absolute error
                    mae = np.mean(np.abs(synthetic_sensitive - matched_values))
                    
                    # Calculate relative error
                    original_range = original_filled[col].max() - original_filled[col].min()
                    relative_mae = mae / original_range if original_range > 0 else 0
                    
                    disclosure_risks[col] = {
                        'mae': float(mae),
                        'relative_mae': float(relative_mae),
                        'disclosure_risk': float(1 - relative_mae)  # Higher when synthetic matches original
                    }
                    
                    logger.info(f"Attribute Disclosure ({col}) - MAE: {mae:.4f}, Risk: {disclosure_risks[col]['disclosure_risk']:.4f}")
                
                results['attribute_disclosure'] = disclosure_risks
                
                # Calculate average disclosure risk
                avg_risk = np.mean([r['disclosure_risk'] for r in disclosure_risks.values()])
                results['attribute_disclosure_summary'] = {
                    'avg_disclosure_risk': float(avg_risk),
                    'quasi_identifiers': quasi_identifiers,
                    'sensitive_attributes': sensitive_cols
                }
                logger.info(f"Average Attribute Disclosure Risk: {avg_risk:.4f}")
            else:
                logger.info("No sensitive attributes available for disclosure risk calculation")
                results['attribute_disclosure'] = {'error': 'No sensitive attributes'}
        else:
            logger.info("No numeric columns available for Attribute Disclosure calculation")
            results['attribute_disclosure'] = {'error': 'No numeric columns'}
            
    except Exception as e:
        logger.warning(f"Attribute Disclosure calculation failed: {e}")
        results['attribute_disclosure'] = {'error': str(e)}
    
    # ==========================================
    # Summary Visualization
    # ==========================================
    if plots_dir:
        try:
            logger.info("Creating privacy summary visualization...")
            
            # Collect all risk scores
            risk_scores = {}
            
            if 'sdmetrics_new_row_synthesis' in results and results['sdmetrics_new_row_synthesis']:
                risk_scores['New Row Synthesis\n(SDMetrics)'] = results['sdmetrics_new_row_synthesis']
            
            if 'privacy_score' in results:
                risk_scores['Privacy Score\n(k/l/t)'] = results['privacy_score']
            
            if isinstance(results.get('k_anonymity_synthetic'), int):
                # Normalize k-anonymity to 0-1 scale (inverse: higher k = lower risk)
                k_risk = 1.0 - min(results['k_anonymity_synthetic'] / 10.0, 1.0)
                risk_scores['k-Anonymity Risk'] = k_risk
            
            if 'membership_inference' in results and 'membership_rate' in results['membership_inference']:
                risk_scores['Membership\nInference'] = results['membership_inference']['membership_rate']
            
            if 'attribute_disclosure_summary' in results and 'avg_disclosure_risk' in results['attribute_disclosure_summary']:
                risk_scores['Attribute\nDisclosure'] = results['attribute_disclosure_summary']['avg_disclosure_risk']
            
            if risk_scores:
                fig, ax = plt.subplots(figsize=(12, 6))
                colors = ['green' if v < 0.1 else 'orange' if v < 0.3 else 'red' 
                         for v in risk_scores.values()]
                bars = ax.bar(risk_scores.keys(), risk_scores.values(), color=colors, alpha=0.7, edgecolor='black')
                ax.set_ylabel('Risk Score')
                ax.set_title('Privacy Risk Summary Across Multiple Metrics')
                ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Low Risk (<0.1)')
                ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Medium Risk (<0.3)')
                ax.set_ylim(0, max(1.0, max(risk_scores.values()) * 1.1))
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'privacy_summary.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info("Privacy summary visualization saved")
        except Exception as e:
            logger.warning(f"Failed to create privacy summary visualization: {e}")
    
    logger.info("="*60)
    logger.info("PRIVACY EVALUATION COMPLETE")
    logger.info("="*60)
    
    return results


# Example usage:
# results = evaluate_privacy(original_df, synthetic_df, plots_dir='./privacy_plots')
