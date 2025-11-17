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
        
        # Handle timedelta columns
        for col in original_data.columns:
            if original_data[col].dtype.kind == 'm':
                original_data[col] = original_data[col].dt.days
                
        # Load synthetic data
        synthetic_data_path = os.path.join(outdir, f"synthdata_{synth_type}_{num_points}.parquet")
        synthetic_data = pd.read_parquet(synthetic_data_path)
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


def evaluate_privacy(original_data, synthetic_data, plots_dir=None):
    """
    Comprehensive privacy evaluation using multiple frameworks.
    
    Args:
        original_data (pd.DataFrame): Original dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        plots_dir (str, optional): Directory to save plots
    
    Returns:
        dict: Privacy metrics and analysis results
    """
    
    results = {}
    
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
                real_data=original_data,
                synthetic_data=synthetic_data,
                metadata=None  # Can be enhanced with metadata
            )
            results['sdmetrics_new_row_synthesis'] = float(nrs_score)
            logger.info(f"New Row Synthesis Score: {nrs_score:.4f} (higher is better, indicates novelty)")
        except Exception as e:
            logger.warning(f"New Row Synthesis calculation failed: {e}")
            results['sdmetrics_new_row_synthesis'] = None
        
        # Categorical CAP (privacy risk for categorical columns)
        try:
            categorical_cols = original_data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                cap_scores = {}
                for col in categorical_cols:
                    try:
                        cap_score = CategoricalCAP.compute(
                            real_data=original_data,
                            synthetic_data=synthetic_data,
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
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                lr_scores = {}
                for col in numeric_cols:
                    try:
                        lr_score = NumericalLR.compute(
                            real_data=original_data,
                            synthetic_data=synthetic_data,
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
    # 2. Anonymeter Privacy Metrics
    # ==========================================
    logger.info("="*60)
    logger.info("ANONYMETER PRIVACY EVALUATION")
    logger.info("="*60)
    
    try:
        from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator
        
        logger.info("Calculating Anonymeter privacy metrics...")
        
        # Prepare data splits for Anonymeter
        n_attacks = min(2000, len(original_data) // 4)
        control_size = min(1000, len(synthetic_data) // 4)
        
        # Singling Out Attack
        try:
            logger.info("Running Singling Out attack...")
            singling_evaluator = SinglingOutEvaluator(
                ori=original_data,
                syn=synthetic_data,
                control=synthetic_data.sample(n=control_size, random_state=42),
                n_attacks=n_attacks
            )
            singling_evaluator.evaluate(mode='univariate')
            singling_risk = singling_evaluator.risk()
            
            results['anonymeter_singling_out'] = {
                'attack_rate': float(singling_risk.attack_rate),
                'baseline_rate': float(singling_risk.baseline_rate),
                'control_rate': float(singling_risk.control_rate),
                'risk_score': float(singling_risk.attack_rate - singling_risk.baseline_rate)
            }
            logger.info(f"Singling Out Attack Rate: {singling_risk.attack_rate:.4f}")
            logger.info(f"Singling Out Baseline Rate: {singling_risk.baseline_rate:.4f}")
            logger.info(f"Singling Out Risk Score: {results['anonymeter_singling_out']['risk_score']:.4f}")
        except Exception as e:
            logger.warning(f"Singling Out evaluation failed: {e}")
            results['anonymeter_singling_out'] = {'error': str(e)}
        
        # Linkability Attack
        try:
            logger.info("Running Linkability attack...")
            # Select numeric columns for linkability
            numeric_cols = list(original_data.select_dtypes(include=[np.number]).columns)
            if len(numeric_cols) >= 2:
                aux_cols = [numeric_cols[0]]
                if len(numeric_cols) > 1:
                    aux_cols.append(numeric_cols[1])
                
                linkability_evaluator = LinkabilityEvaluator(
                    ori=original_data,
                    syn=synthetic_data,
                    control=synthetic_data.sample(n=control_size, random_state=42),
                    n_attacks=min(n_attacks, 1000),
                    aux_cols=aux_cols
                )
                linkability_evaluator.evaluate(n_neighbors=10)
                linkability_risk = linkability_evaluator.risk()
                
                results['anonymeter_linkability'] = {
                    'attack_rate': float(linkability_risk.attack_rate),
                    'baseline_rate': float(linkability_risk.baseline_rate),
                    'control_rate': float(linkability_risk.control_rate),
                    'risk_score': float(linkability_risk.attack_rate - linkability_risk.baseline_rate)
                }
                logger.info(f"Linkability Attack Rate: {linkability_risk.attack_rate:.4f}")
                logger.info(f"Linkability Baseline Rate: {linkability_risk.baseline_rate:.4f}")
                logger.info(f"Linkability Risk Score: {results['anonymeter_linkability']['risk_score']:.4f}")
            else:
                logger.info("Not enough numeric columns for linkability evaluation")
                results['anonymeter_linkability'] = {'error': 'Insufficient numeric columns'}
        except Exception as e:
            logger.warning(f"Linkability evaluation failed: {e}")
            results['anonymeter_linkability'] = {'error': str(e)}
        
        # Inference Attack
        try:
            logger.info("Running Inference attack...")
            numeric_cols = list(original_data.select_dtypes(include=[np.number]).columns)
            if len(numeric_cols) >= 2:
                # Use first column as secret, others as auxiliary
                secret = numeric_cols[0]
                aux_cols = numeric_cols[1:min(4, len(numeric_cols))]  # Limit to 3 aux columns
                
                inference_evaluator = InferenceEvaluator(
                    ori=original_data,
                    syn=synthetic_data,
                    control=synthetic_data.sample(n=control_size, random_state=42),
                    n_attacks=min(n_attacks, 1000),
                    aux_cols=aux_cols,
                    secret=secret
                )
                inference_evaluator.evaluate()
                inference_risk = inference_evaluator.risk()
                
                results['anonymeter_inference'] = {
                    'attack_rate': float(inference_risk.attack_rate),
                    'baseline_rate': float(inference_risk.baseline_rate),
                    'control_rate': float(inference_risk.control_rate),
                    'risk_score': float(inference_risk.attack_rate - inference_risk.baseline_rate)
                }
                logger.info(f"Inference Attack Rate: {inference_risk.attack_rate:.4f}")
                logger.info(f"Inference Baseline Rate: {inference_risk.baseline_rate:.4f}")
                logger.info(f"Inference Risk Score: {results['anonymeter_inference']['risk_score']:.4f}")
            else:
                logger.info("Not enough numeric columns for inference evaluation")
                results['anonymeter_inference'] = {'error': 'Insufficient numeric columns'}
        except Exception as e:
            logger.warning(f"Inference evaluation failed: {e}")
            results['anonymeter_inference'] = {'error': str(e)}
            
    except ImportError as e:
        logger.warning(f"Anonymeter not available: {e}")
        results['anonymeter_error'] = str(e)
    
    # ==========================================
    # 3. Synthpop Distance to Closest Record (DiSCO)
    # ==========================================
    logger.info("="*60)
    logger.info("DISCO (DISTANCE TO CLOSEST RECORD)")
    logger.info("="*60)
    
    try:
        from synthpop import DiSCO
        
        logger.info("Calculating DiSCO metric...")
        
        # Select numeric columns for distance calculation
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            original_numeric = original_data[numeric_cols].fillna(original_data[numeric_cols].mean())
            synthetic_numeric = synthetic_data[numeric_cols].fillna(synthetic_data[numeric_cols].mean())
            
            # Normalize the data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            original_scaled = scaler.fit_transform(original_numeric)
            synthetic_scaled = scaler.transform(synthetic_numeric)
            
            disco_evaluator = DiSCO()
            disco_results = disco_evaluator.compute(
                original_scaled,
                synthetic_scaled
            )
            
            results['disco'] = {
                'mean_distance': float(np.mean(disco_results)),
                'median_distance': float(np.median(disco_results)),
                'min_distance': float(np.min(disco_results)),
                'max_distance': float(np.max(disco_results)),
                'std_distance': float(np.std(disco_results))
            }
            
            logger.info(f"DiSCO Mean Distance: {results['disco']['mean_distance']:.4f}")
            logger.info(f"DiSCO Median Distance: {results['disco']['median_distance']:.4f}")
            
            # Plot DiSCO distribution
            if plots_dir:
                plt.figure(figsize=(10, 6))
                plt.hist(disco_results, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                plt.xlabel('Distance to Closest Record')
                plt.ylabel('Frequency')
                plt.title('DiSCO: Distribution of Distances to Closest Original Record')
                plt.axvline(results['disco']['mean_distance'], color='red', linestyle='--', 
                           label=f"Mean: {results['disco']['mean_distance']:.4f}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'disco_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
        else:
            logger.info("No numeric columns available for DiSCO calculation")
            results['disco'] = {'error': 'No numeric columns'}
            
    except ImportError as e:
        logger.warning(f"Synthpop not available: {e}")
        results['disco_error'] = str(e)
    except Exception as e:
        logger.warning(f"DiSCO calculation failed: {e}")
        results['disco'] = {'error': str(e)}
    
    # ==========================================
    # 4. RepU (Representativeness/Utility)
    # ==========================================
    logger.info("="*60)
    logger.info("REPU (REPRESENTATIVENESS)")
    logger.info("="*60)
    
    try:
        # Calculate RepU as distribution similarity measure
        logger.info("Calculating RepU metric...")
        
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            repu_scores = {}
            
            from scipy.stats import wasserstein_distance, ks_2samp
            
            for col in numeric_cols:
                original_col = original_data[col].dropna()
                synthetic_col = synthetic_data[col].dropna()
                
                if len(original_col) > 0 and len(synthetic_col) > 0:
                    # Wasserstein distance (Earth Mover's Distance)
                    wd = wasserstein_distance(original_col, synthetic_col)
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pval = ks_2samp(original_col, synthetic_col)
                    
                    repu_scores[col] = {
                        'wasserstein_distance': float(wd),
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_pval)
                    }
                    logger.info(f"RepU ({col}) - Wasserstein: {wd:.4f}, KS: {ks_stat:.4f}")
            
            results['repu'] = repu_scores
            
            # Calculate average RepU score
            if repu_scores:
                avg_wd = np.mean([s['wasserstein_distance'] for s in repu_scores.values()])
                avg_ks = np.mean([s['ks_statistic'] for s in repu_scores.values()])
                results['repu_summary'] = {
                    'avg_wasserstein_distance': float(avg_wd),
                    'avg_ks_statistic': float(avg_ks)
                }
                logger.info(f"Average RepU Wasserstein Distance: {avg_wd:.4f}")
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
        
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Prepare data
            original_numeric = original_data[numeric_cols].fillna(original_data[numeric_cols].mean())
            synthetic_numeric = synthetic_data[numeric_cols].fillna(synthetic_data[numeric_cols].mean())
            
            # Normalize
            scaler = StandardScaler()
            original_scaled = scaler.fit_transform(original_numeric)
            synthetic_scaled = scaler.transform(synthetic_numeric)
            
            # For each synthetic record, find nearest neighbor in original data
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(original_scaled)
            distances, indices = nbrs.kneighbors(synthetic_scaled)
            
            # Calculate MIA metrics
            distances_flat = distances.flatten()
            threshold = np.percentile(distances_flat, 10)  # 10th percentile as threshold
            
            mia_positive = np.sum(distances_flat < threshold)
            mia_rate = mia_positive / len(distances_flat)
            
            results['membership_inference'] = {
                'mean_distance': float(np.mean(distances_flat)),
                'median_distance': float(np.median(distances_flat)),
                'min_distance': float(np.min(distances_flat)),
                'threshold': float(threshold),
                'potential_members': int(mia_positive),
                'membership_rate': float(mia_rate)
            }
            
            logger.info(f"MIA Mean Distance: {results['membership_inference']['mean_distance']:.4f}")
            logger.info(f"MIA Membership Rate: {mia_rate:.4f}")
            logger.info(f"Potential Members: {mia_positive}/{len(distances_flat)}")
            
            # Plot MIA results
            if plots_dir:
                plt.figure(figsize=(10, 6))
                plt.hist(distances_flat, bins=50, alpha=0.7, color='coral', edgecolor='black')
                plt.axvline(threshold, color='red', linestyle='--', 
                           label=f'Threshold (10th percentile): {threshold:.4f}')
                plt.xlabel('Distance to Nearest Original Record')
                plt.ylabel('Frequency')
                plt.title('Membership Inference Attack: Distance Distribution')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'mia_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
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
        numeric_cols = list(original_data.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(original_data.select_dtypes(include=['object', 'category']).columns)
        
        if len(numeric_cols) > 0:
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler
            
            # Use subset of columns as quasi-identifiers
            n_qi = min(3, len(numeric_cols))  # Use up to 3 columns as quasi-identifiers
            quasi_identifiers = numeric_cols[:n_qi]
            
            # Prepare data with only quasi-identifiers
            original_qi = original_data[quasi_identifiers].fillna(original_data[quasi_identifiers].mean())
            synthetic_qi = synthetic_data[quasi_identifiers].fillna(synthetic_data[quasi_identifiers].mean())
            
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
                    original_sensitive = original_data[col].values
                    synthetic_sensitive = synthetic_data[col].values
                    
                    matched_values = original_sensitive[indices.flatten()]
                    
                    # Calculate mean absolute error
                    mae = np.mean(np.abs(synthetic_sensitive - matched_values))
                    
                    # Calculate relative error
                    original_range = original_data[col].max() - original_data[col].min()
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
            
            if 'anonymeter_singling_out' in results and 'risk_score' in results['anonymeter_singling_out']:
                risk_scores['Singling Out\n(Anonymeter)'] = results['anonymeter_singling_out']['risk_score']
            
            if 'anonymeter_linkability' in results and 'risk_score' in results['anonymeter_linkability']:
                risk_scores['Linkability\n(Anonymeter)'] = results['anonymeter_linkability']['risk_score']
            
            if 'anonymeter_inference' in results and 'risk_score' in results['anonymeter_inference']:
                risk_scores['Inference\n(Anonymeter)'] = results['anonymeter_inference']['risk_score']
            
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
