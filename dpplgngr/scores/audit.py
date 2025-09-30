import pandas as pd
import numpy as np
import seaborn as sns
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata
import warnings
import json
import luigi
import os
import pickle

import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Import the SDV generation step as a dependency
from dpplgngr.train.sdv import SDVGen

import logging
logger = logging.getLogger('luigi-interface')

# meta data
__author__ = 'SB'
__date__ = '2025-09-26'


class SyntheticDataAudit(luigi.Task):
    """Luigi task to audit synthetic data quality against original data."""
    
    gen_config = luigi.Parameter(default="config/synth.json")
    etl_config = luigi.Parameter(default="config/etl.json")
    
    def output(self):
        """Define output files for the audit task."""
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
            'audit_report': luigi.LocalTarget(os.path.join(outdir, f'audit_report_{synth_type}.json')),
            'audit_plots': luigi.LocalTarget(os.path.join(outdir, f'audit_plots_{synth_type}'))
        }
    
    def requires(self):
        """Depend on the synthetic data generation task."""
        return SDVGen(gen_config=self.gen_config, etl_config=self.etl_config)
    
    def run(self):
        """Execute the audit process."""
        logger.info("Starting synthetic data audit...")
        
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
        original_data = original_data[original_data["BMI"]<100]
        original_data = original_data[cols]
        
        # Handle timedelta columns
        for col in original_data.columns:
            if original_data[col].dtype.kind == 'm':
                original_data[col] = original_data[col].dt.days
                
        # Load synthetic data
        synthetic_data_path = os.path.join(outdir, f"synthdata_{synth_type}_{num_points}.parquet")
        synthetic_data = pd.read_parquet(synthetic_data_path)
        
        # Load metadata
        metadata_path = self.input().path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            metadata = SingleTableMetadata.load_from_json(metadata_path)
        else:
            metadata = None
            logger.warning("Metadata file not found, proceeding without metadata")
        
        # Create plots directory
        plots_dir = self.output()['audit_plots'].path
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Run audit and capture results
        audit_results = audit_synthetic_data(
            original_data, 
            synthetic_data, 
            metadata=metadata,
            plots_dir=plots_dir
        )
        
        # Save audit report
        with open(self.output()['audit_report'].path, 'w') as f:
            # Convert numpy objects to Python types for JSON serialization
            audit_results_serializable = {}
            for key, value in audit_results.items():
                if isinstance(value, pd.DataFrame):
                    audit_results_serializable[key] = value.to_dict()
                elif hasattr(value, 'to_dict'):
                    audit_results_serializable[key] = value.to_dict()
                elif isinstance(value, np.ndarray):
                    audit_results_serializable[key] = value.tolist()
                else:
                    try:
                        json.dumps(value)  # Test if serializable
                        audit_results_serializable[key] = value
                    except (TypeError, ValueError):
                        audit_results_serializable[key] = str(value)  # Convert to string as fallback
            
            json.dump(audit_results_serializable, f, indent=2)
        
        logger.info(f"Audit completed. Quality score: {audit_results['quality_score']:.3f}")
        logger.info(f"Audit report saved to: {self.output()['audit_report'].path}")
        logger.info(f"Audit plots saved to: {plots_dir}")


def audit_synthetic_data(original_data, synthetic_data, metadata=None, plots_dir=None):
    """
    Compare synthetic data quality against original data using SDV metrics.
    
    Args:
        original_data (pd.DataFrame): Original dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        metadata (dict, optional): SDV metadata for the dataset
        plots_dir (str, optional): Directory to save plots. If None, plots are shown interactively.
    
    Returns:
        dict: Quality metrics and analysis results
    """
    
    # 1. SDV Quality Evaluation
    logger.info("Evaluating synthetic data quality...")
    try:
        quality_report = evaluate_quality(
            real_data=original_data,
            synthetic_data=synthetic_data,
            metadata=metadata
        )
        
        quality_score = quality_report.get_score()
        logger.info(f"Overall Quality Score: {quality_score:.3f}")
        logger.info("Detailed Quality Metrics:")
        
        # Get details for each property
        quality_details = {}
        try:
            # Try to get all available properties
            properties = quality_report.get_properties()
            for prop in properties:
                prop_details = quality_report.get_details(prop)
                quality_details[prop] = prop_details
                logger.info(f"{prop}: {prop_details}")
        except Exception as e:
            logger.warning(f"Could not get detailed quality metrics: {e}")
            quality_details = {"error": str(e)}
            
    except Exception as e:
        logger.error(f"Quality evaluation failed: {e}")
        quality_score = 0.0
        quality_details = {"error": str(e)}
    
    # 2. Missing Data Analysis
    logger.info("="*50)
    logger.info("MISSING DATA ANALYSIS")
    logger.info("="*50)
    
    original_missing = original_data.isnull().sum()
    synthetic_missing = synthetic_data.isnull().sum()
    
    missing_comparison = pd.DataFrame({
        'Original_Missing': original_missing,
        'Original_Missing_Pct': (original_missing / len(original_data)) * 100,
        'Synthetic_Missing': synthetic_missing,
        'Synthetic_Missing_Pct': (synthetic_missing / len(synthetic_data)) * 100
    })
    
    logger.info(f"Missing data comparison:\n{missing_comparison}")
    
    # Plot missing data comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original data missing values
    original_data.isnull().sum().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Missing Values - Original Data')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Synthetic data missing values
    synthetic_data.isnull().sum().plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('Missing Values - Synthetic Data')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if plots_dir:
        try:
            # Ensure plots directory exists
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, 'missing_values_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to save plot to {plots_dir}: {e}")
            plt.close()
    else:
        plt.show()
    
    # 3. Distribution Comparison
    logger.info("="*50)
    logger.info("DISTRIBUTION ANALYSIS")
    logger.info("="*50)
    
    # Get numeric and categorical columns
    numeric_cols = original_data.select_dtypes(include=[np.number]).columns
    categorical_cols = original_data.select_dtypes(include=['object', 'category']).columns
    
    logger.info(f"Found {len(numeric_cols)} numeric columns: {list(numeric_cols)}")
    logger.info(f"Found {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
    
    # Plot distributions for numeric columns
    if len(numeric_cols) > 0:
        logger.info("Generating numeric distribution plots...")
        fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(12, 4*len(numeric_cols)))
        if len(numeric_cols) == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(numeric_cols):
            # Original distribution
            axes[i, 0].hist(original_data[col].dropna(), bins=30, alpha=0.7, 
                           color='skyblue', label='Original')
            axes[i, 0].set_title(f'{col} - Original Data')
            axes[i, 0].set_ylabel('Frequency')
            
            # Synthetic distribution
            axes[i, 1].hist(synthetic_data[col].dropna(), bins=30, alpha=0.7, 
                           color='lightcoral', label='Synthetic')
            axes[i, 1].set_title(f'{col} - Synthetic Data')
            axes[i, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        if plots_dir:
            try:
                os.makedirs(plots_dir, exist_ok=True)
                plt.savefig(os.path.join(plots_dir, 'numeric_distributions.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"Failed to save numeric distributions plot: {e}")
                plt.close()
        else:
            plt.show()
    
    # Plot distributions for categorical columns
    if len(categorical_cols) > 0:
        logger.info(f"Generating categorical distribution plots for first {min(5, len(categorical_cols))} columns...")
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Original distribution
            original_data[col].value_counts().plot(kind='bar', ax=axes[0], 
                                                  color='skyblue', alpha=0.7)
            axes[0].set_title(f'{col} - Original Data')
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Synthetic distribution
            synthetic_data[col].value_counts().plot(kind='bar', ax=axes[1], 
                                                   color='lightcoral', alpha=0.7)
            axes[1].set_title(f'{col} - Synthetic Data')
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            if plots_dir:
                try:
                    os.makedirs(plots_dir, exist_ok=True)
                    plt.savefig(os.path.join(plots_dir, f'categorical_distribution_{col}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    logger.warning(f"Failed to save categorical plot for {col}: {e}")
                    plt.close()
            else:
                plt.show()
    
    # 4. Statistical Summary Comparison
    logger.info("="*50)
    logger.info("STATISTICAL SUMMARY COMPARISON")
    logger.info("="*50)
    
    # Initialize stats variables
    original_stats = None
    synthetic_stats = None
    if len(numeric_cols) > 0:
        logger.info("Computing statistical summaries...")
        original_stats = original_data[numeric_cols].describe()
        synthetic_stats = synthetic_data[numeric_cols].describe()
        synthetic_stats = synthetic_data[numeric_cols].describe()
        
        logger.info("Original Data Statistics:")
        logger.info(f"\n{original_stats}")
        logger.info("Synthetic Data Statistics:")
        logger.info(f"\n{synthetic_stats}")
        
        # Calculate differences
        logger.info("Statistical Differences (Synthetic - Original):")
        stats_diff = synthetic_stats - original_stats
    # 5. Correlation Analysis
    if len(numeric_cols) > 1:
        logger.info("="*50)
        logger.info("CORRELATION ANALYSIS")
        logger.info("="*50)
        logger.info("Generating correlation matrices...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original correlation matrix
        original_corr = original_data[numeric_cols].corr()
        sns.heatmap(original_corr, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[0], fmt='.2f')
        axes[0].set_title('Original Data - Correlation Matrix')
        
        # Synthetic correlation matrix
        synthetic_corr = synthetic_data[numeric_cols].corr()
        sns.heatmap(synthetic_corr, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1], fmt='.2f')
        axes[1].set_title('Synthetic Data - Correlation Matrix')
        
        plt.tight_layout()
        if plots_dir:
            try:
                os.makedirs(plots_dir, exist_ok=True)
                plt.savefig(os.path.join(plots_dir, 'correlation_matrices.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"Failed to save correlation matrices plot: {e}")
                plt.close()
        else:
            plt.show()
        
        # Correlation difference
        corr_diff = synthetic_corr - original_corr
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_diff, annot=True, cmap='RdBu', center=0, fmt='.3f')
        plt.title('Correlation Difference (Synthetic - Original)')
        plt.tight_layout()
        if plots_dir:
            try:
                os.makedirs(plots_dir, exist_ok=True)
                plt.savefig(os.path.join(plots_dir, 'correlation_difference.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"Failed to save correlation difference plot: {e}")
                plt.close()
        else:
            plt.show()

    
    return {
        'quality_score': quality_score,
        'quality_details': quality_details,
        'missing_data_comparison': missing_comparison,
        'original_stats': original_stats,
        'synthetic_stats': synthetic_stats
    }

# Example usage:
# results = audit_synthetic_data(original_df, synthetic_df, metadata)