#!/usr/bin/env python3
"""
Local Synthesis Runner for Snowflake Pipeline

This script handles the synthesis, audit, and privacy evaluation steps that run
locally on the HPC environment after data has been downloaded from Snowflake.

It's designed to work with data that has been preprocessed in Snowflake and
downloaded to the local filesystem.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_synthesis(etl_config_path, gen_config_path, output_dir):
    """
    Run synthetic data generation using SDVGen.
    
    Args:
        etl_config_path: Path to ETL configuration
        gen_config_path: Path to generation configuration
        output_dir: Directory for output files
    """
    logger.info("=" * 60)
    logger.info("Running Synthetic Data Generation")
    logger.info("=" * 60)
    
    try:
        from dpplgngr.train.sdv import SDVGen
        import dask.dataframe as dd
        import pandas as pd
        
        # Load configurations
        etl_config = load_config(etl_config_path)
        gen_config = load_config(gen_config_path)
        
        # Update paths in gen_config to use output_dir
        gen_config['output_dir'] = output_dir
        
        # Save updated gen_config
        gen_config_temp = os.path.join(output_dir, 'gen_config_temp.json')
        with open(gen_config_temp, 'w') as f:
            json.dump(gen_config, f, indent=2)
        
        logger.info(f"ETL config: {etl_config_path}")
        logger.info(f"Gen config: {gen_config_temp}")
        logger.info(f"Output dir: {output_dir}")
        
        # Load preprocessed data
        preprocessed_file = etl_config.get('preprocessed_file', 
                                          os.path.join(output_dir, 'preprocessed_data.parquet'))
        
        if not os.path.exists(preprocessed_file):
            raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_file}")
        
        logger.info(f"Loading preprocessed data from: {preprocessed_file}")
        
        # For SDVGen, we need to mock the Luigi task structure
        # Create a custom runner that doesn't rely on Luigi dependencies
        logger.info("Initializing SDV generator...")
        
        # Read the preprocessed data
        df = pd.read_parquet(preprocessed_file)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Filter to only the columns specified in gen_config
        gen_columns = gen_config.get('columns', None)
        if gen_columns:
            missing_cols = [c for c in gen_columns if c not in df.columns]
            if missing_cols:
                logger.warning(f"Columns in gen_config not found in data: {missing_cols}")
            available_cols = [c for c in gen_columns if c in df.columns]
            df = df[available_cols]
            logger.info(f"Filtered to {len(available_cols)} columns from gen_config")
        logger.info(f"Data shape for synthesis: {df.shape}")
        
        # Import and configure the generator
        from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
        from sdv.metadata import SingleTableMetadata
        
        # Determine model type from config
        model_type = gen_config.get('model', 'GaussianCopula')
        logger.info(f"Using model: {model_type}")
        
        # Create metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        # Initialize synthesizer based on model type
        if model_type == 'GaussianCopula':
            synthesizer = GaussianCopulaSynthesizer(metadata)
        elif model_type == 'CTGAN':
            synthesizer = CTGANSynthesizer(metadata)
        elif model_type == 'TVAE':
            synthesizer = TVAESynthesizer(metadata)
        else:
            logger.warning(f"Unknown model type: {model_type}, defaulting to GaussianCopula")
            synthesizer = GaussianCopulaSynthesizer(metadata)
        
        # Train the model
        logger.info("Training synthesizer...")
        synthesizer.fit(df)
        
        # Generate synthetic data
        num_samples = gen_config.get('num_samples', len(df))
        logger.info(f"Generating {num_samples} synthetic samples...")
        synthetic_data = synthesizer.sample(num_rows=num_samples)
        
        # Save synthetic data
        output_file = os.path.join(output_dir, 'synthetic_data.parquet')
        synthetic_data.to_parquet(output_file, index=False)
        logger.info(f"Synthetic data saved to: {output_file}")
        logger.info(f"Synthetic data shape: {synthetic_data.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        return False


def run_audit(etl_config_path, gen_config_path, output_dir):
    """
    Run audit evaluation on synthetic data using the full audit pipeline.
    
    Args:
        etl_config_path: Path to ETL configuration
        gen_config_path: Path to generation configuration
        output_dir: Directory with synthetic and real data
    """
    logger.info("=" * 60)
    logger.info("Running Audit Evaluation")
    logger.info("=" * 60)
    
    try:
        from dpplgngr.scores.audit import audit_synthetic_data
        from sdv.metadata import SingleTableMetadata
        import pandas as pd
        import numpy as np
        
        # Load configs
        etl_config = load_config(etl_config_path)
        gen_config = load_config(gen_config_path)
        
        preprocessed_file = etl_config.get('preprocessed_file',
                                          os.path.join(output_dir, 'preprocessed_data.parquet'))
        synthetic_file = os.path.join(output_dir, 'synthetic_data.parquet')
        
        if not os.path.exists(preprocessed_file):
            raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_file}")
        if not os.path.exists(synthetic_file):
            raise FileNotFoundError(f"Synthetic data not found: {synthetic_file}")
        
        logger.info(f"Loading real data from: {preprocessed_file}")
        real_data = pd.read_parquet(preprocessed_file)
        
        logger.info(f"Loading synthetic data from: {synthetic_file}")
        synthetic_data = pd.read_parquet(synthetic_file)
        
        # Filter real_data to only the columns in synthetic_data
        common_cols = [c for c in synthetic_data.columns if c in real_data.columns]
        real_data = real_data[common_cols]
        logger.info(f"Filtered real data to {len(common_cols)} columns matching synthetic data")
        
        # Reset indices to avoid duplicate index issues
        real_data = real_data.reset_index(drop=True)
        synthetic_data = synthetic_data.reset_index(drop=True)
        
        # Handle timedelta columns
        for col in real_data.columns:
            if real_data[col].dtype.kind == 'm':
                real_data[col] = real_data[col].dt.days
        
        # Create metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_data)
        
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'audit_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Run the full audit (distribution plots, correlations, etc.)
        audit_results = audit_synthetic_data(
            real_data,
            synthetic_data,
            metadata=metadata,
            plots_dir=plots_dir
        )
        
        # Save results
        results_serializable = {}
        for key, value in audit_results.items():
            if isinstance(value, pd.DataFrame):
                results_serializable[key] = value.to_dict()
            elif hasattr(value, 'to_dict'):
                results_serializable[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            else:
                results_serializable[key] = value
        
        output_file = os.path.join(output_dir, 'audit_results.json')
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2, default=str)
        
        logger.info(f"Audit results saved to: {output_file}")
        logger.info(f"Audit plots saved to: {plots_dir}")
        logger.info(f"Overall quality score: {audit_results.get('quality_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Audit evaluation failed: {e}", exc_info=True)
        return False


def run_privacy(etl_config_path, gen_config_path, output_dir):
    """
    Run privacy evaluation on synthetic data using the full privacy pipeline.
    
    Args:
        etl_config_path: Path to ETL configuration
        gen_config_path: Path to generation configuration
        output_dir: Directory with synthetic and real data
    """
    logger.info("=" * 60)
    logger.info("Running Privacy Evaluation")
    logger.info("=" * 60)
    
    try:
        from dpplgngr.scores.privacy import evaluate_privacy
        import pandas as pd
        import numpy as np
        
        # Load configs
        etl_config = load_config(etl_config_path)
        gen_config = load_config(gen_config_path)
        
        preprocessed_file = etl_config.get('preprocessed_file',
                                          os.path.join(output_dir, 'preprocessed_data.parquet'))
        synthetic_file = os.path.join(output_dir, 'synthetic_data.parquet')
        
        if not os.path.exists(preprocessed_file):
            raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_file}")
        if not os.path.exists(synthetic_file):
            raise FileNotFoundError(f"Synthetic data not found: {synthetic_file}")
        
        logger.info(f"Loading real data from: {preprocessed_file}")
        real_data = pd.read_parquet(preprocessed_file)
        
        logger.info(f"Loading synthetic data from: {synthetic_file}")
        synthetic_data = pd.read_parquet(synthetic_file)
        
        # Filter real_data to only the columns in synthetic_data
        common_cols = [c for c in synthetic_data.columns if c in real_data.columns]
        real_data = real_data[common_cols]
        logger.info(f"Filtered real data to {len(common_cols)} columns matching synthetic data")
        
        # Reset indices
        real_data = real_data.reset_index(drop=True)
        synthetic_data = synthetic_data.reset_index(drop=True)
        
        # Handle timedelta columns
        for col in real_data.columns:
            if real_data[col].dtype.kind == 'm':
                real_data[col] = real_data[col].dt.days
        
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'privacy_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Run the full privacy evaluation (NewRowSynthesis, k-anonymity,
        # l-diversity, t-closeness, MIA, attribute disclosure, etc.)
        privacy_results = evaluate_privacy(
            real_data,
            synthetic_data,
            plots_dir=plots_dir
        )
        
        # Save results using the privacy module's serializer
        from dpplgngr.scores.privacy import _make_serializable
        results_serializable = _make_serializable(privacy_results)
        
        output_file = os.path.join(output_dir, 'privacy_results.json')
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2, default=str)
        
        logger.info(f"Privacy results saved to: {output_file}")
        logger.info(f"Privacy plots saved to: {plots_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Privacy evaluation failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run synthesis, audit, and privacy evaluation locally'
    )
    parser.add_argument('--etl-config', required=True,
                       help='Path to ETL configuration file')
    parser.add_argument('--gen-config', required=True,
                       help='Path to generation configuration file')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--audit-only', action='store_true',
                       help='Run only audit evaluation')
    parser.add_argument('--privacy-only', action='store_true',
                       help='Run only privacy evaluation')
    parser.add_argument('--skip-synthesis', action='store_true',
                       help='Skip synthesis step')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = True
    
    # Determine which steps to run:
    # --audit-only  -> only audit
    # --privacy-only -> only privacy
    # neither       -> synthesis + audit + privacy (unless --skip-synthesis)
    run_synth = not args.skip_synthesis and not args.audit_only and not args.privacy_only
    run_audit_step = args.audit_only or (not args.audit_only and not args.privacy_only)
    run_privacy_step = args.privacy_only or (not args.audit_only and not args.privacy_only)
    
    if run_synth:
        if not run_synthesis(args.etl_config, args.gen_config, args.output_dir):
            success = False
            logger.error("Synthesis failed")
    
    if run_audit_step:
        if not run_audit(args.etl_config, args.gen_config, args.output_dir):
            logger.warning("Audit evaluation failed (continuing)")
    
    if run_privacy_step:
        if not run_privacy(args.etl_config, args.gen_config, args.output_dir):
            logger.warning("Privacy evaluation failed (continuing)")
    
    if success:
        logger.info("=" * 60)
        logger.info("All requested operations completed successfully")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("=" * 60)
        logger.error("Some operations failed")
        logger.error("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
