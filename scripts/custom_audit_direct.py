"""
Run audit directly without Luigi pipeline.
"""
import pandas as pd
from dpplgngr.scores.audit import audit_synthetic_data

def run_direct_audit(original_file, synthetic_file, output_dir):
    """
    Run audit directly on two parquet files.
    
    Args:
        original_file: Path to original parquet
        synthetic_file: Path to synthetic parquet
        output_dir: Directory to save plots
    """
    # Load data
    print(f"Loading original data from: {original_file}")
    original_data = pd.read_parquet(original_file)
    
    print(f"Loading synthetic data from: {synthetic_file}")
    synthetic_data = pd.read_parquet(synthetic_file)
    
    print(f"\nOriginal data shape: {original_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    
    # Run audit
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = audit_synthetic_data(
        original_data=original_data,
        synthetic_data=synthetic_data,
        metadata=None,  # Will be auto-detected
        plots_dir=output_dir
    )
    
    # Save results
    import json
    results_file = os.path.join(output_dir, 'audit_results.json')
    
    # Serialize results
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            results_serializable[key] = value.to_dict()
        else:
            results_serializable[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✅ Audit completed!")
    print(f"Quality Score: {results['quality_score']:.3f}")
    print(f"Results saved to: {results_file}")
    print(f"Plots saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run direct audit without Luigi')
    parser.add_argument('--original', required=True, help='Original data parquet')
    parser.add_argument('--synthetic', required=True, help='Synthetic data parquet')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    run_direct_audit(args.original, args.synthetic, args.output_dir)