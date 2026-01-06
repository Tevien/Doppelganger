"""
Run privacy evaluation directly without Luigi pipeline.
Standalone script for privacy assessment of synthetic data.
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dpplgngr.scores.privacy import evaluate_privacy


def run_direct_privacy_evaluation(original_file, synthetic_file, output_dir, verbose=True):
    """
    Run privacy evaluation directly on two parquet files.
    
    Args:
        original_file (str): Path to original parquet file
        synthetic_file (str): Path to synthetic parquet file  
        output_dir (str): Directory to save privacy reports and plots
        verbose (bool): Print detailed progress information
    
    Returns:
        dict: Privacy evaluation results
    """
    
    if verbose:
        print("="*70)
        print("DIRECT PRIVACY EVALUATION")
        print("="*70)
        print(f"Original data: {original_file}")
        print(f"Synthetic data: {synthetic_file}")
        print(f"Output directory: {output_dir}")
        print("="*70)
    
    # Load data
    if verbose:
        print("\n📂 Loading data...")
    
    try:
        original_data = pd.read_parquet(original_file)
        if verbose:
            print(f"✓ Original data loaded: {original_data.shape}")
    except Exception as e:
        print(f"❌ Failed to load original data: {e}")
        return None
    
    try:
        synthetic_data = pd.read_parquet(synthetic_file)
        if verbose:
            print(f"✓ Synthetic data loaded: {synthetic_data.shape}")
    except Exception as e:
        print(f"❌ Failed to load synthetic data: {e}")
        return None
    
    # Validate data compatibility
    if set(original_data.columns) != set(synthetic_data.columns):
        print("⚠️  Warning: Column mismatch between original and synthetic data")
        print(f"Original columns: {original_data.columns.tolist()}")
        print(f"Synthetic columns: {synthetic_data.columns.tolist()}")
        
        # Use common columns
        common_cols = list(set(original_data.columns) & set(synthetic_data.columns))
        if len(common_cols) == 0:
            print("❌ No common columns found!")
            return None
        
        print(f"Using {len(common_cols)} common columns")
        original_data = original_data[common_cols]
        synthetic_data = synthetic_data[common_cols]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'privacy_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if verbose:
        print(f"\n📊 Output will be saved to: {output_dir}")
        print(f"📈 Plots will be saved to: {plots_dir}")
    
    # Run privacy evaluation
    if verbose:
        print("\n🔒 Running privacy evaluation...")
        print("This may take several minutes depending on data size...")
    
    try:
        results = evaluate_privacy(
            original_data=original_data,
            synthetic_data=synthetic_data,
            plots_dir=plots_dir
        )
    except Exception as e:
        print(f"❌ Privacy evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Save results to JSON
    results_file = os.path.join(output_dir, 'privacy_report.json')
    
    if verbose:
        print(f"\n💾 Saving results to: {results_file}")
    
    try:
        # Make results JSON serializable
        results_serializable = _make_results_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        if verbose:
            print(f"✓ Results saved successfully")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save JSON results: {e}")
    
    # Print summary
    if verbose:
        print("\n" + "="*70)
        print("PRIVACY EVALUATION SUMMARY")
        print("="*70)
        
        _print_summary(results)
        
        print("\n" + "="*70)
        print("✅ Privacy evaluation completed!")
        print(f"📄 Full report: {results_file}")
        print(f"📊 Visualizations: {plots_dir}")
        print("="*70)
    
    return results


def _make_results_serializable(obj):
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _make_results_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_results_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
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


def _print_summary(results):
    """Print a formatted summary of privacy results."""
    
    # SDMetrics Summary
    print("\n📊 SDMetrics Privacy Metrics:")
    if 'sdmetrics_new_row_synthesis' in results and results['sdmetrics_new_row_synthesis']:
        nrs = results['sdmetrics_new_row_synthesis']
        print(f"  • New Row Synthesis: {nrs:.4f} (higher is better)")
    
    if 'sdmetrics_categorical_cap_avg' in results:
        cap = results['sdmetrics_categorical_cap_avg']
        print(f"  • Categorical CAP (avg): {cap:.4f} (lower is better)")
    
    if 'sdmetrics_numerical_lr_avg' in results:
        lr = results['sdmetrics_numerical_lr_avg']
        print(f"  • Numerical LR (avg): {lr:.4f} (lower is better)")
    
    # Anonymeter Summary
    print("\n🔐 Anonymeter Privacy Attacks:")
    
    if 'anonymeter_singling_out' in results and 'risk_score' in results['anonymeter_singling_out']:
        so = results['anonymeter_singling_out']
        print(f"  • Singling Out Risk: {so['risk_score']:.4f}")
        print(f"    - Attack Rate: {so['attack_rate']:.4f}")
        print(f"    - Baseline: {so['baseline_rate']:.4f}")
    
    if 'anonymeter_linkability' in results and 'risk_score' in results['anonymeter_linkability']:
        link = results['anonymeter_linkability']
        print(f"  • Linkability Risk: {link['risk_score']:.4f}")
        print(f"    - Attack Rate: {link['attack_rate']:.4f}")
        print(f"    - Baseline: {link['baseline_rate']:.4f}")
    
    if 'anonymeter_inference' in results and 'risk_score' in results['anonymeter_inference']:
        inf = results['anonymeter_inference']
        print(f"  • Inference Risk: {inf['risk_score']:.4f}")
        print(f"    - Attack Rate: {inf['attack_rate']:.4f}")
        print(f"    - Baseline: {inf['baseline_rate']:.4f}")
    
    # DiSCO Summary
    if 'disco' in results and 'mean_distance' in results['disco']:
        disco = results['disco']
        print(f"\n📏 DiSCO (Distance to Closest Record):")
        print(f"  • Mean Distance: {disco['mean_distance']:.4f}")
        print(f"  • Median Distance: {disco['median_distance']:.4f}")
        print(f"  • Min Distance: {disco['min_distance']:.4f}")
    
    # RepU Summary
    if 'repu_summary' in results:
        repu = results['repu_summary']
        print(f"\n📐 RepU (Representativeness):")
        print(f"  • Avg Wasserstein Distance: {repu['avg_wasserstein_distance']:.4f}")
        print(f"  • Avg KS Statistic: {repu['avg_ks_statistic']:.4f}")
    
    # Membership Inference Summary
    if 'membership_inference' in results and 'membership_rate' in results['membership_inference']:
        mia = results['membership_inference']
        print(f"\n🎯 Membership Inference Attack:")
        print(f"  • Membership Rate: {mia['membership_rate']:.4f}")
        print(f"  • Potential Members: {mia['potential_members']}")
        print(f"  • Mean Distance: {mia['mean_distance']:.4f}")
    
    # Attribute Disclosure Summary
    if 'attribute_disclosure_summary' in results and 'avg_disclosure_risk' in results['attribute_disclosure_summary']:
        ad = results['attribute_disclosure_summary']
        print(f"\n🔓 Attribute Disclosure Risk:")
        print(f"  • Average Risk: {ad['avg_disclosure_risk']:.4f}")
        print(f"  • Quasi-Identifiers: {', '.join(ad['quasi_identifiers'])}")
        print(f"  • Sensitive Attributes: {', '.join(ad['sensitive_attributes'])}")
    
    # Risk Level Assessment
    print("\n⚠️  Overall Risk Assessment:")
    
    risk_scores = []
    
    if 'anonymeter_singling_out' in results and 'risk_score' in results['anonymeter_singling_out']:
        risk_scores.append(('Singling Out', results['anonymeter_singling_out']['risk_score']))
    
    if 'anonymeter_linkability' in results and 'risk_score' in results['anonymeter_linkability']:
        risk_scores.append(('Linkability', results['anonymeter_linkability']['risk_score']))
    
    if 'anonymeter_inference' in results and 'risk_score' in results['anonymeter_inference']:
        risk_scores.append(('Inference', results['anonymeter_inference']['risk_score']))
    
    if 'membership_inference' in results and 'membership_rate' in results['membership_inference']:
        risk_scores.append(('Membership Inference', results['membership_inference']['membership_rate']))
    
    if 'attribute_disclosure_summary' in results and 'avg_disclosure_risk' in results['attribute_disclosure_summary']:
        risk_scores.append(('Attribute Disclosure', results['attribute_disclosure_summary']['avg_disclosure_risk']))
    
    if risk_scores:
        for name, score in risk_scores:
            risk_level = "🟢 LOW" if score < 0.1 else "🟡 MEDIUM" if score < 0.3 else "🔴 HIGH"
            print(f"  • {name}: {score:.4f} - {risk_level}")
        
        avg_risk = np.mean([s for _, s in risk_scores])
        overall_level = "🟢 LOW" if avg_risk < 0.1 else "🟡 MEDIUM" if avg_risk < 0.3 else "🔴 HIGH"
        print(f"\n  📊 Average Risk Score: {avg_risk:.4f} - {overall_level}")


def main():
    parser = argparse.ArgumentParser(
        description='Direct privacy evaluation for synthetic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/privacy_direct.py \\
    --original data/original.parquet \\
    --synthetic data/synthetic.parquet \\
    --output results/privacy
  
  # Quiet mode
  python scripts/privacy_direct.py \\
    --original data/original.parquet \\
    --synthetic data/synthetic.parquet \\
    --output results/privacy \\
    --quiet
        """
    )
    
    parser.add_argument('--original', required=True, 
                       help='Path to original data parquet file')
    parser.add_argument('--synthetic', required=True,
                       help='Path to synthetic data parquet file')
    parser.add_argument('--output', required=True,
                       help='Output directory for privacy reports and plots')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed progress output')
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.original):
        print(f"❌ Error: Original file not found: {args.original}")
        sys.exit(1)
    
    if not os.path.exists(args.synthetic):
        print(f"❌ Error: Synthetic file not found: {args.synthetic}")
        sys.exit(1)
    
    # Run privacy evaluation
    results = run_direct_privacy_evaluation(
        original_file=args.original,
        synthetic_file=args.synthetic,
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    if results is None:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()