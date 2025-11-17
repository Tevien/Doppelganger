"""
Run SyntheticDataAudit with custom input files, bypassing the ETL pipeline.
"""
import luigi
import json
import os
import tempfile
from dpplgngr.scores.audit import SyntheticDataAudit

def run_audit_custom(original_file, synthetic_file, working_dir, synth_type='GC', num_points=None):
    """
    Run audit with custom files.
    
    Args:
        original_file: Path to original parquet file
        synthetic_file: Path to synthetic parquet file
        working_dir: Directory for audit outputs
        synth_type: Type of synthesizer used (GC, CTGAN, etc.)
        num_points: Number of points (if None, will be inferred from synthetic data)
    """
    
    # Create a temporary config file
    temp_config = {
        "input_file": original_file,
        "working_dir": working_dir,
        "synth_type": synth_type,
        "num_points": num_points or 10000,  # Will be overridden by actual data
        "columns": []  # Will be detected from data
    }
    
    # Create temp config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(temp_config, f)
        temp_config_path = f.name
    
    try:
        # Ensure synthetic data file exists in expected location
        expected_synth_path = os.path.join(working_dir, f"synthdata_{synth_type}_{temp_config['num_points']}.parquet")
        
        # If synthetic file is not in expected location, copy it
        if synthetic_file != expected_synth_path:
            import shutil
            os.makedirs(working_dir, exist_ok=True)
            shutil.copy(synthetic_file, expected_synth_path)
            print(f"Copied synthetic data to: {expected_synth_path}")
        
        # Run the audit task
        success = luigi.build([
            SyntheticDataAudit(
                gen_config=temp_config_path,
                etl_config="config/etl.json"  # Not used since we're bypassing
            )
        ], local_scheduler=True)
        
        if success:
            print(f"\n✅ Audit completed successfully!")
            print(f"📊 Audit report: {os.path.join(working_dir, f'audit_report_{synth_type}.json')}")
            print(f"📈 Audit plots: {os.path.join(working_dir, f'audit_plots_{synth_type}')}")
        else:
            print("\n❌ Audit failed!")
            
    finally:
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run synthetic data audit with custom files')
    parser.add_argument('--original', required=True, help='Path to original data parquet file')
    parser.add_argument('--synthetic', required=True, help='Path to synthetic data parquet file')
    parser.add_argument('--working-dir', required=True, help='Working directory for outputs')
    parser.add_argument('--synth-type', default='GC', help='Synthesizer type (GC, CTGAN, TVAE, etc.)')
    parser.add_argument('--num-points', type=int, help='Number of data points (optional)')
    
    args = parser.parse_args()
    
    run_audit_custom(
        original_file=args.original,
        synthetic_file=args.synthetic,
        working_dir=args.working_dir,
        synth_type=args.synth_type,
        num_points=args.num_points
    )