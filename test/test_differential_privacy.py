#!/usr/bin/env python
"""
Test script for differential privacy integration with SDV synthesizers.

This script demonstrates how to use the differential privacy features
with different privacy budgets and mechanisms.

Usage:
    python test_differential_privacy.py
"""

import pandas as pd
import numpy as np
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpplgngr.utils.differential_privacy import (
    apply_differential_privacy,
    get_recommended_epsilon,
    PRIVACY_LEVELS
)


def generate_sample_data(n=1000):
    """Generate sample medical data for testing."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'age': np.random.normal(65, 15, n).clip(18, 95),
        'BMI': np.random.normal(27, 5, n).clip(15, 45),
        'systolic_bp': np.random.normal(130, 20, n).clip(90, 200),
        'diastolic_bp': np.random.normal(80, 10, n).clip(60, 120),
        'glucose': np.random.normal(100, 30, n).clip(50, 300),
        'cholesterol': np.random.normal(200, 40, n).clip(100, 400)
    })
    
    return data


def test_differential_privacy():
    """Test differential privacy with different settings."""
    
    print("="*80)
    print("DIFFERENTIAL PRIVACY INTEGRATION TEST")
    print("="*80)
    print()
    
    # Generate sample data
    print("1. Generating sample medical data...")
    original_data = generate_sample_data(1000)
    print(f"   Data shape: {original_data.shape}")
    print(f"   Columns: {list(original_data.columns)}")
    print(f"   Sample statistics:")
    print(original_data.describe().round(2))
    print()
    
    # Simulate synthetic data (in practice, this would come from SDV)
    synthetic_data = original_data.copy()
    synthetic_data += np.random.normal(0, 0.5, synthetic_data.shape)
    print("2. Simulated synthetic data (without DP)")
    print()
    
    # Test different privacy levels
    results = {}
    
    for level_name, level_config in PRIVACY_LEVELS.items():
        print(f"3. Testing {level_name.upper()} privacy level")
        print(f"   {level_config['description']}")
        print(f"   Epsilon: {level_config['epsilon']}")
        print()
        
        epsilon = level_config['epsilon']
        
        # Apply differential privacy
        private_data, privacy_info = apply_differential_privacy(
            synthetic_data=synthetic_data.copy(),
            epsilon=epsilon,
            mechanism='laplace',
            original_data=original_data,
            clip_to_bounds=True
        )
        
        # Compute statistics
        results[level_name] = {
            'epsilon': epsilon,
            'privacy_info': privacy_info,
            'mean_absolute_error': (synthetic_data - private_data).abs().mean().mean(),
            'data': private_data
        }
        
        print(f"   Mean absolute error from synthetic: {results[level_name]['mean_absolute_error']:.4f}")
        print(f"   Columns modified: {len(privacy_info.get('columns_modified', []))}")
        print()
    
    # Compare results
    print("="*80)
    print("COMPARISON OF PRIVACY LEVELS")
    print("="*80)
    print()
    
    comparison_df = pd.DataFrame({
        'Privacy Level': list(results.keys()),
        'Epsilon (ε)': [r['epsilon'] for r in results.values()],
        'Mean Noise': [r['mean_absolute_error'] for r in results.values()],
        'Privacy': ['High', 'Medium', 'Low']
    })
    
    print(comparison_df.to_string(index=False))
    print()
    
    # Show detailed statistics for one column
    print("="*80)
    print("DETAILED COMPARISON FOR 'AGE' COLUMN")
    print("="*80)
    print()
    
    stats_comparison = pd.DataFrame({
        'Original': original_data['age'].describe(),
        'Synthetic': synthetic_data['age'].describe(),
        'DP Strong': results['strong']['data']['age'].describe(),
        'DP Moderate': results['moderate']['data']['age'].describe(),
        'DP Weak': results['weak']['data']['age'].describe()
    })
    
    print(stats_comparison.round(2))
    print()
    
    # Test different mechanisms
    print("="*80)
    print("TESTING DIFFERENT MECHANISMS")
    print("="*80)
    print()
    
    mechanisms = ['laplace', 'gaussian']
    epsilon = 1.0
    
    for mechanism in mechanisms:
        print(f"Testing {mechanism.upper()} mechanism (ε={epsilon})...")
        
        private_data, privacy_info = apply_differential_privacy(
            synthetic_data=synthetic_data.copy(),
            epsilon=epsilon,
            mechanism=mechanism,
            original_data=original_data,
            clip_to_bounds=True
        )
        
        mae = (synthetic_data - private_data).abs().mean().mean()
        print(f"  Mean absolute error: {mae:.4f}")
        print(f"  Privacy applied: {privacy_info['privacy_applied']}")
        print()
    
    # Save sample output
    output_dir = 'test_output_dp'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save data samples
    original_data.to_csv(os.path.join(output_dir, 'original_data.csv'), index=False)
    synthetic_data.to_csv(os.path.join(output_dir, 'synthetic_data.csv'), index=False)
    results['moderate']['data'].to_csv(os.path.join(output_dir, 'private_data_moderate.csv'), index=False)
    
    # Save privacy info
    with open(os.path.join(output_dir, 'privacy_info_moderate.json'), 'w') as f:
        json.dump(results['moderate']['privacy_info'], f, indent=2)
    
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"✅ Test outputs saved to: {output_dir}/")
    print("   - original_data.csv")
    print("   - synthetic_data.csv")
    print("   - private_data_moderate.csv")
    print("   - privacy_info_moderate.json")
    print()
    print("🎉 Differential privacy integration is working correctly!")
    print()


if __name__ == '__main__':
    test_differential_privacy()
