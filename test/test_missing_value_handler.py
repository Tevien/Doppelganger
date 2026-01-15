#!/usr/bin/env python3
"""
Unit tests for missing value handler

Tests the MissingValueHandler class and related utility functions.

Author: SB
Date: 2026-01-06
"""

import pytest
import pandas as pd
import numpy as np
from dpplgngr.utils.missing_value_handler import (
    MissingValueHandler,
    prepare_data_for_metrics,
    selective_fill_for_metric,
    fillna_for_metrics
)


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values for testing."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'numeric_1': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
        'numeric_2': [10.0, np.nan, 30.0, np.nan, 50.0, 60.0, 70.0, 80.0, np.nan, 100.0],
        'categorical': ['A', 'B', np.nan, 'A', 'B', 'A', np.nan, 'B', 'A', 'B'],
        'no_missing': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    
    return data


class TestMissingValueHandler:
    """Test suite for MissingValueHandler class."""
    
    def test_initialization(self):
        """Test handler initialization."""
        handler = MissingValueHandler(strategy='conservative')
        assert handler.strategy == 'conservative'
        assert handler.fill_values_ == {}
    
    def test_fit_conservative_numeric(self, sample_data_with_missing):
        """Test fitting with conservative strategy on numeric data."""
        handler = MissingValueHandler(strategy='conservative')
        handler.fit(sample_data_with_missing)
        
        # Conservative should use median
        expected_numeric_1 = sample_data_with_missing['numeric_1'].median()
        assert handler.fill_values_['numeric_1'] == expected_numeric_1
        
    def test_fit_mean_numeric(self, sample_data_with_missing):
        """Test fitting with mean strategy on numeric data."""
        handler = MissingValueHandler(strategy='mean')
        handler.fit(sample_data_with_missing)
        
        expected_numeric_1 = sample_data_with_missing['numeric_1'].mean()
        assert handler.fill_values_['numeric_1'] == expected_numeric_1
    
    def test_fit_categorical(self, sample_data_with_missing):
        """Test fitting with categorical data."""
        handler = MissingValueHandler(strategy='conservative')
        handler.fit(sample_data_with_missing)
        
        # Should use mode for categorical
        mode = sample_data_with_missing['categorical'].mode()[0]
        assert handler.fill_values_['categorical'] == mode
    
    def test_transform(self, sample_data_with_missing):
        """Test transform method."""
        handler = MissingValueHandler(strategy='conservative')
        handler.fit(sample_data_with_missing)
        
        # Transform should fill missing values
        original_nan_count = sample_data_with_missing.isna().sum().sum()
        assert original_nan_count > 0  # Ensure we have missing values
        
        transformed = handler.transform(sample_data_with_missing)
        
        # Should have no missing values after transform
        assert transformed.isna().sum().sum() == 0
        
        # Original should be unchanged
        assert sample_data_with_missing.isna().sum().sum() == original_nan_count
    
    def test_transform_inplace(self, sample_data_with_missing):
        """Test transform with inplace=True."""
        handler = MissingValueHandler(strategy='conservative')
        handler.fit(sample_data_with_missing)
        
        # Make a copy to test inplace
        data_copy = sample_data_with_missing.copy()
        original_nan_count = data_copy.isna().sum().sum()
        
        handler.transform(data_copy, inplace=True)
        
        # Should be modified inplace
        assert data_copy.isna().sum().sum() == 0
    
    def test_fit_transform(self, sample_data_with_missing):
        """Test fit_transform method."""
        handler = MissingValueHandler(strategy='conservative')
        
        original_nan_count = sample_data_with_missing.isna().sum().sum()
        transformed = handler.fit_transform(sample_data_with_missing)
        
        # Should have no missing values
        assert transformed.isna().sum().sum() == 0
        
        # Original should be unchanged
        assert sample_data_with_missing.isna().sum().sum() == original_nan_count
        
        # Handler should be fitted
        assert len(handler.fill_values_) > 0
    
    def test_get_missing_stats(self, sample_data_with_missing):
        """Test get_missing_stats method."""
        handler = MissingValueHandler(strategy='conservative')
        handler.fit(sample_data_with_missing)
        
        stats = handler.get_missing_stats(sample_data_with_missing)
        
        # Should return a DataFrame
        assert isinstance(stats, pd.DataFrame)
        
        # Should have expected columns
        expected_cols = ['column', 'missing_count', 'missing_pct', 'dtype', 'fill_value']
        for col in expected_cols:
            assert col in stats.columns
        
        # Should have row for each column
        assert len(stats) == len(sample_data_with_missing.columns)
    
    def test_all_missing_column(self):
        """Test handling of column with all missing values."""
        data = pd.DataFrame({
            'all_missing': [np.nan, np.nan, np.nan],
            'numeric': [1.0, 2.0, 3.0]
        })
        
        handler = MissingValueHandler(strategy='conservative')
        handler.fit(data)
        
        # Should handle all missing gracefully
        assert 'all_missing' in handler.fill_values_
        transformed = handler.transform(data)
        assert transformed.isna().sum().sum() == 0


class TestPrepareDataForMetrics:
    """Test suite for prepare_data_for_metrics function."""
    
    def test_basic_usage(self, sample_data_with_missing):
        """Test basic usage of prepare_data_for_metrics."""
        # Create two datasets with missing values
        original = sample_data_with_missing.copy()
        synthetic = sample_data_with_missing.copy()
        
        original_filled, synthetic_filled, fill_info = prepare_data_for_metrics(
            original, synthetic, strategy='conservative'
        )
        
        # Both should have no missing values
        assert original_filled.isna().sum().sum() == 0
        assert synthetic_filled.isna().sum().sum() == 0
        
        # Original data should be unchanged
        assert original.isna().sum().sum() > 0
        assert synthetic.isna().sum().sum() > 0
        
        # Fill info should contain expected keys
        assert 'strategy' in fill_info
        assert 'fill_values' in fill_info
        assert fill_info['strategy'] == 'conservative'
    
    def test_no_missing_values(self):
        """Test with data that has no missing values."""
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'C', 'D', 'E']
        })
        
        original_filled, synthetic_filled, fill_info = prepare_data_for_metrics(
            data, data, strategy='conservative'
        )
        
        # Should handle gracefully
        assert original_filled.equals(data)
        assert 'No missing values' in fill_info['message']
    
    def test_consistency_between_datasets(self, sample_data_with_missing):
        """Test that same fill values are used for both datasets."""
        original = sample_data_with_missing.copy()
        # Create synthetic with different missing pattern
        synthetic = sample_data_with_missing.copy()
        synthetic.loc[5:7, 'numeric_1'] = np.nan
        
        original_filled, synthetic_filled, fill_info = prepare_data_for_metrics(
            original, synthetic, strategy='conservative'
        )
        
        # Both should be filled using same values (learned from original)
        fill_values = fill_info['fill_values']
        
        # Check that fill values were applied consistently
        for col in fill_values:
            if pd.api.types.is_numeric_dtype(original[col]):
                # For numeric, filled values should match the fill value where missing
                original_was_missing = original[col].isna()
                if original_was_missing.any():
                    assert all(original_filled.loc[original_was_missing, col] == fill_values[col])


class TestSelectiveFill:
    """Test suite for selective_fill_for_metric function."""
    
    def test_selective_fill(self, sample_data_with_missing):
        """Test selective filling of specific columns."""
        columns_to_fill = ['numeric_1', 'numeric_2']
        
        filled = selective_fill_for_metric(
            sample_data_with_missing,
            columns=columns_to_fill,
            strategy='conservative'
        )
        
        # Specified columns should have no missing values
        assert filled['numeric_1'].isna().sum() == 0
        assert filled['numeric_2'].isna().sum() == 0
        
        # Other columns should still have missing values
        original_categorical_missing = sample_data_with_missing['categorical'].isna().sum()
        assert filled['categorical'].isna().sum() == original_categorical_missing


class TestFillnaForMetrics:
    """Test suite for fillna_for_metrics convenience function."""
    
    def test_fillna_for_metrics(self, sample_data_with_missing):
        """Test the convenience function."""
        filled = fillna_for_metrics(sample_data_with_missing, strategy='conservative')
        
        # Should have no missing values
        assert filled.isna().sum().sum() == 0
        
        # Original should be unchanged
        assert sample_data_with_missing.isna().sum().sum() > 0


def test_integration_example():
    """Integration test simulating real usage."""
    np.random.seed(42)
    
    # Create realistic sample data
    n = 100
    original = pd.DataFrame({
        'age': np.random.normal(65, 15, n),
        'bmi': np.random.normal(27, 5, n),
        'creatinine': np.random.lognormal(0, 0.3, n),
        'diagnosis': np.random.choice(['A', 'B', 'C'], n)
    })
    
    # Introduce missing values
    original.loc[np.random.random(n) < 0.1, 'bmi'] = np.nan
    original.loc[np.random.random(n) < 0.05, 'creatinine'] = np.nan
    
    # Simulate synthetic data
    synthetic = original.copy()
    synthetic.loc[np.random.random(n) < 0.12, 'bmi'] = np.nan
    
    # Prepare for metrics
    original_filled, synthetic_filled, fill_info = prepare_data_for_metrics(
        original, synthetic, strategy='conservative'
    )
    
    # Verify results
    assert original_filled.isna().sum().sum() == 0
    assert synthetic_filled.isna().sum().sum() == 0
    assert original.isna().sum().sum() > 0  # Original preserved
    
    # Verify we can calculate metrics
    from scipy.stats import wasserstein_distance
    for col in ['age', 'bmi', 'creatinine']:
        wd = wasserstein_distance(original_filled[col], synthetic_filled[col])
        assert wd >= 0  # Valid metric


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
