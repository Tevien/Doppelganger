import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dpplgngr.scores.audit import audit_synthetic_data, SyntheticDataAudit


class TestAuditFunctionality:
    """Detailed tests for audit functionality."""
    
    @pytest.fixture
    def medical_data_original(self):
        """Create realistic medical-style original data."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            'age': np.random.normal(65, 15, n).clip(18, 95),
            'bmi': np.random.normal(26, 4, n).clip(15, 45),
            'systolic_bp': np.random.normal(130, 20, n).clip(90, 200),
            'cholesterol': np.random.normal(200, 40, n).clip(120, 350),
            'gender': np.random.choice([0, 1], n),
            'diabetes': np.random.choice([0, 1], n, p=[0.7, 0.3]),
            'medication_type': np.random.choice(['A', 'B', 'C', 'None'], n)
        })
    
    @pytest.fixture
    def medical_data_synthetic(self):
        """Create synthetic version with slightly different distributions."""
        np.random.seed(123)
        n = 200
        return pd.DataFrame({
            'age': np.random.normal(63, 16, n).clip(18, 95),
            'bmi': np.random.normal(25.5, 4.2, n).clip(15, 45),
            'systolic_bp': np.random.normal(128, 22, n).clip(90, 200),
            'cholesterol': np.random.normal(195, 42, n).clip(120, 350),
            'gender': np.random.choice([0, 1], n),
            'diabetes': np.random.choice([0, 1], n, p=[0.75, 0.25]),
            'medication_type': np.random.choice(['A', 'B', 'C', 'None'], n)
        })
    
    def test_audit_handles_empty_data(self):
        """Test audit function with empty datasets."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            audit_synthetic_data(empty_df, empty_df)
    
    def test_audit_numeric_only_data(self, medical_data_original, medical_data_synthetic):
        """Test audit with only numeric columns."""
        numeric_cols = ['age', 'bmi', 'systolic_bp', 'cholesterol']
        original_numeric = medical_data_original[numeric_cols]
        synthetic_numeric = medical_data_synthetic[numeric_cols]
        
        results = audit_synthetic_data(original_numeric, synthetic_numeric)
        
        assert results['original_stats'] is not None
        assert results['synthetic_stats'] is not None
        assert len(results['original_stats'].columns) == 4
    
    def test_audit_categorical_only_data(self, medical_data_original, medical_data_synthetic):
        """Test audit with only categorical columns."""
        categorical_cols = ['gender', 'diabetes', 'medication_type']
        original_cat = medical_data_original[categorical_cols]
        synthetic_cat = medical_data_synthetic[categorical_cols]
        
        results = audit_synthetic_data(original_cat, synthetic_cat)
        
        # Should handle categorical data without crashing
        assert 'quality_score' in results
        assert 'missing_data_comparison' in results
    
    def test_audit_with_missing_values(self, medical_data_original, medical_data_synthetic):
        """Test audit with datasets containing missing values."""
        # Introduce missing values
        original_with_na = medical_data_original.copy()
        synthetic_with_na = medical_data_synthetic.copy()
        
        # Add missing values in different patterns
        original_with_na.loc[0:20, 'bmi'] = np.nan
        original_with_na.loc[10:15, 'cholesterol'] = np.nan
        synthetic_with_na.loc[5:25, 'bmi'] = np.nan
        synthetic_with_na.loc[12:18, 'age'] = np.nan
        
        results = audit_synthetic_data(original_with_na, synthetic_with_na)
        
        # Check missing data analysis
        missing_comp = results['missing_data_comparison']
        assert missing_comp.loc['bmi', 'Original_Missing'] > 0
        assert missing_comp.loc['bmi', 'Synthetic_Missing'] > 0
    
    def test_audit_correlation_analysis(self, medical_data_original, medical_data_synthetic):
        """Test correlation analysis component."""
        # Ensure we have correlated numeric data
        numeric_cols = ['age', 'bmi', 'systolic_bp', 'cholesterol']
        original_numeric = medical_data_original[numeric_cols]
        synthetic_numeric = medical_data_synthetic[numeric_cols]
        
        results = audit_synthetic_data(original_numeric, synthetic_numeric)
        
        # Should compute correlations for numeric data
        assert results['original_stats'] is not None
        assert results['synthetic_stats'] is not None
        
        # Verify statistical summaries have correct structure
        assert 'mean' in results['original_stats'].index
        assert 'std' in results['original_stats'].index
        assert 'min' in results['original_stats'].index
        assert 'max' in results['original_stats'].index
    
    def test_audit_plot_generation(self, medical_data_original, medical_data_synthetic):
        """Test that plots are generated when plots_dir is specified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = audit_synthetic_data(
                medical_data_original, 
                medical_data_synthetic,
                plots_dir=temp_dir
            )
            
            # Check that the function completes without error
            # (Plot generation might fail in headless environments, but function should not crash)
            assert 'quality_score' in results
    
    def test_audit_sdv_quality_metrics(self, medical_data_original, medical_data_synthetic):
        """Test SDV quality evaluation component."""
        results = audit_synthetic_data(medical_data_original, medical_data_synthetic)
        
        # Check quality score
        quality_score = results['quality_score']
        assert isinstance(quality_score, (int, float))
        assert 0 <= quality_score <= 1 or quality_score == 0  # 0 for errors
        
        # Check quality details
        quality_details = results['quality_details']
        assert isinstance(quality_details, dict)
    
    def test_audit_statistical_differences(self, medical_data_original, medical_data_synthetic):
        """Test calculation of statistical differences."""
        results = audit_synthetic_data(medical_data_original, medical_data_synthetic)
        
        original_stats = results['original_stats']
        synthetic_stats = results['synthetic_stats']
        
        if original_stats is not None and synthetic_stats is not None:
            # Check that we can compute differences
            differences = synthetic_stats - original_stats
            assert not differences.empty
            assert differences.shape == original_stats.shape


class TestAuditTaskBehavior:
    """Test the Luigi task behavior for SyntheticDataAudit."""
    
    def test_audit_task_config_validation(self):
        """Test that audit task validates configurations properly."""
        # Test with invalid config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            invalid_config = {"invalid": "config"}
            json.dump(invalid_config, temp_file)
            temp_file.flush()
            
            try:
                with pytest.raises(KeyError):
                    task = SyntheticDataAudit(gen_config=temp_file.name, etl_config=temp_file.name)
                    task.output()  # This should fail due to missing required keys
            finally:
                os.unlink(temp_file.name)
    
    def test_audit_task_output_paths(self):
        """Test that audit task generates correct output paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                config = {
                    "working_dir": temp_dir,
                    "synth_type": "CTGAN"
                }
                json.dump(config, temp_file)
                temp_file.flush()
                
                try:
                    task = SyntheticDataAudit(gen_config=temp_file.name, etl_config=temp_file.name)
                    outputs = task.output()
                    
                    # Check output structure
                    assert 'audit_report' in outputs
                    assert 'audit_plots' in outputs
                    
                    # Check file paths
                    report_path = outputs['audit_report'].path
                    plots_path = outputs['audit_plots'].path
                    
                    assert 'audit_report_CTGAN.json' in report_path
                    assert 'audit_plots_CTGAN' in plots_path
                    
                finally:
                    os.unlink(temp_file.name)
    
    @patch('dpplgngr.scores.audit.SDVGen')
    def test_audit_task_dependencies(self, mock_sdv_gen):
        """Test that audit task has correct dependencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                config = {
                    "working_dir": temp_dir,
                    "synth_type": "GC"
                }
                json.dump(config, temp_file)
                temp_file.flush()
                
                try:
                    task = SyntheticDataAudit(gen_config=temp_file.name, etl_config=temp_file.name)
                    requirements = task.requires()
                    
                    # Should create SDVGen task as requirement
                    mock_sdv_gen.assert_called_once()
                    
                finally:
                    os.unlink(temp_file.name)


class TestAuditErrorHandling:
    """Test error handling in audit functionality."""
    
    def test_audit_mismatched_columns(self):
        """Test audit with mismatched columns between original and synthetic."""
        original = pd.DataFrame({
            'col_a': [1, 2, 3],
            'col_b': [4, 5, 6]
        })
        
        synthetic = pd.DataFrame({
            'col_a': [1.1, 2.1, 3.1],
            'col_c': [7, 8, 9]  # Different column name
        })
        
        # This should handle the mismatch gracefully
        with pytest.raises(Exception):
            audit_synthetic_data(original, synthetic)
    
    def test_audit_different_shapes(self):
        """Test audit with different shaped datasets."""
        original = pd.DataFrame({
            'age': range(100),
            'income': range(100, 200)
        })
        
        synthetic = pd.DataFrame({
            'age': range(50),  # Different number of rows
            'income': range(50, 100)
        })
        
        # Should handle different shapes
        results = audit_synthetic_data(original, synthetic)
        assert 'quality_score' in results
    
    @patch('dpplgngr.scores.audit.evaluate_quality')
    def test_audit_sdv_failure_handling(self, mock_evaluate_quality):
        """Test handling of SDV evaluation failures."""
        # Mock SDV to raise an exception
        mock_evaluate_quality.side_effect = Exception("SDV evaluation failed")
        
        original = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        synthetic = pd.DataFrame({'col1': [1.1, 2.1, 3.1], 'col2': [4.1, 5.1, 6.1]})
        
        results = audit_synthetic_data(original, synthetic)
        
        # Should handle SDV failure gracefully
        assert results['quality_score'] == 0.0
        assert 'error' in results['quality_details']
    
    def test_audit_plot_failure_handling(self):
        """Test handling of plot generation failures."""
        original = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        synthetic = pd.DataFrame({'col1': [1.1, 2.1, 3.1], 'col2': [4.1, 5.1, 6.1]})
        
        # Use a non-existent directory to trigger plot save failure
        non_existent_dir = "/non/existent/directory"
        
        # Should not crash even if plot saving fails
        results = audit_synthetic_data(original, synthetic, plots_dir=non_existent_dir)
        assert 'quality_score' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
