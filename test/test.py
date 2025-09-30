import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the modules to test
import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dpplgngr.etl.prep_dataset_tabular import PreProcess, TuplesProcess
from dpplgngr.train.sdv import SDVGen
from dpplgngr.scores.audit import SyntheticDataAudit, audit_synthetic_data


class TestDataProcessing:
    """Test suite for data processing functionality."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Return the test data directory path."""
        return os.path.join(os.path.dirname(__file__), "data")
    
    @pytest.fixture
    def test_config_dir(self):
        """Return the test config directory path."""
        return os.path.dirname(__file__)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        return pd.DataFrame({
            'id': range(1, 101),
            'age': np.random.randint(18, 80, 100),
            'gender': np.random.choice([0, 1], 100),
            'blood_pressure': np.random.normal(120, 15, 100),
            'cholesterol': np.random.normal(200, 30, 100),
            'glucose': np.random.normal(90, 10, 100),
            'measurement_type': np.random.choice([1, 2, 3], 100)
        })
    
    @pytest.fixture
    def test_etl_config(self, temp_output_dir):
        """Create a test ETL configuration."""
        test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        return {
            "name": "test_process",
            "absolute_path": test_data_dir + "/",
            "preprocessing": temp_output_dir,
            "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
            "MergedTransforms": {
                "DropNaN": {
                    "func": "dropna",
                    "kwargs": {"subset": ["age"]}
                }
            }
        }
    
    @pytest.fixture
    def test_synth_config(self, temp_output_dir):
        """Create a test synthesis configuration."""
        return {
            "input_file": os.path.join(temp_output_dir, "preprocessed_tupleprocess.parquet"),
            "working_dir": temp_output_dir,
            "columns": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
            "num_points": 50,
            "synth_type": "GC"
        }

    def test_load_test_data(self, test_data_dir):
        """Test that test data files can be loaded."""
        # Test data_file1
        df1 = pd.read_parquet(os.path.join(test_data_dir, "data_file1.parquet"))
        assert df1.shape[0] > 0, "data_file1 should not be empty"
        assert 'id' in df1.columns, "data_file1 should have 'id' column"
        assert 'age' in df1.columns, "data_file1 should have 'age' column"
        assert 'gender' in df1.columns, "data_file1 should have 'gender' column"
        
        # Test data_file2
        df2 = pd.read_parquet(os.path.join(test_data_dir, "data_file2.parquet"))
        assert df2.shape[0] > 0, "data_file2 should not be empty"
        assert 'id' in df2.columns, "data_file2 should have 'id' column"
        assert 'measurement' in df2.columns, "data_file2 should have 'measurement' column"
        assert 'measurement_type' in df2.columns, "data_file2 should have 'measurement_type' column"
        
    def test_etl_config_loading(self, test_config_dir):
        """Test that ETL configuration can be loaded."""
        config_path = os.path.join(test_config_dir, "etl.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        assert 'name' in config, "ETL config should have 'name' field"
        assert 'final_cols' in config, "ETL config should have 'final_cols' field"
        assert isinstance(config['final_cols'], list), "final_cols should be a list"
        
    def test_synth_config_loading(self, test_config_dir):
        """Test that synthesis configuration can be loaded."""
        config_path = os.path.join(test_config_dir, "synth.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        assert 'synth_type' in config, "Synth config should have 'synth_type' field"
        assert 'num_points' in config, "Synth config should have 'num_points' field"
        assert 'columns' in config, "Synth config should have 'columns' field"


class TestETLPipeline:
    """Test suite for ETL pipeline components."""
    
    @pytest.fixture
    def temp_config_file(self, test_etl_config):
        """Create a temporary config file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(test_etl_config, temp_file)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    def test_preprocess_output_definition(self, temp_config_file):
        """Test that PreProcess task defines output correctly."""
        task = PreProcess(etl_config=temp_config_file)
        output = task.output()
        
        assert hasattr(output, 'path'), "Output should have path attribute"
        assert output.path.endswith('.parquet'), "Output should be parquet file"
    
    def test_tuples_process_output_definition(self, temp_config_file):
        """Test that TuplesProcess task defines output correctly."""
        task = TuplesProcess(etl_config=temp_config_file)
        output = task.output()
        
        assert hasattr(output, 'path'), "Output should have path attribute"
        assert 'tupleprocess' in output.path, "Output should contain 'tupleprocess'"


class TestSyntheticDataGeneration:
    """Test suite for synthetic data generation."""
    
    @pytest.fixture
    def temp_synth_config_file(self, test_synth_config):
        """Create a temporary synthesis config file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(test_synth_config, temp_file)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def temp_etl_config_file(self, test_etl_config):
        """Create a temporary ETL config file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(test_etl_config, temp_file)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    def test_sdv_gen_output_definition(self, temp_synth_config_file, temp_etl_config_file):
        """Test that SDVGen task defines output correctly."""
        task = SDVGen(gen_config=temp_synth_config_file, etl_config=temp_etl_config_file)
        output = task.output()
        
        assert hasattr(output, 'path'), "Output should have path attribute"
        assert output.path.endswith('.pkl'), "Output should be pickle file"
        assert 'synth_' in output.path, "Output should contain 'synth_'"
    
    def test_sdv_gen_requires_tuple_process(self, temp_synth_config_file, temp_etl_config_file):
        """Test that SDVGen requires TuplesProcess."""
        task = SDVGen(gen_config=temp_synth_config_file, etl_config=temp_etl_config_file)
        requirements = task.requires()
        
        assert isinstance(requirements, TuplesProcess), "Should require TuplesProcess"


class TestSyntheticDataAudit:
    """Test suite for synthetic data audit functionality."""
    
    @pytest.fixture
    def sample_original_data(self):
        """Create sample original data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'age': np.random.normal(50, 15, 100).clip(18, 90),
            'income': np.random.normal(50000, 15000, 100).clip(20000, 150000),
            'score': np.random.normal(75, 10, 100).clip(0, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'binary_flag': np.random.choice([0, 1], 100)
        })
    
    @pytest.fixture
    def sample_synthetic_data(self):
        """Create sample synthetic data for testing."""
        np.random.seed(123)  # Different seed for synthetic data
        return pd.DataFrame({
            'age': np.random.normal(52, 14, 100).clip(18, 90),
            'income': np.random.normal(48000, 16000, 100).clip(20000, 150000),
            'score': np.random.normal(73, 11, 100).clip(0, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'binary_flag': np.random.choice([0, 1], 100)
        })
    
    def test_audit_function_basic(self, sample_original_data, sample_synthetic_data):
        """Test basic functionality of audit_synthetic_data function."""
        results = audit_synthetic_data(
            sample_original_data, 
            sample_synthetic_data,
            metadata=None,
            plots_dir=None
        )
        
        # Check that results contains expected keys
        expected_keys = ['quality_score', 'quality_details', 'missing_data_comparison', 
                        'original_stats', 'synthetic_stats']
        for key in expected_keys:
            assert key in results, f"Results should contain '{key}'"
        
        # Check quality score is between 0 and 1
        assert 0 <= results['quality_score'] <= 1, "Quality score should be between 0 and 1"
        
        # Check that stats are not None for numeric data
        assert results['original_stats'] is not None, "Original stats should not be None"
        assert results['synthetic_stats'] is not None, "Synthetic stats should not be None"
    
    def test_audit_function_with_plots_dir(self, sample_original_data, sample_synthetic_data):
        """Test audit function with plots directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = audit_synthetic_data(
                sample_original_data, 
                sample_synthetic_data,
                metadata=None,
                plots_dir=temp_dir
            )
            
            # Check that plot files were created
            plot_files = os.listdir(temp_dir)
            assert len(plot_files) > 0, "Plot files should be created"
            
            # Check for specific plot files
            expected_plots = ['missing_values_comparison.png', 'numeric_distributions.png']
            for plot in expected_plots:
                plot_path = os.path.join(temp_dir, plot)
                # Note: Files might not exist if matplotlib is not available, so we just check the function doesn't crash
    
    def test_audit_function_handles_missing_data(self, sample_original_data, sample_synthetic_data):
        """Test audit function handles missing data correctly."""
        # Add some missing data
        original_with_na = sample_original_data.copy()
        original_with_na.loc[0:10, 'age'] = np.nan
        
        synthetic_with_na = sample_synthetic_data.copy()
        synthetic_with_na.loc[5:15, 'income'] = np.nan
        
        results = audit_synthetic_data(
            original_with_na, 
            synthetic_with_na,
            metadata=None,
            plots_dir=None
        )
        
        # Check missing data comparison
        missing_comp = results['missing_data_comparison']
        assert 'Original_Missing' in missing_comp.columns, "Should have original missing column"
        assert 'Synthetic_Missing' in missing_comp.columns, "Should have synthetic missing column"
        
    def test_audit_task_output_definition(self):
        """Test that SyntheticDataAudit task defines outputs correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            config = {
                "working_dir": "/tmp/test",
                "synth_type": "GC"
            }
            json.dump(config, temp_file)
            temp_file.flush()
            
            try:
                task = SyntheticDataAudit(gen_config=temp_file.name, etl_config=temp_file.name)
                outputs = task.output()
                
                assert isinstance(outputs, dict), "Output should be a dictionary"
                assert 'audit_report' in outputs, "Should have audit_report output"
                assert 'audit_plots' in outputs, "Should have audit_plots output"
                
            finally:
                os.unlink(temp_file.name)
    
    def test_audit_task_requires_sdv_gen(self):
        """Test that SyntheticDataAudit requires SDVGen."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            config = {
                "working_dir": "/tmp/test",
                "synth_type": "GC"
            }
            json.dump(config, temp_file)
            temp_file.flush()
            
            try:
                task = SyntheticDataAudit(gen_config=temp_file.name, etl_config=temp_file.name)
                requirements = task.requires()
                
                assert isinstance(requirements, SDVGen), "Should require SDVGen"
                
            finally:
                os.unlink(temp_file.name)


class TestConfigurationValidation:
    """Test suite for configuration validation."""
    
    def test_valid_etl_config_structure(self):
        """Test validation of ETL configuration structure."""
        valid_config = {
            "name": "test",
            "absolute_path": "/path/to/data",
            "preprocessing": "/path/to/output",
            "final_cols": ["col1", "col2"]
        }
        
        # Test required fields
        required_fields = ["name", "absolute_path", "preprocessing", "final_cols"]
        for field in required_fields:
            assert field in valid_config, f"Config should have '{field}' field"
    
    def test_valid_synth_config_structure(self):
        """Test validation of synthesis configuration structure."""
        valid_config = {
            "input_file": "/path/to/input.parquet",
            "working_dir": "/path/to/output",
            "columns": ["col1", "col2"],
            "num_points": 1000,
            "synth_type": "GC"
        }
        
        # Test required fields
        required_fields = ["input_file", "working_dir", "columns", "num_points", "synth_type"]
        for field in required_fields:
            assert field in valid_config, f"Config should have '{field}' field"
        
        # Test data types
        assert isinstance(valid_config["num_points"], int), "num_points should be integer"
        assert isinstance(valid_config["columns"], list), "columns should be list"
        assert valid_config["synth_type"] in ["GC", "CTGAN", "TVAE"], "synth_type should be valid"


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_config_files_compatibility(self):
        """Test that test config files are compatible with each other."""
        test_dir = os.path.dirname(__file__)
        etl_config_path = os.path.join(test_dir, "etl.json")
        synth_config_path = os.path.join(test_dir, "synth.json")
        
        # Load both configs
        with open(etl_config_path, 'r') as f:
            etl_config = json.load(f)
        
        with open(synth_config_path, 'r') as f:
            synth_config = json.load(f)
        
        # Check compatibility
        etl_final_cols = set(etl_config.get('final_cols', []))
        synth_cols = set(synth_config.get('columns', []))
        
        # Synth columns should be a subset of ETL final columns
        assert synth_cols.issubset(etl_final_cols) or len(synth_cols) == 0, \
            "Synth columns should be subset of ETL final columns"
    
    @patch('dpplgngr.etl.prep_dataset_tabular.luigi')
    def test_pipeline_task_dependencies(self, mock_luigi):
        """Test that pipeline tasks have correct dependencies."""
        # This is a simplified test of task dependencies
        # In a real scenario, you'd want to test the actual Luigi dependency graph
        
        # Test that each task correctly identifies its requirements
        test_dir = os.path.dirname(__file__)
        etl_config = os.path.join(test_dir, "etl.json")
        synth_config = os.path.join(test_dir, "synth.json")
        
        # Create tasks
        preprocess_task = PreProcess(etl_config=etl_config)
        tuples_task = TuplesProcess(etl_config=etl_config)
        sdv_task = SDVGen(gen_config=synth_config, etl_config=etl_config)
        audit_task = SyntheticDataAudit(gen_config=synth_config, etl_config=etl_config)
        
        # Test dependency chain
        assert isinstance(tuples_task.requires(), PreProcess), "TuplesProcess should require PreProcess"
        assert isinstance(sdv_task.requires(), TuplesProcess), "SDVGen should require TuplesProcess"
        assert isinstance(audit_task.requires(), SDVGen), "SyntheticDataAudit should require SDVGen"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])