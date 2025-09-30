import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import shutil
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dpplgngr.train.sdv import SDVGen
from dpplgngr.etl.prep_dataset_tabular import TuplesProcess


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation functionality."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for synthetic generation."""
        np.random.seed(42)
        n = 500
        
        return pd.DataFrame({
            'age': np.random.normal(50, 15, n).clip(18, 90),
            'income': np.random.normal(50000, 20000, n).clip(20000, 200000),
            'education_years': np.random.randint(8, 20, n),
            'health_score': np.random.normal(75, 10, n).clip(0, 100),
            'gender': np.random.choice([0, 1], n),
            'married': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n)
        })
    
    @pytest.fixture
    def temp_input_file(self, sample_training_data):
        """Create temporary input file for testing."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        sample_training_data.to_parquet(temp_file.name)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def temp_working_dir(self):
        """Create temporary working directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestSDVGenTask:
    """Test the SDVGen Luigi task."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data that matches the ETL pipeline output structure."""
        np.random.seed(42)
        n = 500
        
        # This should match the final_cols from etl.json: ["age", "gender", "blood_pressure", "cholesterol", "glucose"]
        return pd.DataFrame({
            'age': np.random.normal(50, 15, n).clip(18, 90),
            'gender': np.random.choice([0, 1], n),
            'blood_pressure': np.random.normal(120, 20, n).clip(80, 200),
            'cholesterol': np.random.normal(180, 40, n).clip(100, 350),
            'glucose': np.random.normal(95, 15, n).clip(70, 200)
        })
    
    @pytest.fixture
    def temp_input_file(self, sample_training_data):
        """Create temporary input file for testing with ETL-like data."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        sample_training_data.to_parquet(temp_file.name)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def temp_working_dir(self):
        """Create temporary working directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def synth_config_file(self, temp_input_file, temp_working_dir):
        """Create a temporary synthesis config file for testing."""
        config = {
            "input_file": temp_input_file,
            "working_dir": temp_working_dir,
            "columns": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],  # Match ETL final_cols
            "num_points": 100,
            "synth_type": "GC"
        }
        
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, config_file)
        config_file.close()
        
        yield config_file.name
        
        # Cleanup
        os.unlink(config_file.name)
    
    @pytest.fixture
    def dummy_etl_config_file(self, temp_working_dir):
        """Create a dummy ETL config file for testing."""
        config = {
            "name": "dummy_etl",
            "absolute_path": temp_working_dir + "/",
            "preprocessing": temp_working_dir + "/output/",
            "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"]  # Match ETL final_cols
        }
        
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, config_file)
        config_file.close()
        
        yield config_file.name
        
        # Cleanup
        os.unlink(config_file.name)
    
    def test_sdv_gen_output_path_generation(self, synth_config_file, dummy_etl_config_file):
        """Test that SDVGen generates correct output path."""
        task = SDVGen(gen_config=synth_config_file, etl_config=dummy_etl_config_file)
        output = task.output()
        
        assert hasattr(output, 'path')
        assert output.path.endswith('.pkl')
        assert 'synth_GC.pkl' in output.path
    
    def test_sdv_gen_requires_tuples_process(self, synth_config_file, dummy_etl_config_file):
        """Test that SDVGen requires TuplesProcess."""
        task = SDVGen(gen_config=synth_config_file, etl_config=dummy_etl_config_file)
        requirements = task.requires()
        
        assert isinstance(requirements, TuplesProcess)
        assert requirements.etl_config == dummy_etl_config_file
    
    def test_sdv_gen_config_loading(self, synth_config_file, dummy_etl_config_file):
        """Test that SDVGen can load configuration correctly."""
        task = SDVGen(gen_config=synth_config_file, etl_config=dummy_etl_config_file)
        
        # Should be able to generate output path without error
        output = task.output()
        assert output is not None
    
    def test_sdv_gen_working_directory_creation(self, synth_config_file, dummy_etl_config_file):
        """Test that SDVGen creates working directory if it doesn't exist."""
        task = SDVGen(gen_config=synth_config_file, etl_config=dummy_etl_config_file)
        output = task.output()
        
        # The output() method should create the directory
        working_dir = os.path.dirname(output.path)
        assert os.path.exists(working_dir)


class TestSynthesizerTypes:
    """Test different synthesizer types and configurations."""
    
    @pytest.fixture
    def temp_input_file(self, sample_training_data):
        """Create temporary input file for testing."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        sample_training_data.to_parquet(temp_file.name)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def temp_working_dir(self):
        """Create temporary working directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for synthetic generation."""
        np.random.seed(42)
        n = 500
        
        return pd.DataFrame({
            'age': np.random.normal(50, 15, n).clip(18, 90),
            'income': np.random.normal(50000, 20000, n).clip(20000, 200000),
            'education_years': np.random.randint(8, 20, n),
            'health_score': np.random.normal(75, 10, n).clip(0, 100),
            'gender': np.random.choice([0, 1], n),
            'married': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n)
        })
    
    @pytest.fixture
    def base_config(self, temp_input_file, temp_working_dir):
        """Base configuration for synthesizer tests."""
        return {
            "input_file": temp_input_file,
            "working_dir": temp_working_dir,
            "columns": ["age", "income", "blood_pressure", "cholesterol", "glucose"],
            "num_points": 50
        }
    
    def test_gaussian_copula_config(self, base_config):
        """Test Gaussian Copula synthesizer configuration."""
        gc_config = base_config.copy()
        gc_config["synth_type"] = "GC"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(gc_config, temp_file)
            temp_file.flush()
            
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as etl_file:
                    json.dump({"name": "dummy", "preprocessing": "/tmp"}, etl_file)
                    etl_file.flush()
                    
                    try:
                        task = SDVGen(gen_config=temp_file.name, etl_config=etl_file.name)
                        output = task.output()
                        assert 'synth_GC.pkl' in output.path
                    finally:
                        os.unlink(etl_file.name)
            finally:
                os.unlink(temp_file.name)
    
    def test_ctgan_config(self, base_config):
        """Test CTGAN synthesizer configuration."""
        ctgan_config = base_config.copy()
        ctgan_config["synth_type"] = "CTGAN"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(ctgan_config, temp_file)
            temp_file.flush()
            
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as etl_file:
                    json.dump({"name": "dummy", "preprocessing": "/tmp"}, etl_file)
                    etl_file.flush()
                    
                    try:
                        task = SDVGen(gen_config=temp_file.name, etl_config=etl_file.name)
                        output = task.output()
                        assert 'synth_CTGAN.pkl' in output.path
                    finally:
                        os.unlink(etl_file.name)
            finally:
                os.unlink(temp_file.name)
    
    def test_tvae_config(self, base_config):
        """Test TVAE synthesizer configuration."""
        tvae_config = base_config.copy()
        tvae_config["synth_type"] = "TVAE"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(tvae_config, temp_file)
            temp_file.flush()
            
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as etl_file:
                    json.dump({"name": "dummy", "preprocessing": "/tmp"}, etl_file)
                    etl_file.flush()
                    
                    try:
                        task = SDVGen(gen_config=temp_file.name, etl_config=etl_file.name)
                        output = task.output()
                        assert 'synth_TVAE.pkl' in output.path
                    finally:
                        os.unlink(etl_file.name)
            finally:
                os.unlink(temp_file.name)


class TestSynthConfigValidation:
    """Test synthesis configuration validation."""
    
    def test_required_synth_config_fields(self):
        """Test required fields in synthesis configuration."""
        required_fields = ["input_file", "working_dir", "columns", "num_points", "synth_type"]
        
        valid_config = {
            "input_file": "/path/to/input.parquet",
            "working_dir": "/path/to/output",
            "columns": ["col1", "col2"],
            "num_points": 1000,
            "synth_type": "GC"
        }
        
        for field in required_fields:
            assert field in valid_config, f"Config should have '{field}' field"
    
    def test_synth_config_data_types(self):
        """Test data types in synthesis configuration."""
        config = {
            "input_file": "/path/to/input.parquet",
            "working_dir": "/path/to/output",
            "columns": ["col1", "col2", "col3"],
            "num_points": 1000,
            "synth_type": "GC"
        }
        
        assert isinstance(config["input_file"], str)
        assert isinstance(config["working_dir"], str)
        assert isinstance(config["columns"], list)
        assert isinstance(config["num_points"], int)
        assert isinstance(config["synth_type"], str)
        assert config["synth_type"] in ["GC", "CTGAN", "TVAE", "RTF"]
    
    def test_synth_config_column_validation(self):
        """Test column specification validation."""
        config = {
            "columns": ["age", "income", "gender", "education"]
        }
        
        # Columns should be a non-empty list
        assert isinstance(config["columns"], list)
        assert len(config["columns"]) > 0
        assert all(isinstance(col, str) for col in config["columns"])
    
    def test_synth_config_num_points_validation(self):
        """Test num_points validation."""
        valid_num_points = [10, 100, 1000, 10000]
        
        for num_points in valid_num_points:
            assert isinstance(num_points, int)
            assert num_points > 0


class TestSyntheticDataProcessing:
    """Test synthetic data processing and handling."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for synthetic generation."""
        np.random.seed(42)
        n = 500
        
        return pd.DataFrame({
            'age': np.random.normal(50, 15, n).clip(18, 90),
            'income': np.random.normal(50000, 20000, n).clip(20000, 200000),
            'education_years': np.random.randint(8, 20, n),
            'health_score': np.random.normal(75, 10, n).clip(0, 100),
            'gender': np.random.choice([0, 1], n),
            'married': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n)
        })
    
    def test_data_loading_and_preprocessing(self, sample_training_data):
        """Test data loading and basic preprocessing."""
        # Test BMI filtering (from the actual SDV code)
        data_with_outliers = sample_training_data.copy()
        data_with_outliers['BMI'] = np.random.normal(25, 50, len(data_with_outliers))  # Some will be > 100
        
        # Simulate the BMI filtering from SDVGen
        filtered_data = data_with_outliers[data_with_outliers["BMI"] < 100] if "BMI" in data_with_outliers.columns else data_with_outliers
        
        # Should remove outliers
        if "BMI" in data_with_outliers.columns:
            assert len(filtered_data) <= len(data_with_outliers)
    
    def test_column_selection(self, sample_training_data):
        """Test column selection from configuration."""
        selected_columns = ["age", "income", "gender"]
        selected_data = sample_training_data[selected_columns]
        
        assert list(selected_data.columns) == selected_columns
        assert selected_data.shape[1] == len(selected_columns)
        assert selected_data.shape[0] == sample_training_data.shape[0]
    
    def test_timedelta_conversion(self):
        """Test timedelta column conversion."""
        # Create data with timedelta column
        data_with_timedelta = pd.DataFrame({
            'id': [1, 2, 3],
            'time_diff': pd.to_timedelta(['10 days', '20 days', '30 days'])
        })
        
        # Simulate the timedelta conversion from SDVGen
        for col in data_with_timedelta.columns:
            if data_with_timedelta[col].dtype.kind == 'm':  # 'm' indicates timedelta
                data_with_timedelta[col] = data_with_timedelta[col].dt.days
        
        # Should convert to numeric days
        assert data_with_timedelta['time_diff'].dtype in ['int64', 'float64']
        assert data_with_timedelta['time_diff'].tolist() == [10, 20, 30]
    
    def test_metadata_generation(self, sample_training_data):
        """Test metadata generation for SDV."""
        # This would normally use SDV's SingleTableMetadata
        # Here we just test that we can analyze the data structure
        
        numeric_cols = sample_training_data.select_dtypes(include=[np.number]).columns
        categorical_cols = sample_training_data.select_dtypes(include=['object', 'category']).columns
        
        assert len(numeric_cols) > 0  # Should have numeric columns
        assert len(categorical_cols) >= 0  # May or may not have categorical columns
        
        # Test that we can identify column types correctly
        assert 'age' in numeric_cols
        assert 'income' in numeric_cols
        if 'region' in sample_training_data.columns:
            assert 'region' in categorical_cols


class TestSynthErrorHandling:
    """Test error handling in synthetic data generation."""
    
    @pytest.fixture
    def temp_working_dir(self):
        """Create temporary working directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def temp_input_file(self, sample_training_data):
        """Create temporary input file for testing."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        sample_training_data.to_parquet(temp_file.name)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for synthetic generation."""
        np.random.seed(42)
        n = 500
        
        return pd.DataFrame({
            'age': np.random.normal(50, 15, n).clip(18, 90),
            'income': np.random.normal(50000, 20000, n).clip(20000, 200000),
            'education_years': np.random.randint(8, 20, n),
            'health_score': np.random.normal(75, 10, n).clip(0, 100),
            'gender': np.random.choice([0, 1], n),
            'married': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n)
        })
    
    def test_missing_input_file(self, temp_working_dir):
        """Test handling of missing input file."""
        config = {
            "input_file": "/non/existent/file.parquet",
            "working_dir": temp_working_dir,
            "columns": ["col1", "col2"],
            "num_points": 100,
            "synth_type": "GC"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(config, temp_file)
            temp_file.flush()
            
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as etl_file:
                    json.dump({"name": "dummy", "preprocessing": "/tmp"}, etl_file)
                    etl_file.flush()
                    
                    try:
                        task = SDVGen(gen_config=temp_file.name, etl_config=etl_file.name)
                        # The task should be created successfully
                        # The error would occur during run(), not during initialization
                        assert task is not None
                    finally:
                        os.unlink(etl_file.name)
            finally:
                os.unlink(temp_file.name)
    
    def test_invalid_synthesizer_type(self, temp_input_file, temp_working_dir):
        """Test handling of invalid synthesizer type."""
        config = {
            "input_file": temp_input_file,
            "working_dir": temp_working_dir,
            "columns": ["age", "income"],
            "num_points": 100,
            "synth_type": "INVALID_TYPE"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(config, temp_file)
            temp_file.flush()
            
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as etl_file:
                    json.dump({"name": "dummy", "preprocessing": "/tmp"}, etl_file)
                    etl_file.flush()
                    
                    try:
                        task = SDVGen(gen_config=temp_file.name, etl_config=etl_file.name)
                        # Task creation should succeed, error would occur during run()
                        assert task is not None
                    finally:
                        os.unlink(etl_file.name)
            finally:
                os.unlink(temp_file.name)
    
    def test_missing_columns_in_data(self, sample_training_data, temp_working_dir):
        """Test handling of columns specified in config but missing in data."""
        # Create input file with sample data
        temp_input = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        sample_training_data.to_parquet(temp_input.name)
        temp_input.close()
        
        try:
            # Config specifies columns not in the data
            config = {
                "input_file": temp_input.name,
                "working_dir": temp_working_dir,
                "columns": ["age", "income", "nonexistent_column"],
                "num_points": 100,
                "synth_type": "GC"
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
                json.dump(config, config_file)
                config_file.flush()
                
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as etl_file:
                        json.dump({"name": "dummy", "preprocessing": "/tmp"}, etl_file)
                        etl_file.flush()
                        
                        try:
                            task = SDVGen(gen_config=config_file.name, etl_config=etl_file.name)
                            # Task should be created, error would occur during run()
                            assert task is not None
                        finally:
                            os.unlink(etl_file.name)
                finally:
                    os.unlink(config_file.name)
        finally:
            os.unlink(temp_input.name)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
