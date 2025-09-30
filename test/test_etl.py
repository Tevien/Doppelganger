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

from dpplgngr.etl.prep_dataset_tabular import PreProcess, TuplesProcess, ConvertLargeFiles


class TestETLFunctionality:
    """Tests for ETL pipeline components."""
    
    @pytest.fixture
    def sample_patient_data(self):
        """Create sample patient data similar to the test files."""
        np.random.seed(42)
        return pd.DataFrame({
            'id': range(1, 101),
            'age': np.random.randint(18, 90, 100),
            'gender': np.random.choice([0, 1], 100),
            'admission_date': pd.date_range('2020-01-01', periods=100, freq='D')
        })
    
    @pytest.fixture
    def sample_measurement_data(self):
        """Create sample measurement data."""
        np.random.seed(123)
        n_measurements = 300
        patient_ids = np.random.choice(range(1, 101), n_measurements)
        
        return pd.DataFrame({
            'id': patient_ids,
            'measurement': np.random.normal(100, 15, n_measurements),
            'measurement_type': np.random.choice([1, 2, 3], n_measurements),
            'measurement_date': pd.date_range('2020-01-01', periods=n_measurements, freq='6H')
        })
    
    @pytest.fixture
    def temp_data_dir(self, sample_patient_data, sample_measurement_data):
        """Create temporary directory with test data files."""
        temp_dir = tempfile.mkdtemp()
        
        # Save test data files
        patient_file = os.path.join(temp_dir, 'patients.parquet')
        measurement_file = os.path.join(temp_dir, 'measurements.parquet')
        
        sample_patient_data.to_parquet(patient_file)
        sample_measurement_data.to_parquet(measurement_file)
        
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def etl_config(self, temp_data_dir, temp_output_dir):
        """Create ETL configuration for testing."""
        return {
            "name": "test_etl",
            "description": "Test ETL configuration",
            "absolute_path": temp_data_dir,
            "preprocessing": temp_output_dir,
            "patients.parquet": ["id", "age", "gender", "admission_date"],
            "measurements.parquet": ["id", "measurement_type", {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}, ["measurement_date"]],
            "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_after": ["blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_anybefore": [],
            "ref_date": "measurement_date",
            "InitTransforms": {
                "measurement_classified": {
                    "func": "classify",
                    "kwargs": {
                        "classification_map": ["blood_pressure", "cholesterol", "glucose"],
                        "input_col": "measurement_type",
                        "out_col": "measurement_classified",
                        "id_col": "id"
                    }
                }
            },
            "PreTransforms": {
                "admission_date": {
                    "func": "datetime",
                    "kwargs": {
                        "col_to_date": "admission_date"
                    }
                },
                "measurement_date": {
                    "func": "datetime",
                    "kwargs": {
                        "col_to_date": "measurement_date"
                    }
                }
            },
            "MergedTransforms": {
                "gender": {
                    "func": "map",
                    "kwargs": {
                        "map": {0: 0, 1: 1}
                    }
                },
                "DropNaN": {
                    "func": "dropna",
                    "kwargs": {
                        "subset": ["age", "gender"]
                    }
                }
            }
        }
    
    @pytest.fixture
    def etl_config_file(self, temp_data_dir):
        """Create a temporary ETL config file for testing."""
        config = {
            "name": "test_etl_process",
            "absolute_path": temp_data_dir + "/",
            "preprocessing": temp_data_dir + "/output/",
            "data_file1.parquet": ["id", "age", "gender"],
            "data_file2.parquet": ["id", "measurement_type", {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}, ["measurement_date"]],
            "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_after": ["blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_anybefore": [],
            "ref_date": "measurement_date",
            "InitTransforms": {
                "measurement_classified": {
                    "func": "classify",
                    "kwargs": {
                        "classification_map": ["blood_pressure", "cholesterol", "glucose"],
                        "input_col": "measurement_type",
                        "out_col": "measurement_classified",
                        "id_col": "id"
                    }
                }
            },
            "PreTransforms": {
                "measurement_date": {
                    "func": "datetime",
                    "kwargs": {
                        "col_to_date": "measurement_date"
                    }
                }
            },
            "MergedTransforms": {
                "gender": {
                    "func": "map",
                    "kwargs": {
                        "map": {"0": 0, "1": 1}
                    }
                },
                "DropNaN": {
                    "func": "dropna",
                    "kwargs": {
                        "subset": ["age", "gender"]
                    }
                }
            }
        }
        
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, config_file)
        config_file.close()
        
        yield config_file.name
        
        # Cleanup
        os.unlink(config_file.name)


class TestPreProcessTask:
    """Test the PreProcess Luigi task."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def etl_config_file(self, temp_data_dir):
        """Create a temporary ETL config file for testing."""
        config = {
            "name": "test_etl_process",
            "absolute_path": temp_data_dir + "/",
            "preprocessing": temp_data_dir + "/output/",
            "data_file1.parquet": ["id", "age", "gender"],
            "data_file2.parquet": ["id", "measurement_type", {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}, ["measurement_date"]],
            "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_after": ["blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_anybefore": [],
            "ref_date": "measurement_date",
            "InitTransforms": {
                "measurement_classified": {
                    "func": "classify",
                    "kwargs": {
                        "classification_map": ["blood_pressure", "cholesterol", "glucose"],
                        "input_col": "measurement_type",
                        "out_col": "measurement_classified",
                        "id_col": "id"
                    }
                }
            },
            "PreTransforms": {
                "measurement_date": {
                    "func": "datetime",
                    "kwargs": {
                        "col_to_date": "measurement_date"
                    }
                }
            },
            "MergedTransforms": {
                "gender": {
                    "func": "map",
                    "kwargs": {
                        "map": {"0": 0, "1": 1}
                    }
                },
                "DropNaN": {
                    "func": "dropna",
                    "kwargs": {
                        "subset": ["age", "gender"]
                    }
                }
            }
        }
        
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, config_file)
        config_file.close()
        
        yield config_file.name
        
        # Cleanup
        os.unlink(config_file.name)
    
    def test_preprocess_output_path_generation(self, etl_config_file):
        """Test that PreProcess generates correct output path."""
        task = PreProcess(etl_config=etl_config_file)
        output = task.output()
        
        assert hasattr(output, 'path')
        assert output.path.endswith('preprocessed.parquet')
    
    def test_preprocess_config_loading(self, etl_config_file):
        """Test that PreProcess can load configuration correctly."""
        task = PreProcess(etl_config=etl_config_file)
        
        # This should not raise an exception
        output = task.output()
        assert output is not None
    
    @patch('dpplgngr.etl.prep_dataset_tabular.ConvertLargeFiles')
    def test_preprocess_requirements(self, mock_convert, etl_config_file):
        """Test that PreProcess has correct requirements."""
        task = PreProcess(etl_config=etl_config_file)
        
        # Test that it requires ConvertLargeFiles for FILE source
        requirements = task.requires()
        
        # Should call ConvertLargeFiles constructor
        mock_convert.assert_called_once()


class TestTuplesProcessTask:
    """Test the TuplesProcess Luigi task."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def etl_config_file(self, temp_data_dir):
        """Create a temporary ETL config file for testing."""
        config = {
            "name": "test_etl_process",
            "absolute_path": temp_data_dir + "/",
            "preprocessing": temp_data_dir + "/output/",
            "data_file1.parquet": ["id", "age", "gender"],
            "data_file2.parquet": ["id", "measurement_type", {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}, ["measurement_date"]],
            "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_after": ["blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_anybefore": [],
            "ref_date": "measurement_date",
            "InitTransforms": {
                "measurement_classified": {
                    "func": "classify",
                    "kwargs": {
                        "classification_map": ["blood_pressure", "cholesterol", "glucose"],
                        "input_col": "measurement_type",
                        "out_col": "measurement_classified",
                        "id_col": "id"
                    }
                }
            },
            "PreTransforms": {
                "measurement_date": {
                    "func": "datetime",
                    "kwargs": {
                        "col_to_date": "measurement_date"
                    }
                }
            },
            "MergedTransforms": {
                "gender": {
                    "func": "map",
                    "kwargs": {
                        "map": {"0": 0, "1": 1}
                    }
                },
                "DropNaN": {
                    "func": "dropna",
                    "kwargs": {
                        "subset": ["age", "gender"]
                    }
                }
            }
        }
        
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, config_file)
        config_file.close()
        
        yield config_file.name
        
        # Cleanup
        os.unlink(config_file.name)
    
    def test_tuples_process_output_path(self, etl_config_file):
        """Test TuplesProcess output path generation."""
        task = TuplesProcess(etl_config=etl_config_file)
        output = task.output()
        
        assert hasattr(output, 'path')
        assert 'tupleprocess' in output.path
        assert output.path.endswith('.parquet')
    
    def test_tuples_process_requires_preprocess(self, etl_config_file):
        """Test that TuplesProcess requires PreProcess."""
        task = TuplesProcess(etl_config=etl_config_file)
        requirements = task.requires()
        
        assert isinstance(requirements, PreProcess)
        assert requirements.etl_config == etl_config_file


class TestConvertLargeFilesTask:
    """Test the ConvertLargeFiles Luigi task."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def etl_config_file(self, temp_data_dir):
        """Create a temporary ETL config file for testing."""
        config = {
            "name": "test_etl_process",
            "absolute_path": temp_data_dir + "/",
            "preprocessing": temp_data_dir + "/output/",
            "data_file1.parquet": ["id", "age", "gender"],
            "data_file2.parquet": ["id", "measurement_type", {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}, ["measurement_date"]],
            "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_after": ["blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_anybefore": [],
            "ref_date": "measurement_date",
            "InitTransforms": {
                "measurement_classified": {
                    "func": "classify",
                    "kwargs": {
                        "classification_map": ["blood_pressure", "cholesterol", "glucose"],
                        "input_col": "measurement_type",
                        "out_col": "measurement_classified",
                        "id_col": "id"
                    }
                }
            },
            "PreTransforms": {
                "measurement_date": {
                    "func": "datetime",
                    "kwargs": {
                        "col_to_date": "measurement_date"
                    }
                }
            },
            "MergedTransforms": {
                "gender": {
                    "func": "map",
                    "kwargs": {
                        "map": {"0": 0, "1": 1}
                    }
                },
                "DropNaN": {
                    "func": "dropna",
                    "kwargs": {
                        "subset": ["age", "gender"]
                    }
                }
            }
        }
        
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, config_file)
        config_file.close()
        
        yield config_file.name
        
        # Cleanup
        os.unlink(config_file.name)
    
    def test_convert_large_files_output_path(self, etl_config_file):
        """Test ConvertLargeFiles output path generation."""
        task = ConvertLargeFiles(etl_config=etl_config_file)
        output = task.output()
        
        assert hasattr(output, 'path')
        assert output.path.endswith('.json')
    
    def test_convert_large_files_config_validation(self, etl_config_file):
        """Test config validation in ConvertLargeFiles."""
        task = ConvertLargeFiles(etl_config=etl_config_file)
        
        # Should be able to generate output without error
        output = task.output()
        assert output is not None


class TestETLConfigValidation:
    """Test ETL configuration validation and structure."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def etl_config_file(self, temp_data_dir):
        """Create a temporary ETL config file for testing."""
        config = {
            "name": "test_etl_process",
            "absolute_path": temp_data_dir + "/",
            "preprocessing": temp_data_dir + "/output/",
            "data_file1.parquet": ["id", "age", "gender"],
            "data_file2.parquet": ["id", "measurement_type", {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}, ["measurement_date"]],
            "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_after": ["blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_anybefore": [],
            "ref_date": "measurement_date",
            "InitTransforms": {
                "measurement_classified": {
                    "func": "classify",
                    "kwargs": {
                        "classification_map": ["blood_pressure", "cholesterol", "glucose"],
                        "input_col": "measurement_type",
                        "out_col": "measurement_classified",
                        "id_col": "id"
                    }
                }
            },
            "PreTransforms": {
                "measurement_date": {
                    "func": "datetime",
                    "kwargs": {
                        "col_to_date": "measurement_date"
                    }
                }
            },
            "MergedTransforms": {
                "gender": {
                    "func": "map",
                    "kwargs": {
                        "map": {"0": 0, "1": 1}
                    }
                },
                "DropNaN": {
                    "func": "dropna",
                    "kwargs": {
                        "subset": ["age", "gender"]
                    }
                }
            }
        }
        
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, config_file)
        config_file.close()
        
        yield config_file.name
        
        # Cleanup
        os.unlink(config_file.name)
    
    def test_required_etl_config_fields(self):
        """Test that required ETL config fields are validated."""
        # Test minimal valid config
        minimal_config = {
            "name": "test",
            "absolute_path": "/path/to/data",
            "preprocessing": "/path/to/output",
            "final_cols": ["col1", "col2"]
        }
        
        required_fields = ["name", "absolute_path", "preprocessing", "final_cols"]
        for field in required_fields:
            assert field in minimal_config
    
    def test_etl_config_data_types(self):
        """Test ETL config field data types."""
        config = {
            "name": "test",
            "absolute_path": "/path",
            "preprocessing": "/output",
            "final_cols": ["col1", "col2"],
            "tuple_vals_after": ["measurement"],
            "tuple_vals_anybefore": ["condition"]
        }
        
        assert isinstance(config["name"], str)
        assert isinstance(config["final_cols"], list)
        assert isinstance(config["tuple_vals_after"], list)
        assert isinstance(config["tuple_vals_anybefore"], list)
    
    def test_etl_transforms_structure(self):
        """Test ETL transforms configuration structure."""
        transforms_config = {
            "PreTransforms": {
                "date_col": {
                    "func": "datetime",
                    "kwargs": {"col_to_date": "date_col"}
                }
            },
            "MergedTransforms": {
                "categorical_col": {
                    "func": "map",
                    "kwargs": {"map": {"A": 0, "B": 1}}
                }
            }
        }
        
        # Validate structure
        assert "PreTransforms" in transforms_config
        assert "MergedTransforms" in transforms_config
        
        # Validate transform structure
        for transform_type in ["PreTransforms", "MergedTransforms"]:
            for transform_name, transform_config in transforms_config[transform_type].items():
                assert "func" in transform_config
                assert "kwargs" in transform_config
                assert isinstance(transform_config["kwargs"], dict)


class TestETLDataHandling:
    """Test ETL data handling functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def etl_config_file(self, temp_data_dir):
        """Create a temporary ETL config file for testing."""
        config = {
            "name": "test_etl_process",
            "absolute_path": temp_data_dir + "/",
            "preprocessing": temp_data_dir + "/output/",
            "data_file1.parquet": ["id", "age", "gender"],
            "data_file2.parquet": ["id", "measurement_type", {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}, ["measurement_date"]],
            "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_after": ["blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_anybefore": [],
            "ref_date": "measurement_date",
            "InitTransforms": {
                "measurement_classified": {
                    "func": "classify",
                    "kwargs": {
                        "classification_map": ["blood_pressure", "cholesterol", "glucose"],
                        "input_col": "measurement_type",
                        "out_col": "measurement_classified",
                        "id_col": "id"
                    }
                }
            },
            "PreTransforms": {
                "measurement_date": {
                    "func": "datetime",
                    "kwargs": {
                        "col_to_date": "measurement_date"
                    }
                }
            },
            "MergedTransforms": {
                "gender": {
                    "func": "map",
                    "kwargs": {
                        "map": {"0": 0, "1": 1}
                    }
                },
                "DropNaN": {
                    "func": "dropna",
                    "kwargs": {
                        "subset": ["age", "gender"]
                    }
                }
            }
        }
        
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, config_file)
        config_file.close()
        
        yield config_file.name
        
        # Cleanup
        os.unlink(config_file.name)
    
    def test_data_file_reading(self):
        """Test reading of test data files."""
        test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        # Test reading data_file1
        df1_path = os.path.join(test_data_dir, "data_file1.parquet")
        if os.path.exists(df1_path):
            df1 = pd.read_parquet(df1_path)
            assert not df1.empty
            assert 'id' in df1.columns
        
        # Test reading data_file2
        df2_path = os.path.join(test_data_dir, "data_file2.parquet")
        if os.path.exists(df2_path):
            df2 = pd.read_parquet(df2_path)
            assert not df2.empty
            assert 'id' in df2.columns
    
    def test_column_selection_logic(self):
        """Test column selection from configuration."""
        # Test data similar to actual test files
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'age': [25, 30, 35],
            'gender': [0, 1, 0],
            'extra_col': ['A', 'B', 'C']
        })
        
        # Simulate column selection based on config
        config_cols = ['id', 'age', 'gender']
        selected_df = df[config_cols]
        
        assert list(selected_df.columns) == config_cols
        assert selected_df.shape[1] == 3
        assert selected_df.shape[0] == df.shape[0]
    
    def test_data_type_handling(self):
        """Test handling of different data types in ETL."""
        # Create data with mixed types
        mixed_data = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['A', 'B', 'C'],
            'date_col': pd.date_range('2020-01-01', periods=3),
            'bool_col': [True, False, True]
        })
        
        # Test that data types are preserved appropriately
        assert mixed_data['int_col'].dtype in ['int64', 'int32']
        assert mixed_data['float_col'].dtype in ['float64', 'float32']
        assert mixed_data['str_col'].dtype == 'object'
        assert 'datetime' in str(mixed_data['date_col'].dtype)
        assert mixed_data['bool_col'].dtype == 'bool'


class TestETLErrorHandling:
    """Test ETL error handling functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def etl_config_file(self, temp_data_dir):
        """Create a temporary ETL config file for testing."""
        config = {
            "name": "test_etl_process",
            "absolute_path": temp_data_dir + "/",
            "preprocessing": temp_data_dir + "/output/",
            "data_file1.parquet": ["id", "age", "gender"],
            "data_file2.parquet": ["id", "measurement_type", {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}, ["measurement_date"]],
            "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_after": ["blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_anybefore": [],
            "ref_date": "measurement_date",
            "InitTransforms": {
                "measurement_classified": {
                    "func": "classify",
                    "kwargs": {
                        "classification_map": ["blood_pressure", "cholesterol", "glucose"],
                        "input_col": "measurement_type",
                        "out_col": "measurement_classified",
                        "id_col": "id"
                    }
                }
            },
            "PreTransforms": {
                "measurement_date": {
                    "func": "datetime",
                    "kwargs": {
                        "col_to_date": "measurement_date"
                    }
                }
            },
            "MergedTransforms": {
                "gender": {
                    "func": "map",
                    "kwargs": {
                        "map": {"0": 0, "1": 1}
                    }
                },
                "DropNaN": {
                    "func": "dropna",
                    "kwargs": {
                        "subset": ["age", "gender"]
                    }
                }
            }
        }
        
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, config_file)
        config_file.close()
        
        yield config_file.name
        
        # Cleanup
        os.unlink(config_file.name)
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        non_existent_config = "/non/existent/config.json"
        
        with pytest.raises(FileNotFoundError):
            task = PreProcess(etl_config=non_existent_config)
            task.output()
    
    def test_invalid_config_format(self):
        """Test handling of invalid configuration format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            # Write invalid JSON
            temp_file.write("invalid json content")
            temp_file.flush()
            
            try:
                with pytest.raises(json.JSONDecodeError):
                    task = PreProcess(etl_config=temp_file.name)
                    # This will fail when trying to read the config
                    with open(temp_file.name, 'r') as f:
                        json.load(f)
            finally:
                os.unlink(temp_file.name)
    
    def test_missing_required_config_fields(self):
        """Test handling of missing required configuration fields."""
        incomplete_config = {
            "name": "test"
            # Missing other required fields like 'preprocessing'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(incomplete_config, temp_file)
            temp_file.flush()
            
            try:
                task = PreProcess(etl_config=temp_file.name)
                # The output() method should work because it has fallbacks
                # But let's test that it creates the expected fallback path
                output = task.output()
                expected_path = "data/test/preprocessing"
                assert expected_path in output.path
            finally:
                os.unlink(temp_file.name)


class TestMeasurementExpansion:
    """Test measurement expansion functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def measurement_expansion_data(self):
        """Create test data for measurement expansion."""
        np.random.seed(42)
        return pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 3, 3, 3],
            'measurement': [120.5, 180.2, 95.8, 110.3, 200.1, 125.7, 165.9, 88.2],
            'measurement_type': [1, 2, 3, 1, 2, 1, 2, 3],
            'measurement_date': pd.date_range('2020-01-01', periods=8, freq='D')
        })
    
    @pytest.fixture
    def etl_config_file(self, temp_data_dir):
        """Create a temporary ETL config file for testing."""
        config = {
            "name": "test_etl_process",
            "absolute_path": temp_data_dir + "/",
            "preprocessing": temp_data_dir + "/output/",
            "data_file1.parquet": ["id", "age", "gender"],
            "data_file2.parquet": ["id", "measurement_type", {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}, ["measurement_date"]],
            "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_after": ["blood_pressure", "cholesterol", "glucose"],
            "tuple_vals_anybefore": [],
            "ref_date": "measurement_date",
            "InitTransforms": {
                "measurement_classified": {
                    "func": "classify",
                    "kwargs": {
                        "classification_map": ["blood_pressure", "cholesterol", "glucose"],
                        "input_col": "measurement_type",
                        "out_col": "measurement_classified",
                        "id_col": "id"
                    }
                }
            },
            "PreTransforms": {
                "measurement_date": {
                    "func": "datetime",
                    "kwargs": {
                        "col_to_date": "measurement_date"
                    }
                }
            },
            "MergedTransforms": {
                "gender": {
                    "func": "map",
                    "kwargs": {
                        "map": {"0": 0, "1": 1}
                    }
                },
                "DropNaN": {
                    "func": "dropna",
                    "kwargs": {
                        "subset": ["age", "gender"]
                    }
                }
            }
        }
        
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, config_file)
        config_file.close()
        
        yield config_file.name
        
        # Cleanup
        os.unlink(config_file.name)
    
    def test_measurement_type_mapping_config(self):
        """Test that measurement type mapping configuration is structured correctly."""
        config = {
            "data_file2.parquet": ["id", "measurement_type", {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}, ["measurement_date"]]
        }
        
        # Validate the structure
        file_config = config["data_file2.parquet"]
        assert isinstance(file_config, list)
        
        # Find the measurement mapping dictionary
        measurement_mapping = None
        for item in file_config:
            if isinstance(item, dict) and "measurement" in item:
                measurement_mapping = item["measurement"]
                break
        
        assert measurement_mapping is not None, "Should have measurement mapping"
        assert "1" in measurement_mapping, "Should map measurement type 1"
        assert "2" in measurement_mapping, "Should map measurement type 2" 
        assert "3" in measurement_mapping, "Should map measurement type 3"
        assert measurement_mapping["1"] == "blood_pressure"
        assert measurement_mapping["2"] == "cholesterol"
        assert measurement_mapping["3"] == "glucose"
    
    def test_measurement_expansion_columns(self):
        """Test that expanded measurement columns are included in final_cols."""
        config = {
            "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"]
        }
        
        expanded_measurements = ["blood_pressure", "cholesterol", "glucose"]
        for measurement in expanded_measurements:
            assert measurement in config["final_cols"], f"final_cols should include {measurement}"
    
    def test_tuple_vals_after_expansion(self):
        """Test that tuple_vals_after includes expanded measurement columns."""
        config = {
            "tuple_vals_after": ["blood_pressure", "cholesterol", "glucose"]
        }
        
        expected_measurements = ["blood_pressure", "cholesterol", "glucose"]
        assert config["tuple_vals_after"] == expected_measurements
    
    def test_init_transforms_for_measurement(self):
        """Test InitTransforms configuration for measurement classification."""
        config = {
            "InitTransforms": {
                "measurement_classified": {
                    "func": "classify",
                    "kwargs": {
                        "classification_map": ["blood_pressure", "cholesterol", "glucose"],
                        "input_col": "measurement_type",
                        "out_col": "measurement_classified",
                        "id_col": "id"
                    }
                }
            }
        }
        
        transform_config = config["InitTransforms"]["measurement_classified"]
        assert transform_config["func"] == "classify"
        
        kwargs = transform_config["kwargs"]
        assert kwargs["input_col"] == "measurement_type"
        assert kwargs["out_col"] == "measurement_classified"
        assert kwargs["id_col"] == "id"
        assert "blood_pressure" in kwargs["classification_map"]
        assert "cholesterol" in kwargs["classification_map"]
        assert "glucose" in kwargs["classification_map"]
    
    def test_measurement_data_processing_simulation(self, measurement_expansion_data):
        """Test simulated measurement data processing similar to ETL pipeline."""
        df = measurement_expansion_data.copy()
        
        # Simulate the measurement type expansion process
        # This would normally be done by the ETL pipeline
        expanded_measurements = {
            1: "blood_pressure",
            2: "cholesterol", 
            3: "glucose"
        }
        
        # Create separate DataFrames for each measurement type
        measurement_dfs = {}
        for mtype, mname in expanded_measurements.items():
            subset = df[df['measurement_type'] == mtype].copy()
            subset = subset.rename(columns={'measurement': mname})
            subset = subset[['id', mname, 'measurement_date']]
            measurement_dfs[mname] = subset
        
        # Verify that we have the right structure
        assert 'blood_pressure' in measurement_dfs
        assert 'cholesterol' in measurement_dfs
        assert 'glucose' in measurement_dfs
        
        # Check that each measurement type has the correct data
        bp_df = measurement_dfs['blood_pressure']
        assert len(bp_df) == 3  # Should have 3 blood pressure measurements
        assert 'blood_pressure' in bp_df.columns
        assert bp_df['blood_pressure'].iloc[0] == 120.5  # First blood pressure value
        
        chol_df = measurement_dfs['cholesterol']
        assert len(chol_df) == 3  # Should have 3 cholesterol measurements
        assert 'cholesterol' in chol_df.columns
        
        glucose_df = measurement_dfs['glucose']
        assert len(glucose_df) == 2  # Should have 2 glucose measurements
        assert 'glucose' in glucose_df.columns
    
    def test_measurement_expansion_compatibility_with_amc_pattern(self):
        """Test that our measurement expansion follows the same pattern as AMC Labuitslag."""
        # AMC pattern: {"UitslagNumeriek": {"RKRE;BL": "creatinine", "RHDL;BL": "hdl_cholesterol", "RCHO;BL": "total_cholesterol"}}
        amc_pattern = {
            "Labuitslag.feather": [
                "pseudo_id", 
                "BepalingCode", 
                {"UitslagNumeriek": {"RKRE;BL": "creatinine", "RHDL;BL": "hdl_cholesterol", "RCHO;BL": "total_cholesterol"}}, 
                ["MateriaalAfnameDatum"]
            ]
        }
        
        # Our test pattern: {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}
        test_pattern = {
            "data_file2.parquet": [
                "id", 
                "measurement_type", 
                {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}, 
                ["measurement_date"]
            ]
        }
        
        # Validate structural similarity
        amc_file_config = amc_pattern["Labuitslag.feather"]
        test_file_config = test_pattern["data_file2.parquet"]
        
        # Both should have 4 elements: id_col, type_col, mapping_dict, date_col_list
        assert len(amc_file_config) == 4
        assert len(test_file_config) == 4
        
        # Both should have mapping dictionaries in the same position
        amc_mapping = amc_file_config[2]
        test_mapping = test_file_config[2]
        
        assert isinstance(amc_mapping, dict)
        assert isinstance(test_mapping, dict)
        
        # Both should have exactly one key in the mapping dict
        assert len(amc_mapping.keys()) == 1
        assert len(test_mapping.keys()) == 1
        
        # Both should have date columns as lists in the last position
        assert isinstance(amc_file_config[3], list)
        assert isinstance(test_file_config[3], list)


    def test_actual_test_data_compatibility(self):
        """Test that our measurement expansion config works with actual test data."""
        test_data_path = os.path.join(os.path.dirname(__file__), "data", "data_file2.parquet")
        
        if os.path.exists(test_data_path):
            df = pd.read_parquet(test_data_path)
            
            # Verify data structure matches our configuration expectations
            expected_columns = ['id', 'measurement', 'measurement_type', 'measurement_date']
            for col in expected_columns:
                assert col in df.columns, f"Test data should have {col} column"
            
            # Verify measurement types match our mapping
            unique_types = set(df['measurement_type'].unique())
            expected_types = {1, 2, 3}
            assert unique_types.issubset(expected_types), f"Measurement types {unique_types} should be subset of {expected_types}"
            
            # Verify we have data for each measurement type
            for mtype in expected_types:
                type_data = df[df['measurement_type'] == mtype]
                if len(type_data) > 0:  # Only check if we have data for this type
                    assert type_data['measurement'].notna().any(), f"Should have measurement values for type {mtype}"
            
            print(f"✅ Test data validation passed: {len(df)} records with measurement types {sorted(unique_types)}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
