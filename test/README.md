# Doppelganger Test Suite

This directory contains comprehensive pytest unit tests for the Doppelganger package functionality.

## Test Structure

The test suite is organized into several specialized test files:

### Core Test Files

1. **`test.py`** - Main test suite with comprehensive integration tests
2. **`test_etl.py`** - Specialized tests for ETL pipeline components
3. **`test_synth.py`** - Tests for synthetic data generation functionality
4. **`test_audit.py`** - Tests for synthetic data audit and quality evaluation

### Test Data

- **`data/data_file1.parquet`** - Sample patient data (1000 records with id, age, gender)
- **`data/data_file2.parquet`** - Sample measurement data (3000 records with measurements and dates, expanded by type)

### Configuration Files

- **`etl.json`** - Test ETL configuration for processing the sample data
- **`synth.json`** - Test synthesis configuration for generating synthetic data

## Test Categories

The tests are organized into several categories using pytest markers:

### Unit Tests (`@pytest.mark.unit`)
- Test individual functions and methods
- Mock external dependencies
- Fast execution

### Integration Tests (`@pytest.mark.integration`)
- Test component interactions
- Use real data files
- Test complete workflows

### Component-Specific Tests
- **ETL Tests**: Data preprocessing, transformation, and Luigi task functionality
- **Synthesis Tests**: Synthetic data generation, different synthesizer types
- **Audit Tests**: Quality evaluation, statistical comparison, plot generation

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --type etl
python run_tests.py --type synth
python run_tests.py --type audit

# Run with coverage report
python run_tests.py --coverage

# Install dependencies and run tests
python run_tests.py --install-deps
```

### Using pytest directly

```bash
# Run all tests
pytest test/ -v

# Run specific test file
pytest test/test_etl.py -v

# Run tests with specific markers
pytest test/ -m "unit" -v
pytest test/ -m "integration" -v

# Run with coverage
pytest test/ --cov=dpplgngr --cov-report=html
```

## Test Coverage

The test suite covers:

### ETL Pipeline (`test_etl.py`)
- ✅ Configuration loading and validation
- ✅ Luigi task output path generation
- ✅ Task dependency relationships
- ✅ Data file reading and processing
- ✅ Column selection and data type handling
- ✅ Error handling for missing files and invalid configs

### Synthetic Data Generation (`test_synth.py`)
- ✅ SDVGen Luigi task functionality
- ✅ All synthesizer types (GC, CTGAN, TVAE)
- ✅ Configuration validation
- ✅ Data preprocessing and column selection
- ✅ Metadata generation
- ✅ Error handling for invalid inputs

### Data Audit (`test_audit.py`)
- ✅ Quality evaluation using SDV metrics
- ✅ Missing data analysis
- ✅ Statistical comparison
- ✅ Distribution analysis
- ✅ Correlation analysis
- ✅ Plot generation and saving
- ✅ Error handling for edge cases

### Integration (`test.py`)
- ✅ End-to-end pipeline testing
- ✅ Configuration compatibility
- ✅ Task dependency validation
- ✅ Real data processing

## Test Data Description

### data_file1.parquet
- **Records**: 1000
- **Columns**: id, age, gender
- **Purpose**: Simulates patient demographic data
- **Data Types**: Integer ID, age values 18-90, binary gender (0/1)

### data_file2.parquet
- **Records**: 3000
- **Columns**: id, measurement, measurement_type, measurement_date
- **Purpose**: Simulates medical measurements over time with type-based expansion
- **Data Types**: Patient ID references, numeric measurements, categorical types (1=blood_pressure, 2=cholesterol, 3=glucose), dates
- **Expansion**: Measurements are expanded from generic 'measurement' column to specific columns based on measurement_type

## Measurement Type Expansion

The test suite now includes functionality to test measurement type expansion, similar to the AMC Labuitslag processing pattern. This feature allows:

### Pattern Matching with AMC
- **AMC Pattern**: `{"UitslagNumeriek": {"RKRE;BL": "creatinine", "RHDL;BL": "hdl_cholesterol"}}`
- **Test Pattern**: `{"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}`

### Configuration Structure
```json
"data_file2.parquet": [
    "id",                    // Patient identifier
    "measurement_type",      // Type classification column
    {
        "measurement": {     // Value column to expand
            "1": "blood_pressure",  // Type 1 → blood_pressure column
            "2": "cholesterol",     // Type 2 → cholesterol column  
            "3": "glucose"          // Type 3 → glucose column
        }
    },
    ["measurement_date"]     // Date columns (as list)
]
```

### Test Coverage for Expansion
- ✅ Configuration structure validation
- ✅ Column mapping correctness
- ✅ InitTransforms for classification
- ✅ tuple_vals_after configuration
- ✅ Compatibility with actual test data
- ✅ Pattern matching with AMC structure

### How It Works
1. **Raw Data**: Single `measurement` column with `measurement_type` indicator
2. **Expansion**: Creates separate columns (`blood_pressure`, `cholesterol`, `glucose`)
3. **Processing**: Each measurement type becomes a separate feature for ML/synthesis
4. **Tuples**: After-reference-date processing for each expanded measurement

This mirrors the real-world AMC data processing where lab results with different test codes are expanded into specific biomarker columns.

## Configuration Examples

### ETL Configuration (`etl.json`)
```json
{
    "name": "test_data_process",
    "absolute_path": "./test/data/",
    "preprocessing": "./test/output/",
    "data_file2.parquet": ["id", "measurement_type", {"measurement": {"1": "blood_pressure", "2": "cholesterol", "3": "glucose"}}, ["measurement_date"]],
    "final_cols": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
    "tuple_vals_after": ["blood_pressure", "cholesterol", "glucose"],
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
    }
}
```

### Synthesis Configuration (`synth.json`)
```json
{
    "input_file": "./test/output/preprocessed_tupleprocess.parquet",
    "working_dir": "./test/output/",
    "columns": ["age", "gender", "blood_pressure", "cholesterol", "glucose"],
    "num_points": 100,
    "synth_type": "GC"
}
```

## Writing New Tests

### Test Structure
```python
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from dpplgngr.module import YourClass

class TestYourFunctionality:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({...})
    
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        result = your_function(sample_data)
        assert result is not None
        
    @pytest.mark.unit
    def test_unit_functionality(self):
        """Unit test with mocking."""
        # Test implementation
        pass
        
    @pytest.mark.integration
    def test_integration_functionality(self):
        """Integration test with real data."""
        # Test implementation
        pass
```

### Best Practices
1. **Use fixtures** for reusable test data and configurations
2. **Mock external dependencies** in unit tests
3. **Use temporary files/directories** for file operations
4. **Test both success and failure cases**
5. **Use descriptive test names** that explain what is being tested
6. **Add appropriate markers** for test categorization

## Continuous Integration

The test suite is designed to work in CI/CD environments:

- **No external dependencies** required for basic functionality
- **Temporary file handling** prevents conflicts
- **Comprehensive error handling** for missing optional packages
- **Configurable verbosity** and output formats

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the Doppelganger package is in the Python path
2. **Missing Dependencies**: Install pytest and related packages
3. **File Permissions**: Ensure test data files are readable
4. **Temporary Directory Issues**: Tests clean up temporary files automatically

### Debug Mode
```bash
# Run with maximum verbosity and no capture
pytest test/ -v -s --tb=long

# Run single test for debugging
pytest test/test_etl.py::TestETLFunctionality::test_specific_function -v -s
```

## Contributing

When adding new functionality to Doppelganger:

1. **Add corresponding tests** to the appropriate test file
2. **Update test data** if new data formats are needed
3. **Add configuration examples** for new features
4. **Update this documentation** with new test categories
5. **Ensure all tests pass** before submitting changes
