# Doppelganger

Doppelganger is a package designed for the creation of digital twin models and development of synthetic data
from clinical data sources.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── config             <- JSON files for preprocessing and training configurations
    ├── doppelganger       <- Source code for use in this project.
    │   ├── etl            <- Luigi pipeline steps for data processing
    │   ├── models         <- Custom PyTorch model classes
    │   └── scores         <- Heart failure scoring functions
    │   └── train          <- PyTorch lightning training modules
    |   └── utils          <- Utility functions
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    |
    ├── results            <- Outputs from training and processing (model pth files, ROCs, AUCs, etc.)
    │
    ├── scripts            <- Source code for use in this project.
    │
    └── setup.py           <- makes project pip installable (pip install -e .) so src can be imported


--------

Installation
------------

### Option 1: Using uv (Recommended for Development)

**Step 1: Install uv**
```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or with pip
pip install uv
```

**Step 2: Clone and setup the project**
```bash
git clone <repository-url>
cd Doppelganger
uv venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install -e .
```

### Option 2: Using the Pre-built Container

If you have a pre-built container image (`doppelganger.sif`), you can skip the installation and use it directly:

```bash
# Assumes doppelganger.sif is available in your working directory
apptainer shell doppelganger.sif
source /opt/python/bin/activate
```

Configuration
-------------

The pipeline requires two main configuration files:

### 1. ETL Configuration (`etl.json`)

This file controls data preprocessing and transformation steps:

```json
{
    "name": "name",
    "description": "dataset description",

    "absolute_path": "INPUT DATA FOLDER",
    "preprocessing": "WHERE WILL THE INTERMEDIATE AND OUTPUT FILES BE",

    "filename_type1.parquet": ["pseudo_id", "col1", "col2", "ref_date"],
    "filename_type2.parquet": ["pseudo_id", "value", {"type_id": {"type1_id": "name1", "type2_id": "name2"}}, ["date_column_name"]],
  

    "final_cols": ["pseudo_id", "col1", "col2", "name1", "name2", "ref_date"],

    "tuple_vals_after": ["name1", "name2"],
    "tuple_vals_anybefore": [],
    "ref_date": "ref_date",

    "InitTransforms": {
    },
    "PreTransforms": {
    },
    "MergedTransforms": {
    }
}
```
Note that 3 types of transformations can be specified:
- `InitTransforms`: Applied to individual files on load
- `PreTransforms`: Applied to individual files after initial processing
- `MergedTransforms`: Applied to the final merged dataframe
See the functions in the 'dpplgngr.utils.functions' module for available transformations.

### 2. Synthesis Configuration (`synth.json`)

This file controls synthetic data generation parameters:

```json
{
    "input_file": "FILE LOCATION",
    "working_dir": "OUTPUT DIRECTORY",
    "columns": ["col1", "col2", "name1", "name2"],
    "num_points": 10000,
    "synth_type": "GC"
}
```

**Available synthesis types:**
- `GC`: Gaussian Copula
- `CTGAN`: Conditional Tabular GAN 
- `TVAE`: Tabular Variational Autoencoder
- `RTF`: RealTabFormer - GPT2
- `GNN`: Graph Neural Network (for relational data) **Coming Soon**

Usage
-----

### Running the Complete Pipeline

**Option 1: Using uv environment**
```bash
# Activate your environment
source venv/bin/activate

# Run the complete pipeline
python scripts/run_amc.py --etlconfig config/etl_amc_v3.json --genconfig config/synth.json
```

**Option 2: Using the container**
```bash
# Run complete pipeline in container
apptainer run doppelganger.sif etl config/etl_amc_v3.json config/synth.json
```

### Running Individual Pipeline Steps

You can also run individual Luigi tasks:

**1. Data Conversion and Preprocessing**
```python
import luigi
from dpplgngr.etl.prep_dataset_tabular import ConvertLargeFiles, PreProcess

# Convert and preprocess data
luigi.build([
    ConvertLargeFiles(etl_config="config/etl_amc_v3.json"),
    PreProcess(etl_config="config/etl_amc_v3.json")
], local_scheduler=True)
```

**2. Synthetic Data Generation**
```python
import luigi
from dpplgngr.train.sdv import SDVGen

# Generate synthetic data
luigi.build([
    SDVGen(
        gen_config="config/synth.json",
        etl_config="config/etl_amc_v3.json"
    )
], local_scheduler=True)
```

**3. Data Quality Audit**
```python
import luigi
from dpplgngr.scores.audit import SyntheticDataAudit

# Audit synthetic data quality
luigi.build([
    SyntheticDataAudit(
        gen_config="config/synth.json",
        etl_config="config/etl_amc_v3.json"
    )
], local_scheduler=True)
```

### Container Commands

**Individual steps using container:**
```bash
# Data preprocessing only
apptainer exec doppelganger.sif python -c "
import luigi
from dpplgngr.etl.prep_dataset_tabular import ConvertLargeFiles, PreProcess
luigi.build([ConvertLargeFiles(etl_config='config/etl_amc_v3.json'), PreProcess(etl_config='config/etl_amc_v3.json')], local_scheduler=True)
"

# Synthetic data generation only
apptainer run doppelganger.sif synth config/synth.json config/etl_amc_v3.json

# Data audit only
apptainer run doppelganger.sif audit config/synth.json config/etl_amc_v3.json

# Run tests
apptainer run doppelganger.sif test
```

### Binding Data Directories

When using containers, bind your data directories:

```bash
# Bind local data directory to container
apptainer exec -B /local/data:/data -B /local/output:/output doppelganger.sif \
    python scripts/run_amc.py --etlconfig /data/config/etl_amc_v3.json --genconfig /data/config/synth.json
```

## Synthetic Data Audit

The package includes a comprehensive auditing system for evaluating synthetic data quality using the `SyntheticDataAudit` Luigi task.

### Features

- **SDV Quality Evaluation**: Uses the Synthetic Data Vault (SDV) library to compute overall quality scores
- **Missing Data Analysis**: Compares missing value patterns between original and synthetic data
- **Distribution Analysis**: Visual comparison of numeric and categorical distributions
- **Statistical Summary**: Side-by-side comparison of descriptive statistics
- **Correlation Analysis**: Heatmaps showing correlation matrices and differences

### Usage

The audit functionality is integrated as a Luigi task that automatically depends on the synthetic data generation step:

```python
from dpplgngr.scores.audit import SyntheticDataAudit
import luigi

# Run the audit task
luigi.build([
    SyntheticDataAudit(
        gen_config="config/synth.json",
        etl_config="config/etl.json"
    )
], local_scheduler=True)
```

Or use the provided script:

```bash
python run_audit.py
```

### Outputs

The audit task produces:
- `audit_report_{synth_type}.json`: Comprehensive quality metrics and statistics
- `audit_plots_{synth_type}/`: Directory containing visualization plots
  - `missing_values_comparison.png`
  - `numeric_distributions.png`
  - `categorical_distribution_{column}.png`
  - `correlation_matrices.png`
  - `correlation_difference.png`

### Configuration

The audit task uses the same configuration files as the synthetic data generation task:
- `gen_config`: Points to the synthesis configuration JSON
- `etl_config`: Points to the ETL configuration JSON

## Building the Container (Optional)

If you need to build the container from scratch:

```bash
# Build the container image (requires Apptainer/Singularity)
./build_container.sh

# Or specify a custom build directory
./build_container.sh /path/to/build/directory

# Force rebuild if container exists
./build_container.sh --force

# Skip tests during build
./build_container.sh --skip-tests
```

The build script (`build_container.sh`) provides several options:
- Automatic dependency checking
- Build directory specification
- Force rebuild option
- Container testing after build
- Usage examples and help

## Container Usage

The Doppelganger package can be containerized using Apptainer (Singularity) for reproducible execution across different environments. The container uses **uv** for fast Python package management instead of conda, resulting in faster builds and smaller images.

### Building the Container

```bash
# Build the container image (requires Apptainer/Singularity)
apptainer build doppelganger.sif doppelganger.def
```

The container build process:
1. Uses Ubuntu 22.04 as the base image
2. Installs system dependencies and Python 3.10
3. Installs **uv** for fast Python package management
4. Creates a virtual environment at `/opt/python`
5. Installs all Python packages from `requirements-container.txt` using uv
6. Copies the Doppelganger package and configurations

### Running the Container

The container provides several convenient commands:

```bash
# Interactive shell
apptainer shell doppelganger.sif

# Run ETL pipeline
apptainer run doppelganger.sif etl [etl_config] [gen_config]

# Run synthetic data generation
apptainer run doppelganger.sif synth [gen_config] [etl_config]

# Run data audit
apptainer run doppelganger.sif audit [gen_config] [etl_config]

# Run test suite
apptainer run doppelganger.sif test

# Direct execution
apptainer exec doppelganger.sif python /opt/doppelganger/scripts/run_amc.py
```

### Container Structure

- `/opt/doppelganger/` - Main application directory
- `/opt/python/` - Python virtual environment (managed by uv)
- `/opt/doppelganger/config/` - Configuration files
- `/opt/doppelganger/data/` - Data directory
- `/opt/doppelganger/output/` - Output directory
- `/opt/doppelganger/logs/` - Log files

