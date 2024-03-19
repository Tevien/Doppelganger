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

Usage
-----

Installation and setup:
```shell script
git clone ...
cd doppelganger
conda create -p ./venv python=3.8 -y
conda activate ./venv
pip install -r requirements.txt
pip install -e .
```

