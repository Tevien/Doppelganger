"""
Study Folder Configuration System

This module provides utilities to automatically discover and configure
data files and parameters based on a study folder structure.

Expected study folder structure:
    study_folder/
        preprocessing/
            preprocessed.parquet                    # Main preprocessed data
            preprocessed_tupleprocess.parquet       # After tuple processing (optional)
        imputed/
            imputed_mean.parquet                    # Alternative imputations
            imputed_median.parquet
            imputed_knn.parquet
            imputed_iterative.parquet
        scores/
            with_scores_*.parquet                   # Data with calculated scores
        evaluation/
            evaluation_report.json                  # Performance comparison
        config/
            maggic.json                             # Score configuration
            study_config.json                       # Study-level configuration

Author: SB
Date: 2025-10-31
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class StudyConfig:
    """
    Configuration manager for study folders.
    
    Automatically discovers data files and creates appropriate configurations
    for the pipeline.
    """
    
    def __init__(self, study_folder: str):
        """
        Initialize study configuration.
        
        Parameters
        ----------
        study_folder : str
            Path to the study folder
        """
        self.study_folder = Path(study_folder)
        
        if not self.study_folder.exists():
            raise ValueError(f"Study folder does not exist: {study_folder}")
        
        # Define standard subdirectories
        self.preprocessing_dir = self.study_folder / 'preprocessing'
        self.imputed_dir = self.study_folder / 'imputed'
        self.scores_dir = self.study_folder / 'scores'
        self.evaluation_dir = self.study_folder / 'evaluation'
        self.config_dir = self.study_folder / 'config'
        
        # Create directories if they don't exist
        for dir_path in [self.imputed_dir, self.scores_dir, self.evaluation_dir, self.config_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized study configuration for: {study_folder}")
    
    def get_preprocessed_file(self, prefer_tupleprocess: bool = True) -> Optional[str]:
        """
        Get the main preprocessed data file.
        
        Parameters
        ----------
        prefer_tupleprocess : bool
            If True, prefer tupleprocessed file over basic preprocessed
        
        Returns
        -------
        str or None
            Path to preprocessed file
        """
        # Look for tupleprocessed file first if preferred
        if prefer_tupleprocess:
            candidates = [
                'preprocessed_tupleprocess.parquet',
                'preprocessed_tupleprocess_graph_imputed.parquet',
            ]
            for candidate in candidates:
                file_path = self.preprocessing_dir / candidate
                if file_path.exists():
                    logger.info(f"Found tupleprocessed file: {file_path}")
                    return str(file_path)
        
        # Look for basic preprocessed file
        candidates = [
            'preprocessed.parquet',
            'preprocessed_imputed.parquet',
        ]
        for candidate in candidates:
            file_path = self.preprocessing_dir / candidate
            if file_path.exists():
                logger.info(f"Found preprocessed file: {file_path}")
                return str(file_path)
        
        logger.warning("No preprocessed file found")
        return None
    
    def get_imputed_files(self) -> Dict[str, str]:
        """
        Get all imputed data files.
        
        Returns
        -------
        dict
            Dictionary mapping imputation strategy to file path
        """
        imputed_files = {}
        
        if not self.imputed_dir.exists():
            logger.warning(f"Imputed directory does not exist: {self.imputed_dir}")
            return imputed_files
        
        # Look for imputed files
        for file_path in self.imputed_dir.glob('imputed_*.parquet'):
            strategy = file_path.stem.replace('imputed_', '')
            imputed_files[strategy] = str(file_path)
            logger.info(f"Found imputed file ({strategy}): {file_path}")
        
        return imputed_files
    
    def get_scored_files(self) -> Dict[str, str]:
        """
        Get all files with calculated scores.
        
        Returns
        -------
        dict
            Dictionary mapping label to file path
        """
        scored_files = {}
        
        if not self.scores_dir.exists():
            logger.warning(f"Scores directory does not exist: {self.scores_dir}")
            return scored_files
        
        # Look for files with scores
        for file_path in self.scores_dir.glob('with_scores_*.parquet'):
            label = file_path.stem.replace('with_scores_', '')
            scored_files[label] = str(file_path)
            logger.info(f"Found scored file ({label}): {file_path}")
        
        return scored_files
    
    def get_score_config(self, score_name: str = 'maggic') -> Optional[str]:
        """
        Get score configuration file.
        
        Parameters
        ----------
        score_name : str
            Name of the score (default: 'maggic')
        
        Returns
        -------
        str or None
            Path to score configuration file
        """
        config_file = self.config_dir / f'{score_name}.json'
        
        if config_file.exists():
            logger.info(f"Found score config: {config_file}")
            return str(config_file)
        
        logger.warning(f"Score config not found: {config_file}")
        return None
    
    def get_output_path(self, subdirectory: str, filename: str) -> str:
        """
        Get output path for a file in a subdirectory.
        
        Parameters
        ----------
        subdirectory : str
            Subdirectory name ('imputed', 'scores', 'evaluation')
        filename : str
            Output filename
        
        Returns
        -------
        str
            Full output path
        """
        subdir_path = self.study_folder / subdirectory
        subdir_path.mkdir(parents=True, exist_ok=True)
        return str(subdir_path / filename)
    
    def save_study_metadata(self, metadata: dict):
        """
        Save study metadata to config directory.
        
        Parameters
        ----------
        metadata : dict
            Metadata to save
        """
        metadata_file = self.config_dir / 'study_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved study metadata to: {metadata_file}")
    
    def load_study_metadata(self) -> Optional[dict]:
        """
        Load study metadata from config directory.
        
        Returns
        -------
        dict or None
            Study metadata if exists
        """
        metadata_file = self.config_dir / 'study_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def create_default_maggic_config(self, column_mapping: dict) -> str:
        """
        Create a default MAGGIC configuration file.
        
        Parameters
        ----------
        column_mapping : dict
            Mapping from MAGGIC variables to dataset columns
        
        Returns
        -------
        str
            Path to created configuration file
        """
        config = {
            "score_name": "maggic",
            "column_mapping": column_mapping
        }
        
        config_file = self.config_dir / 'maggic.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created MAGGIC config: {config_file}")
        return str(config_file)
    
    def get_pipeline_config(self) -> dict:
        """
        Get complete pipeline configuration.
        
        Returns
        -------
        dict
            Complete pipeline configuration with all file paths
        """
        config = {
            'study_folder': str(self.study_folder),
            'preprocessing': {
                'dir': str(self.preprocessing_dir),
                'file': self.get_preprocessed_file(),
            },
            'imputed': {
                'dir': str(self.imputed_dir),
                'files': self.get_imputed_files(),
            },
            'scores': {
                'dir': str(self.scores_dir),
                'files': self.get_scored_files(),
                'config': self.get_score_config(),
            },
            'evaluation': {
                'dir': str(self.evaluation_dir),
            },
        }
        
        return config
    
    def print_summary(self):
        """Print a summary of the study configuration."""
        print(f"\n{'='*60}")
        print(f"Study Folder Configuration")
        print(f"{'='*60}")
        print(f"Study folder: {self.study_folder}")
        print(f"\nPreprocessed data:")
        preprocessed = self.get_preprocessed_file()
        if preprocessed:
            print(f"  ✓ {preprocessed}")
        else:
            print(f"  ✗ Not found")
        
        print(f"\nImputed datasets:")
        imputed = self.get_imputed_files()
        if imputed:
            for strategy, path in imputed.items():
                print(f"  ✓ {strategy}: {path}")
        else:
            print(f"  ✗ None found")
        
        print(f"\nScored datasets:")
        scored = self.get_scored_files()
        if scored:
            for label, path in scored.items():
                print(f"  ✓ {label}: {path}")
        else:
            print(f"  ✗ None found")
        
        print(f"\nScore configuration:")
        score_config = self.get_score_config()
        if score_config:
            print(f"  ✓ {score_config}")
        else:
            print(f"  ✗ Not found")
        
        print(f"\n{'='*60}\n")


def discover_study_files(study_folder: str) -> dict:
    """
    Discover all relevant files in a study folder.
    
    Parameters
    ----------
    study_folder : str
        Path to study folder
    
    Returns
    -------
    dict
        Dictionary with discovered files
    """
    config = StudyConfig(study_folder)
    return config.get_pipeline_config()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python study_config.py <study_folder>")
        sys.exit(1)
    
    study_folder = sys.argv[1]
    config = StudyConfig(study_folder)
    config.print_summary()
