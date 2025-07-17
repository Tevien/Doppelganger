# -*- coding: utf-8 -*-
"""CT dataset"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

import os
from torch.utils.data import Dataset
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
import numpy as np


class Tabular(Dataset):
    """Tabular Dataset"""

    def __init__(self, filename, tr_test="train", 
                 foldgenseed=42,
                 index_id="subject_id",
                 label=None):
        """
        Initialization
        Args:
            filename: Path to the parquet file with annotations.
            fold: Which fold to return
            tr_test: Test or train
            foldgenseed: Seed for the fold generator
            index_id: Name of the index column
            label: Name of the label column
        """
        self.dd = dd.read_parquet(filename)
        self.dd.set_index(index_id)

        # If no label, make dummy array with same length as data
        if not label:
            y = np.zeros(len(self.dd))
        else:
            y = self.dd[label].values.compute()
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(self.dd.to_dask_array(lengths=True), y, test_size=0.2, 
                                                            random_state=foldgenseed,
                                                            shuffle=True)
        self.cols = self.dd.columns
        self.tr_test = tr_test
        if tr_test == "train":
            self.X = X_train
            self.y = y_train
        elif tr_test == "test":
            self.X = X_test
            self.y = y_test
        else:
            raise ValueError("tr_test must be 'train' or 'test'")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = np.array(self.X[idx].astype(np.float32)), np.array(self.y[idx].astype(np.float32))
        return sample
    
    def save_train_test(self, save_dir):
        """Save train and test data to parquet"""
        # Make array into dataframe
        dd_X = dd.from_array(self.X, columns=self.cols)
        dd_y = dd.from_array(self.y, columns=["label"])
        self.X.to_parquet(os.path.join(save_dir, f"{self.tr_test}_X.parquet"))
        self.y.to_parquet(os.path.join(save_dir, f"{self.tr_test}_y.parquet"))