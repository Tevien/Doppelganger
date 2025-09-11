import os
import lightning.pytorch as pl
import torch
from torch import utils
from lightning.pytorch.loggers import TensorBoardLogger

from dpplgngr.utils.utils_data import get_dataset
from dpplgngr.utils.utils_train import get_model
from dpplgngr.etl.prep_dataset_tabular import ImputeScaleCategorize

import logging
import json
import luigi
import os

logger = logging.getLogger('luigi-interface')

# meta data
__author__ = 'SB'
__date__ = '2023-09-25'

class TrainModel(luigi.Task):
    lu_output_path = luigi.Parameter(default='model.pt')
    train_config = luigi.Parameter(default="config/train.json")

    def requires(self):
        return ImputeScaleCategorize()
    
    def output(self):
        with open(self.train_config, 'r') as f:
            input_json = json.load(f)
        prefix = f"results/{input_json['name']}"
        return luigi.LocalTarget(os.path.join(prefix, self.lu_output_path))
    
    def run(self):
        with open(self.train_config, 'r') as f:
            input_json = json.load(f)
        
        # Make output dir
        abs_path = input_json["absolute_path"]
        res_dir = f"results/{input_json['name']}"
        if not os.path.isdir(os.path.join(abs_path, res_dir)):
            os.makedirs(os.path.join(abs_path, res_dir))
        
        # Setup logging to output dir
        logging.basicConfig(filename=os.path.join(abs_path, res_dir, "train.log"), level=logging.INFO)
        # Setup tensorboard logging
        loggerTB = TensorBoardLogger(save_dir=os.path.join(abs_path, res_dir, "tensorboard_logs"),
                                     name=f"{input_json['name']}_TB")
        
        # Get data
        data_args = input_json["data_args"]
        data, data_val = get_dataset(input_json["datatype"], data_args)

        # Get model
        number_features = data.X.shape[1]
        print(f"Number of features: {number_features}")
        model_name = input_json["model"]
        model_args = input_json["model_args"]
        model_args["input_size"] = number_features
        model = get_model(model_name, model_args)

        train_loader = utils.data.DataLoader(data)
        val_loader = utils.data.DataLoader(data_val)

        trainer = pl.Trainer(limit_train_batches=input_json["batchlim"], 
                             max_epochs=input_json["epochs"],
                             logger=loggerTB, 
                             limit_val_batches=input_json["batchlim"] if input_json["batchlim"] > 0 else None)
        trainer.fit(model, train_loader, val_loader)

        # Save final trained model
        torch.save({'model_state_dict': model.state_dict()}, 
                    os.path.join(abs_path, res_dir, self.lu_output_path))

if __name__ == '__main__':
    luigi.build([TrainModel()], workers=1, local_scheduler=True)