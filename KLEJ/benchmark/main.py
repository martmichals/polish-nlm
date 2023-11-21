import os
import yaml
import torch
import random
import argparse
import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
from benchmark.task import TASKS
from pytorch_lightning.loggers import WandbLogger
from benchmark.config import Config
from benchmark.dataset import Datasets
from pytorch_lightning.callbacks import ModelCheckpoint
from benchmark.model import KlejTransformer
from benchmark.trainer import TrainerWithPredictor

# High MM precision for tensor cores
torch.set_float32_matmul_precision('high')

# Set up the seed
def set_seed(seed: int, num_gpu: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if num_gpu > 0:
        torch.cuda.manual_seed_all(seed)

# Run the training
def main(args) -> None:
    # Read in config
    with open(args.config, 'r') as inf:
        config = Config.model_validate(
            yaml.safe_load(inf),
            strict=True
        )
    config.task_name = args.task_name

    # Set up seeds
    seed_everything(config.seed, workers=True)

    # TODO: Download artifacts, configure W&B run here

    # Initialize the task, model and dataset
    task = TASKS[config.task_name](config)
    datasets = Datasets(task)
    model = KlejTransformer(task, datasets)

    # Create a logger
    if config.logger_path is not None:
        if not os.path.exists(config.logger_path):
            os.makedirs(config.logger_path)
        logger = WandbLogger(
            name = config.run_id,
            save_dir = config.logger_path
        )
    else:
        logger = None

    # Create a trainer
    # TODO: Change some arguments here
    trainer = TrainerWithPredictor(
        devices=1,
        logger=logger,
        accelerator='gpu',
        deterministic=True,
        max_epochs=config.num_train_epochs,
        gradient_clip_val=config.max_grad_norm,
        log_every_n_steps=10,
        accumulate_grad_batches=config.gradient_accumulation_steps
    )
    trainer.fit(model)

    # predict
    pred = trainer.predict()['labels']
    pd.DataFrame({'target': pred}).to_csv(config.predict_path, index=False)

if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(
        description='Launch a fine-tuning run for HerBERT'
    )

    # Add the config file argument
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to the JSON config file to use for the run'
    )
    parser.add_argument(
        '--prefix', 
        type=str,
        help='Prefix to use in the run name'
    )
    parser.add_argument(
        '--task_name',
        type=str,
        help='Task name to run training for'
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Call main function to fine-tune the model
    main(args)
