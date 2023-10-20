import argparse
import os
import random
import typing as t

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from klejbenchmark_baselines.config import Config
from klejbenchmark_baselines.dataset import Datasets
from klejbenchmark_baselines.model import KlejTransformer
from klejbenchmark_baselines.task import TASKS
from klejbenchmark_baselines.trainer import TrainerWithPredictor

# High MM precision for tensor cores
torch.set_float32_matmul_precision('high')

def parse_bool(value: str) -> bool:
    if value.lower() in ['true', 't', 'yes', 'y', '1']:
        return True
    elif value.lower() in ['false', 'f', 'no', 'n', '0']:
        return False
    else:
        raise ValueError(f'Invalid argument value: {value}')


def parse_args() -> t.Dict[str, t.Any]:
    parser = argparse.ArgumentParser()

    # Task
    parser.add_argument(
        "--task-name", type=str, required=True, choices=TASKS.keys(),
        help="Name of the task.",
    )
    parser.add_argument(
        "--run-id", type=str, required=True,
        help="Unique identifier of the run.",
    )

    # Data
    parser.add_argument(
        "--task-path", type=str, required=True,
        help="Path to the task datasets.",
    )
    parser.add_argument(
        "--predict-path", type=str, required=True,
        help="Path to store predictions for the test set.",
    )
    parser.add_argument(
        "--logger-path", type=str, required=True,
        help="Path to store tensorboard logs.",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, required=True,
        help="Path to store saved model.",
    )

    # Tokenizer
    parser.add_argument(
        "--tokenizer-name-or-path", type=str, required=True,
        help="Name or path to the tokenizer.",
    )
    parser.add_argument(
        "--max-seq-length", type=int, required=False,
        help="Maximum length of the sequence.",
    )
    parser.add_argument(
        "--do-lower-case", type=parse_bool, required=False,
        help="Set this flag if you are using an uncased model.",
    )

    # Model
    parser.add_argument(
        "--model-name-or-path", type=str, required=True,
        help="Name or path to the model.",
    )
    parser.add_argument(
        "--learning-rate", type=float, required=False,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--adam-epsilon", type=float, required=False,
        help="Epsilon parameter for the ADAM optimizer.",
    )
    parser.add_argument(
        "--warmup-steps", type=int, required=False,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--batch-size", type=int, required=False,
        help="Number of rows in one batch.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, required=False,
        help="Number of batches for which to accumulate gradients.",
    )
    parser.add_argument(
        "--num-train-epochs", type=int, required=False,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--weight-decay", type=float, required=False,
        help="Value of the weight decay.",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, required=False,
        help="Maximum value of the gradient norm.",
    )

    # Other
    parser.add_argument(
        "--seed", type=int, required=False,
        help="Random seed.",
    )
    parser.add_argument(
        "--num-workers", type=int, required=False,
        help="Number of processes for the DataLoader. Set 0 for using the main process.",
    )
    parser.add_argument(
        "--num-gpu", type=int, required=False,
        help="Number of used GPUs. Set 0 for using CPU.",
    )

    return vars(parser.parse_args())


def set_seed(seed: int, num_gpu: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if num_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def run() -> None:
    # Parse arguments
    args_dict = parse_args()
    config = Config.from_argparse(args_dict)

    # Set up task
    task = TASKS[args_dict['task_name']](config)

    # Set up training, validation, and test sets
    datasets = Datasets(task)

    # Set up seeds
    set_seed(config.seed, config.num_gpu)

    # Initialize the model
    model = KlejTransformer(task, datasets)

    # Create a logger
    if not os.path.exists(config.logger_path):
        os.makedirs(config.logger_path)
    logger = WandbLogger(
        name = config.run_id,
        save_dir = config.logger_path
    )

    # Create a trainer
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
    run()
