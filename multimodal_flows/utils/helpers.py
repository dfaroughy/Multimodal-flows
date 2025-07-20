import json
import yaml
import os
import warnings
import torch.distributed as dist
import pytorch_lightning as L
import argparse

from pathlib import Path
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def set_logger(config):

    logger = CometLogger(api_key=config.comet_api_key,
                         project=config.project,
                         workspace=config.comet_workspace,
                         offline_directory=config.dir,
                         experiment_key=config.experiment_id if config.experiment_id else None
                         )

    if config.experiment_id is None: # if new experiment
        
        logger.experiment.log_parameters(vars(config))
        logger.experiment.add_tags(config.tags)

        if logger.experiment.get_key() is not None:

            config.experiment_id = logger.experiment.get_key()
            path = f"{config.dir}/{config.project}/{config.experiment_id}"
            os.makedirs(path, exist_ok=True)

            with open(os.path.join(path, "config.yaml"), "w") as f:
                yaml.safe_dump(vars(config), f, sort_keys=False, default_flow_style=False)

    return logger



def load_from_experiment(config_path):

    with open(f"{config_path}/config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
        config = argparse.Namespace(**config_dict)

    return config


def get_rank():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return dist.get_rank()


class SimpleLogger:
    @staticmethod
    def info(message, condition=True):
        if condition:
            print("\033[94m\033[1mINFO: \033[0m\033[00m", message)
        return

    @staticmethod
    def warn(message, condition=True):
        if condition:
            print("\033[31m\033[1mWARNING: \033[0m\033[00m", message)
        return

    @staticmethod
    def warnings_off():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)


def get_unique_dir(base_dir, exist_ok=False):
    """Returns a unique directory path by appending an integer suffix if needed."""
    if os.path.exists(base_dir) and not exist_ok:
        counter = 1
        new_dir = f"{base_dir}_{counter}"
        while os.path.exists(new_dir):
            counter += 1
            new_dir = f"{base_dir}_{counter}"
        return new_dir
    return base_dir


def setup_logging_dir(base_dir, exist_ok=False):
    """
    In a distributed setting, only the rank 0 process creates the directory.
    The unique directory path is then broadcasted to all processes.
    """
    unique_dir = None

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        unique_dir = get_unique_dir(base_dir, exist_ok=exist_ok)
        os.makedirs(unique_dir, exist_ok=True)

    if dist.is_available() and dist.is_initialized():
        unique_dir_list = [unique_dir if unique_dir is not None else ""]
        dist.broadcast_object_list(unique_dir_list, src=0)
        unique_dir = unique_dir_list[0]

    return Path(unique_dir)
