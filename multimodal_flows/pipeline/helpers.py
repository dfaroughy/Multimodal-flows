import json
import os
import warnings
from pathlib import Path
import torch.distributed as dist


def get_from_json(key, path, name="metadata.json"):
    path = os.path.join(path, name)
    with open(path, "r") as f:
        file = json.load(f)
    return file[key]


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
