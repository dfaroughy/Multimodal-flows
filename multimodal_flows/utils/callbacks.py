import os
import json, yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from copy import deepcopy

from pytorch_lightning.callbacks import Callback, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.utilities import rank_zero_only
from timm.utils.model_ema import ModelEmaV2

from utils.helpers import SimpleLogger as log
from utils.tensorclass import TensorMultiModal

class FlowGeneratorCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experiment_dir = Path(f'{config.dir}/{config.project}/{config.experiment_id}')
        self.tag = f"_{config.tag}" if config.tag else ""

    def on_predict_start(self, trainer, pl_module):
        self.batched_data = []

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.batched_data.append(outputs)

    def on_predict_end(self, trainer, pl_module):
        rank = trainer.global_rank
        self._save_results_local(rank)
        trainer.strategy.barrier()  # wait for all ranks to finish

        if trainer.is_global_zero:
            self._gather_results_global(trainer)
            self._clean_temp_files()

    def _save_results_local(self, rank):
        data = TensorMultiModal.cat(self.batched_data, dim=0)
        data.save_to(f"{self.experiment_dir}/temp_data{self.tag}_{rank}.h5")

    @rank_zero_only
    def _gather_results_global(self, trainer):
    
        os.mkdir(f'{self.experiment_dir}/generation_results{self.tag}')

        with open(f'{self.experiment_dir}/generation_results{self.tag}/configs.yaml' , 'w' ) as outfile:
            yaml.dump( self.config.__dict__, outfile, sort_keys=False)

        temp_files = self.experiment_dir.glob(f"temp_data{self.tag}_*.h5")
        sample = TensorMultiModal.cat([TensorMultiModal.load_from(str(f)) for f in temp_files], dim=0)

        if sample.has_continuous: # post-process continuous features

            mu = torch.tensor(self.config.metadata['mean'])
            sig = torch.tensor(self.config.metadata['std'])
            sample.continuous = (sample.continuous * sig) + mu

        sample.apply_mask()  # ensure mask is applied to pads
        sample.save_to(f'{self.experiment_dir}/generation_results{self.tag}/generated_sample.h5')

    def _clean_temp_files(self):
        for f in self.experiment_dir.glob(f"temp_data{self.tag}_*.h5"):
            f.unlink()


class GPTGeneratorCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experiment_dir = Path(f'{config.dir}/{config.project}/{config.experiment_id}')
        self.tag = f"_{config.tag}" if config.tag else ""

    def on_predict_start(self, trainer, pl_module):
        self.batched_data = []

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.batched_data.append(outputs)

    def on_predict_end(self, trainer, pl_module):
        rank = trainer.global_rank
        self._save_results_local(rank)
        trainer.strategy.barrier()  # wait for all ranks to finish

        if trainer.is_global_zero:
            self._gather_results_global(trainer)
            self._clean_temp_files()

    def _save_results_local(self, rank):
        data = torch.cat(self.batched_data, dim=0)
        path = f"{self.experiment_dir}/temp_data_{rank}.pt"
        torch.save(data, path)

    @rank_zero_only
    def _gather_results_global(self, trainer):
    
        os.mkdir(f'{self.experiment_dir}/generation_results{self.tag}')

        with open(f'{self.experiment_dir}/generation_results{self.tag}/configs.yaml' , 'w' ) as outfile:
            yaml.dump( self.config.__dict__, outfile, sort_keys=False)

        temp_files = self.experiment_dir.glob(f"temp_data_*.pt")
        sample = torch.cat([torch.load(str(f)) for f in temp_files], dim=0)
        np.save(f'{self.experiment_dir}/generation_results{self.tag}/sample.npy', sample)
        print(f'INFO: first event: {sample[0]}')

    def _clean_temp_files(self):
        for f in self.experiment_dir.glob(f"temp_data_*.pt"):
            f.unlink()


class TrainLoggerCallback(Callback):
    """
    Callback to log epoch-level metrics (from training_step/validation_step outputs)
    *and* to log the average gates at the end of each training epoch.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epoch_metrics = {"train": {}, "val": {}}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._track_metrics("train", outputs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._track_metrics("val", outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_epoch_metrics("train", pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_epoch_metrics("val", pl_module)

    def _track_metrics(self, stage, outputs):
        for key, val in outputs.items():
            self.epoch_metrics[stage].setdefault(key, []).append(
                val.detach().cpu().item() if isinstance(val, torch.Tensor) else float(val)
            )

    def _log_epoch_metrics(self, stage, pl_module):
        for key, vals in self.epoch_metrics[stage].items():
            epoch_avg = sum(vals) / len(vals)
            pl_module.log(
                key if stage=="train" else f"val_{key}",
                epoch_avg,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        self.epoch_metrics[stage].clear()


class EMACallback(Callback):
    def __init__(self, decay=0.9999, swap_on_val: bool = True):
        super().__init__()
        self.decay = decay
        self.swap_on_val = swap_on_val
        self.ema_model = None
        self._backup = None

    def setup(self, trainer, pl_module, stage=None):
        # 1) find the device your model lives on
        device = next(pl_module.model.parameters()).device  # torch.device('cuda:0'), etc.
        # 2) build the EMA copy on *that* same device
        self.ema_model = ModelEmaV2(
            pl_module.model, 
            decay=self.decay, 
            device=str(device)       # e.g. "cuda:0"
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.ema_model.update(pl_module.model)

    def on_validation_start(self, trainer, pl_module):
        if not self.swap_on_val:
            return
        self._backup = deepcopy(pl_module.model.state_dict())
        pl_module.model.load_state_dict(self.ema_model.module.state_dict())

    def on_validation_end(self, trainer, pl_module):
        if not self.swap_on_val or self._backup is None:
            return
        pl_module.model.load_state_dict(self._backup)
        self._backup = None




class ProgressBarCallback(Callback):
    """
    Callback to customize the progress bar theme.
    """

    def __init__(self):
        super().__init__()

        self.theme = RichProgressBarTheme(description="green_yellow",
                                        progress_bar="green1",
                                        progress_bar_finished="green1",
                                        progress_bar_pulse="#6206E0",
                                        batch_progress="green_yellow",
                                        time="grey82",
                                        processing_speed="grey82",
                                        metrics="grey82",
                                        metrics_text_delimiter="\n",
                                        metrics_format=".3e")

    def on_train_start(self, trainer, pl_module):
        trainer.progress_bar = RichProgressBar(theme=self.theme)

    def on_validation_start(self, trainer, pl_module):
        trainer.progress_bar = RichProgressBar(theme=self.theme)
    
    def on_predict_start(self, trainer, pl_module):
        trainer.progress_bar = RichProgressBar(theme=self.theme)


