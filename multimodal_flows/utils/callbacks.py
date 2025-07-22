import os
import json, yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
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
        # containers for batch‐level metrics
        self.epoch_metrics = {"train": {}, "val": {}}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # outputs is whatever your training_step returns
        self._track_metrics("train", outputs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._track_metrics("val", outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        # first log the standard epoch‐averaged metrics
        self._log_epoch_metrics("train", pl_module)

        # now compute & log the average gates across all layers
        gxs, gys = [], []
        for block in pl_module.model.transformer.attn_blocks:
            gxs.append(torch.sigmoid(block.gate_x))
            gys.append(torch.sigmoid(block.gate_y))
        avg_gx = torch.stack(gxs).mean()
        avg_gy = torch.stack(gys).mean()

        # log them—will show in prog bar and any logger (TB, WandB, etc.)
        pl_module.log("avg_gate_x", avg_gx, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        pl_module.log("avg_gate_y", avg_gy, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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
