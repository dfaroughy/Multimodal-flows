import json
import torch
from pathlib import Path
from typing import Dict

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only

from pipeline.configs import ExperimentConfigs
from pipeline.helpers import SimpleLogger as log


class ModelCheckpointCallback(ModelCheckpoint):
    """
    A wrapper around Lightning's ModelCheckpoint to initialize using ExperimentConfigs.
    """

    def __init__(self, config: ExperimentConfigs):
        """
        Initialize the callback using a configuration object.
        Args:
            config (ExperimentConfigs): The configuration object containing checkpoint settings.
        """
        args = config.checkpoints.__dict__.copy()
        del args['patience']
        del args['stopping_threshold']

        super().__init__(**args)


class ExperimentLoggerCallback(Callback):
    """
    Callback to log epoch-level metrics dynamically during training and validation,
    supporting additional custom metrics beyond loss.
    """

    def __init__(self, config: ExperimentConfigs):
        super().__init__()
        self.config = config
        self.sync_dist = False
        self.epoch_metrics = {"train": {}, "val": {}}

    def setup(self, trainer, pl_module, stage=None):
        """Set up distributed synchronization if required."""
        self.sync_dist = trainer.world_size > 1
        self.checkpoint_dir_created = False if hasattr(trainer, "config") else True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Accumulate metrics for epoch-level logging during training."""
        self._track_metrics("train", outputs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Accumulate metrics for epoch-level logging during validation."""
        self._track_metrics("val", outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        """Log metrics at the end of a training epoch."""
        self._log_epoch_metrics("train", pl_module, trainer)
        self._save_metada_to_dir(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log metrics at the end of a validation epoch."""
        self._log_epoch_metrics("val", pl_module, trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_train_end(trainer, pl_module)

    @rank_zero_only
    def _save_metada_to_dir(self, trainer):
        """Save metadata to the checkpoint directory"""

        while not self.checkpoint_dir_created and Path(trainer.config.path).exists():
            self.checkpoint_dir_created = True
            trainer.config.save(Path(trainer.config.path))

            with open(Path(trainer.config.path) / "metadata.json", "w") as f:
                json.dump(trainer.metadata, f, indent=4)

            log.info("Config file and metadata save to experiment path.")

    def _track_metrics(self, stage: str, outputs: Dict[str, torch.Tensor]):
        """
        Accumulate metrics for epoch-level logging.
        Args:
            stage (str): Either "train" or "val".
            outputs (Dict[str, Any]): Dictionary of metrics from the batch.
        """
        for key, value in outputs.items():
            if key not in self.epoch_metrics[stage]:
                self.epoch_metrics[stage][key] = []

            if isinstance(value, torch.Tensor):  # Handle tensor values
                self.epoch_metrics[stage][key].append(value.detach().cpu().item())

            elif isinstance(value, (float, int)):  # Handle float or int values
                self.epoch_metrics[stage][key].append(value)

            else:
                raise TypeError(
                    f"Unsupported metric type for key '{key}': {type(value)}"
                )

    def _log_epoch_metrics(self, stage: str, pl_module, trainer):
        """
        Compute and log metrics for the epoch, and log them using the Comet logger if available.
        Args:
            stage (str): Either "train" or "val".
            pl_module: LightningModule to log metrics.
            trainer: The Lightning Trainer instance.
        """
        epoch_metrics = {}
        for key, values in self.epoch_metrics[stage].items():
            epoch_metric = sum(values) / len(values)
            self.log(
                key,
                epoch_metric,
                on_epoch=True,
                logger=True,
                sync_dist=self.sync_dist,
            )
            epoch_metrics[key] = epoch_metric
            
        self.epoch_metrics[stage].clear()  # Reset for next epoch
