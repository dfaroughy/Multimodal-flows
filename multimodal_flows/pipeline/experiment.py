from comet_ml import ExistingExperiment
from pytorch_lightning.loggers import CometLogger
import os
from typing import List, Union
import lightning.pytorch as L
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities import rank_zero_only

from pipeline.configs import ExperimentConfigs, progress_bar
from pipeline.helpers import SimpleLogger as log
from model.multimodal_bridge_matching import MultiModalBridgeMatching
from pipeline.callbacks_trainer import ModelCheckpointCallback, ExperimentLoggerCallback
from pipeline.callbacks_generator import JetGeneratorCallback


class ExperimentPipeline:
    """
    A robust pipeline for configuring, training, and testing PyTorch Lightning models.
    """

    def __init__(
        self,
        datamodule: L.LightningDataModule,
        config: str = None,
        experiment_path: str = None,
        load_ckpt: str = "last.ckpt",
        accelerator: str = "gpu",
        strategy: str = "ddp",
        devices: str = "auto",
        num_nodes: int = 1,
        sync_batchnorm: bool = False,
        tags: Union[str, List[str]] = None,
        config_update: dict = None
    ):
        """
        Initialize the pipeline with configurations and components.

        Args:
            config (str): Path to the config file (used for training from scratch).
            experiment_path (str): Path to a saved experiment for resuming/inference training (optional).
            load_checkpoint (str): Name of the checkpoint file to load only if experiment_path is provided.
            accelerator (str): Type of accelerator to use (e.g., "gpu").
            strategy (str): Training strategy (e.g., "ddp").
            devices (str): Devices to use (e.g., "auto").
            num_nodes (int): Number of nodes for distributed training.
            log_every_n_steps (int): Logging frequency.
            val_check_interval (float): Validation check interval.
            config_update (dict): override model/data/train config matching keys.
        """

        self.datamodule = datamodule
        self.accelerator = accelerator
        self.strategy = strategy
        self.devices = devices
        self.num_nodes = num_nodes
        self.sync_batchnorm = sync_batchnorm
        self.ckpt_path = None
        self.config_update = config_update
        self.tags = tags

        if not experiment_path and config:
            log.info("starting new experiment!")
            self.config = self._load_config(config)
            self.model = self._setup_model()
            self.logger = self._setup_logger()
        else:
            log.info("loading from existing experiment.")
            assert experiment_path, "provide experiment path"
            assert load_ckpt, "provide checkpoint name to load"
            self.config_update = config
            self.experiment_path = experiment_path
            self.ckpt_path = os.path.join(experiment_path, "checkpoints", load_ckpt)
            self.model = MultiModalBridgeMatching.load_from_checkpoint(self.ckpt_path)
            self.config = self.model.config
            self.logger = self._setup_logger(new_experiment=False)

        self.config.update(self.config_update)
        self.callbacks = self._setup_callbacks_list()
        self.datamodule = self._setup_datamodule()

    def train(self):
        """
        Only train the model using the configured Trainer and DataModule.
        """

        self.trainer = self._setup_trainer()

        if self.new_experiment:
            self.trainer.config = self.config
            self.trainer.metadata = self.metadata

        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.ckpt_path,
        )

    def generate(self):
        """
        Generate new target data from (pre) trained model using test source data.
        """
        self.generator = self._setup_trainer()
        self.generator.predict(
            self.model,
            datamodule=self.datamodule,
            ckpt_path=self.ckpt_path,
        )

    # ...helper methods

    @staticmethod
    def _load_config(config: Union[str, ExperimentConfigs]) -> ExperimentConfigs:
        """
        Load experiment configurations from the given file path.
        """
        if isinstance(config, ExperimentConfigs):
            return config
        return ExperimentConfigs(config)

    def _setup_logger(self, new_experiment=True) -> Union[CometLogger, None]:
        """
        Set up the logger based on experiment configuration.
        Logger is initialized only on rank 0 for distributed training.
        """
        self.new_experiment = new_experiment
        if hasattr(self.config, "comet_logger"):
            if self.new_experiment:
                return self._setup_comet_logger_new()
            else:
                return self._setup_comet_logger_existing()
        return None

    @rank_zero_only
    def _setup_comet_logger_new(self) -> CometLogger:
        """
        Initialize a new CometLogger instance for a new experiment.
        """
        tags = [self.config.data.modality, f"{self.num_nodes} nodes"]
        tags += self.tags if isinstance(self.tags, list) else [self.tags]

        logger = CometLogger(**self.config.comet_logger.__dict__)
        logger.experiment.log_parameters(self.config.to_dict())
        logger.experiment.add_tags(tags)

        self.config.comet_logger.experiment_key = logger.experiment.get_key()
        path = os.path.join(
            self.config.comet_logger.save_dir,
            self.config.comet_logger.project_name,
            self.config.comet_logger.experiment_key,
        )
        self.config.update({"path": path})
        log.info(f"Experiment path: {self.config.path}")
        return logger

    @rank_zero_only
    def _setup_comet_logger_existing(self) -> CometLogger:
        """
        Reconnect to an existing Comet experiment for resuming training.
        """
        self.config.checkpoints.dirpath = os.path.join(
            self.experiment_path, "checkpoints"
        )
        experiment = ExistingExperiment(
            api_key=self.config.comet_logger.api_key,
            experiment_key=self.config.comet_logger.experiment_key,
        )
        experiment.log_parameters(self.config.to_dict())
        return CometLogger(**self.config.comet_logger.__dict__)

    def _setup_datamodule(self):
        """
        Prepare the data module for training and validation datasets.
        Saves metadata for later use.
        """
        data = self.datamodule(config=self.config)
        self.metadata = data.metadata
        return data

    def _setup_model(self) -> MultiModalBridgeMatching:
        """
        Set up the model using the loaded configurations.
        """
        return MultiModalBridgeMatching(self.config)

    def _setup_callbacks_list(self, transform=None) -> List[L.Callback]:
        """
        Configure and return the necessary callbacks for training.
        """
        callbacks = []
        callbacks.append(RichProgressBar(theme=RichProgressBarTheme(**progress_bar)))
        callbacks.append(ModelCheckpointCallback(self.config))

        if self.config.checkpoints.patience:
            callbacks.append(EarlyStopping(monitor=self.config.checkpoints.monitor, 
                                           mode=self.config.checkpoints.mode,
                                           patience=self.config.checkpoints.patience,
                                           stopping_threshold=self.config.checkpoints.stopping_threshold,
                                           check_finite=True
                                                ))
                                                
        callbacks.append(ExperimentLoggerCallback(self.config))
        callbacks.append(JetGeneratorCallback(self.config))

        return callbacks

    def _setup_trainer(self) -> L.Trainer:
        """
        Configure the PyTorch Lightning trainer dynamically.
        """
        trainer_config = {
            "max_epochs": self.config.trainer.max_epochs,
            "accelerator": self.accelerator,
            "strategy": self.strategy,
            "devices": self.devices,
            "gradient_clip_val": self.config.trainer.gradient_clip_val,
            "sync_batchnorm": self.sync_batchnorm,
            "callbacks": self.callbacks,
            "logger": self.logger,
            "num_nodes": self.num_nodes,
        }

        return L.Trainer(**trainer_config)
