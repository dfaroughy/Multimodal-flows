import yaml
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Union
from pipeline.helpers import SimpleLogger as log


@dataclass
class CometLoggerConfig:
    project_name: str
    save_dir: Optional[str] = None
    experiment_name: str = None
    workspace: str = "dfaroughy"
    api_key: str = "8ONjCXJ1ogsqG1UxQzKxYn7tz"


@dataclass
class DataConfig:
    modality: str = None
    batch_size: int = None
    split_ratios: List[float] = None
    transform: str = None
    continuous_features: List[str] = None
    discrete_features: str = None
    target_name: str = None
    target_path: str = None
    target_train_files: Optional[List[str]] = None
    target_test_files: Optional[List[str]] = None
    source_name: str = None
    source_path: str = None
    source_train_files: Optional[List[str]] = None
    source_test_files: Optional[List[str]] = None
    num_jets: Optional[int] = None
    min_num_particles: int = None
    max_num_particles: int = None
    dim_continuous: Optional[int] = 0
    dim_discrete: Optional[int] = 0
    dim_context_continuous: Optional[int] = 0
    dim_context_discrete: Optional[int] = 0
    vocab_size: Optional[int] = 0
    vocab_freq: Optional[List[float]] = None
    vocab_size_context: Optional[int] = 0
    num_workers: Optional[int] = 0
    pin_memory: Optional[bool] = False


@dataclass
class EncoderConfig:
    name: str
    num_blocks: Optional[int]
    dim_hidden_local: Optional[int]
    num_heads: Optional[int] = None # for transformer
    dim_hidden_glob: Optional[int] = None
    skip_connection: Optional[bool] = True
    dropout: Optional[float] = 0.0
    data_augmentation: Optional[bool] = False
    dim_emb_time: Optional[int] = 0
    dim_emb_continuous: Optional[int] = 0
    dim_emb_discrete: Optional[int] = 0
    dim_emb_context_continuous: Optional[int] = 0
    dim_emb_context_discrete: Optional[int] = 0


@dataclass
class TrainerConfig:
    max_epochs: int
    optimizer_name: str
    optimizer_params: Dict[str, Union[float, List[float], bool]]
    scheduler_name: Optional[str] = None
    scheduler_params: Optional[Dict[str, Union[float, List[float], bool]]] = None
    gradient_clip_val: float = 1.0


@dataclass
class ModelConfig:
    bridge_continuous: Optional[str] = None
    bridge_discrete: Optional[str] = None
    solver_continuous: Optional[str] = None
    solver_discrete: Optional[str] = None
    sigma: Optional[float] = None
    gamma: Optional[float] = None
    thermostat_fn: Optional[str] = None
    loss_weights: Optional[List[float]] = 'fixed'
    num_timesteps: int = None
    time_eps: float = 0.0
    topk: int = None
    save_path_history_snapshots: Optional[List[int]] = None


@dataclass
class CheckpointsConfig:
    dirpath: Optional[str] = None
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 1
    filename: str = "best"
    save_last: bool = True
    patience: Optional[int] = None
    stopping_threshold: Optional[float] = None


@dataclass
class ExperimentConfigs:
    path: str = None
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    checkpoints: CheckpointsConfig = field(default_factory=CheckpointsConfig)
    comet_logger: CometLoggerConfig = field(default_factory=CometLoggerConfig)

    def __init__(self, config):
        self._load_config(config)
        assert self.encoder.dim_emb_time > 0, "Non-zero dim_emb_time must be provided."

        if self.data.continuous_features is not None:
            self.data.dim_continuous = len(self.data.continuous_features)

        if self.data.discrete_features is not None:
            assert self.data.vocab_size > 0, (
                "vocab_size must be provided for discrete features"
            )

        if self.data.discrete_features == "onehot":
            self.data.dim_continuous += self.data.vocab_size

        if self.data.modality == "multi-modal" or self.data.modality == "discrete":
            self.data.dim_discrete = 1

    def update(self, config):
        if isinstance(config, dict):
            for key in config:
                if key == "path":
                    self.path = config["path"]
                else:
                    for sub_key, value in config[key].items():
                        setattr(getattr(self, key), sub_key, value)

    def _load_config(self, config: Union[dict, str]):
        if isinstance(config, str):
            with open(config, "r") as file:
                config = yaml.safe_load(file)
        self.data = DataConfig(**config["data"])
        self.encoder = EncoderConfig(**config["encoder"])
        self.trainer = TrainerConfig(**config["trainer"])
        self.model = ModelConfig(**config["model"])
        self.checkpoints = CheckpointsConfig(**config["checkpoints"])
        self.comet_logger = CometLoggerConfig(**config["comet_logger"])

    def print(self, indent=0):
        for key, value in self.__dict__.items():
            print(f"{' ' * indent}{key}:")
            if isinstance(value, dict):
                for k, v in value.__dict__.items():
                    if isinstance(v, dict):
                        for k_, v_ in v.items():
                            print(f"{' ' * (indent + 8)}{k_}: {v_}")
                    else:
                        print(f"{' ' * (indent + 4)}{k}: {v}")
            else:
                print(f"{' ' * (indent + 1)}{value}")

    def save(self, path, name="config.yaml"):
        """
        Saves the configuration parameters to a YAML file.
        """
        path = os.path.join(path, name)
        if not os.path.exists(path):
            log.info(f"Saving configuration to {path}")
            with open(path, "w") as f:
                yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self):
        return asdict(self)


progress_bar = {
    "description": "green_yellow",
    "progress_bar": "green1",
    "progress_bar_finished": "green1",
    "progress_bar_pulse": "#6206E0",
    "batch_progress": "green_yellow",
    "time": "grey82",
    "processing_speed": "grey82",
    "metrics": "grey82",
    "metrics_text_delimiter": "\n",
    "metrics_format": ".3e",
}
