import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import lightning.pytorch as L
import json
import os

# from pipeline.helpers import SimpleLogger as log
# from pipeline.registry import registered_datasets as JetDataset
from datamodules.datasets import MultiModalDataset, data_coupling_collate_fn


class JetDataModule(L.LightningDataModule):
    """DataModule for handling source-target-context coupling for particle cloud data."""

    def __init__(
        self,
        config,
        ):
        super().__init__()

        self.config = config
        self.transform = config.data.transform
        self.features = {'continuous': config.data.continuous_features,
                         'discrete': config.data.discrete_features} 
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.pin_memory = config.data.pin_memory
        self.split_ratios = config.data.split_ratios

        self.num_jets = config.data.num_jets
        self.max_num_particles = config.data.max_num_particles
        self.metadata = {}

        name_source = config.data.source_name
        path_source = config.data.source_path
        train_files_source = config.data.source_train_files
        test_files_source = config.data.source_test_files

        name_target = config.data.target_name
        path_target = config.data.target_path
        train_files_target = config.data.target_train_files
        test_files_target = config.data.target_test_files

        if path_source:
            self.source_dataset_train = JetDataset[name_source](
                data_dir=path_source,
                data_files=train_files_source,
            )
            self.source_dataset_pred = JetDataset[name_source](
                data_dir=path_source,
                data_files=test_files_source,
            )

        if path_target:
            self.target_dataset_train = JetDataset[name_target](
                data_dir=path_target,
                data_files=train_files_target,
            )
            self.target_dataset_pred = JetDataset[name_target](
                data_dir=path_target,
                data_files=test_files_target,
            )

    #####################
    # Lightning methods #
    #####################

    def setup(self, stage=None):
        """Setup datasets for train, validation, and test splits."""

        if stage == "fit" or stage is None:
            log.info("Setting up datasets for training...")

            self._prepare_fit_datasets()
            dataset = MultiModalDataset(self)

            idx0, idx1 = self._idx_data_split(dataset, self.split_ratios)

            self.train_dataset = Subset(dataset, idx0) if idx0 else None
            self.val_dataset = Subset(dataset, idx1) if idx1 else None

        if stage == "predict":
            log.info("Setting up datasets for generation...")

            self._prepare_predict_datasets()
            self.predict_dataset = MultiModalDataset(self)

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def predict_dataloader(self):
        return self._get_dataloader(self.predict_dataset, shuffle=False)

    #####################
    # Data prep methods #
    #####################

    def _get_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            collate_fn=data_coupling_collate_fn,
        )

    def _prepare_fit_datasets(self):
        if hasattr(self, "source_dataset_train"):
            source, metadata = self.source_dataset_train(
                self.num_jets,
                self.max_num_particles,
                download=True,
                transform=self.transform,
                features=self.features,
            )
            self.source = source
            self.metadata["source"] = metadata 

        if hasattr(self, "target_dataset_train"):
            target, metadata = self.target_dataset_train(
                self.num_jets,
                self.max_num_particles,
                download=True,
                transform=self.transform,
                features=self.features,
            )
            self.target = target
            self.metadata["target"] = metadata

        self._clear_unavailable_modes()  # remove data modes that are not needed

    def _prepare_predict_datasets(self):
        if hasattr(self, "source_dataset_pred"):
            source, _ = self.source_dataset_pred(
                self.num_jets,
                self.max_num_particles,
                download=True,
                transform=self.transform,
                features=self.features,
            )
            self.source = source

        if hasattr(self, "target_dataset_pred"):
            target, _ = self.target_dataset_pred(
                self.num_jets,
                self.max_num_particles,
                download=True,
                features=self.features,
            )
            self.target = target

        self._clear_unavailable_modes()  # remove data modes that are not needed

    def _clear_unavailable_modes(self):
        if self.config.data.modality == "continuous":
            if hasattr(self, "source"):
                del self.source.discrete
            del self.target.discrete

        elif self.config.data.modality == "discrete":
            if hasattr(self, "source"):
                del self.source.continuous
            del self.target.continuous

        elif self.config.data.modality == "multi-modal":
            pass

        else:
            raise ValueError(
                "Invalid data modality. Specify 'continuous', 'discrete', or 'multi-modal'."
            )

    def _idx_data_split(self, dataset, ratios):
        assert np.abs(1.0 - sum(ratios)) < 1e-3, "Split ratios not sum to 1!"
        total_size = len(dataset)
        train_size = int(total_size * ratios[0])
        valid_size = int(total_size * ratios[1])
        idx = torch.arange(total_size)
        idx_train = idx[:train_size].tolist() if train_size > 0 else None
        idx_valid = (
            idx[train_size : train_size + valid_size].tolist()
            if valid_size > 0
            else None
        )
        return idx_train, idx_valid

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        log.info(f"Loading metadata from {metadata_file}.")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata
