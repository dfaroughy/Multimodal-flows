import numpy as np
import torch
import h5py
import os
import urllib.request
import json
from torch.utils.data import DataLoader, Subset
import lightning.pytorch as L

from pipeline.helpers import SimpleLogger as log
from datamodules.datasets import MultiModalDataset, data_coupling_collate_fn
from tensorclass import TensorMultiModal


class AspenOpenJets:
    """data constructor for the Aspen OpenJets dataset."""

    def __init__(
        self,
        data_dir,
        data_files=None,
        url="https://www.fdr.uni-hamburg.de/record/16505/files",
    ):
        self.url = url
        self.data_dir = data_dir
        self.data_files = data_files

    def __call__(
        self,
        num_jets=None,
        max_num_particles=150,
        download=False,
        transform=None,
        features={"continuous": ["pt", "eta_rel", "phi_rel"], "discrete": "tokens"},
        pt_order=True,
        padding = 'zeros',
    ) -> TensorMultiModal:
        """
        Fetch the data, either from the provided files or by downloading it.
        Args:
            file_paths (list of str): List of absolute paths to the data files.
            download (bool): If True, downloads the file if not found.
        Returns:
            time, continuous, discrete, mask: Parsed data components.
        """

        if isinstance(self.data_files, str):
            self.data_files = [self.data_files]

        list_continuous_feats = []
        list_discrete_feats = []
        list_masks = []
        jet_count = 0
        self.pt_order = pt_order
        self.padding = padding

        if features["discrete"] == "onehot":
            if features["continuous"] is not None:
                features["continuous"].append("onehot")
            else:
                features["continuous"] = ["onehot"]
                
        for datafile in self.data_files:
            path = os.path.join(self.data_dir, datafile)

            if download:
                if not os.path.exists(path):
                    log.warn(f"File {datafile} not found. Downloading from {self.url}")
                    self._download_file(path)
                else:
                    log.info(f"File {datafile} already exists. Skipping download.")

            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"File {datafile} not found in {self.data_dir}."
                )

            feats, mask = self._read_aoj_file(path, num_jets)
            

            if features["continuous"]:  
                list_continuous_feats.append(
                    torch.cat([feats[x] for x in features["continuous"]], dim=-1)
                )

            if features["discrete"] == "tokens":
                list_discrete_feats.append(torch.tensor(feats[features["discrete"]]))

            list_masks.append(mask)

            # halt if enough jets:
            if num_jets:
                jet_count += len(list_masks[-1])
                if jet_count > num_jets:
                    break

        continuous = (
            torch.cat(list_continuous_feats, dim=0)[:num_jets, :max_num_particles, :]
            if len(list_continuous_feats)
            else None
        )

        discrete = (
            torch.cat(list_discrete_feats, dim=0)[:num_jets, :max_num_particles, :]
            if len(list_discrete_feats)
            else None
        )

        mask = torch.cat(list_masks, dim=0)[:num_jets, :max_num_particles, :]

        continuous, discrete, mask, metadata = self._preprocess(
            continuous, discrete, mask, transform
        )

        output = TensorMultiModal(None, continuous, discrete, mask)
        output.apply_mask()

        return output, metadata

    def _preprocess(self, continuous, discrete, mask, transform):
        metadata = self._extract_metadata(continuous, mask)

        if transform == "standardize":
            mean = torch.tensor(metadata["mean"])
            std = torch.tensor(metadata["std"])
            continuous = (continuous - mean) / std

        if transform == "normalize":
            min_val = torch.tensor(metadata["min"])
            max_val = torch.tensor(metadata["max"])
            continuous = (continuous - min_val) / (max_val - min_val)

        if transform == "log_pt":
            continuous[:, :, 0] = torch.log(continuous[:, :, 0] + 1e-6)
            metadata = self._extract_metadata(continuous, mask)
            mean = torch.tensor(metadata["mean"])
            std = torch.tensor(metadata["std"])
            continuous = (continuous - mean) / std

        if not self.pt_order:

            # shuffle particles within jets

            idx = torch.randperm(mask.shape[1])

            if continuous is not None:
                continuous = continuous[:, idx, :]
            if discrete is not None:
                discrete = discrete[:, idx, :]
            mask = mask[:, idx, :]

        return continuous, discrete, mask, metadata

    def _read_aoj_file(self, filepath, num_jets=None):
        """Reads and processes a single .h5 file from the AOJ dataset."""

        try:
            with h5py.File(filepath, "r") as f:
                PFCands = f["PFCands"][:num_jets] if num_jets else f["PFCands"][:]
        except (OSError, KeyError) as e:
            raise ValueError(f"Error reading file {filepath}: {e}")

        feats, mask = self._compute_continuous_coordinates(PFCands)
        feats["tokens"] = self._map_pid_to_tokens(PFCands[:, :, -2])[:, :, None]
        feats["onehot"] = np.eye(9)[feats["tokens"]].squeeze(2)
        mask = torch.tensor(mask[:, :, None], dtype=torch.long)

        for key in feats:
            feats[key] = torch.tensor(feats[key], dtype=torch.float32)

        return feats, mask

    def _download_file(self, target_file):
        """Download a file from a URL to a local path.
        """
        filename = os.path.basename(target_file)
        full_url = f"{self.url}/{filename}"  # Append filename to the base URL

        try:
            urllib.request.urlretrieve(full_url, target_file)
            log.info(f"Downloaded {target_file} successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download file from {full_url}. Error: {e}")

    def _filter_particles(self, PFCands):
        """Filter and remove bad particle candidates.
        """
        mask_bad_pids = np.abs(PFCands[:, :, -2]) < 11
        PFCands[mask_bad_pids] = np.zeros_like(PFCands[mask_bad_pids])
        return PFCands

    def _pt_order(self, PFCands):
        """Sort particles by transverse momentum (pt).
        """
        pt = np.sqrt(PFCands[:, :, 0]**2 + PFCands[:, :, 1]**2)
        idx = np.argsort(pt, axis=1)
        PFCands = np.array([PFCands[j][i][::-1] for j, i in enumerate(idx)])
        return PFCands  

    def _map_pid_to_tokens(self, pid):
        """Map particle IDs to predefined values."""

        pid_map = {
            22: 1,  # Photon
            130: 2,  # Neutral Hadron
            -211: 3,  # Charged Hadron
            211: 4,  # Charged Hadron
            -11: 5,  # Electron
            11: 6,  # Positron
            -13: 7,  # Muon
            13: 8,  # Antimuon
        }
        pid = np.vectorize(lambda x: pid_map.get(x, 0))(pid)
        return pid

    def _compute_continuous_coordinates(self, PFCands):
        """Compute relative kinematic and spatial coordinates."""

        PFCands = self._filter_particles(PFCands)
        PFCands = self._pt_order(PFCands)

        px, py, pz, e = (
            PFCands[:, :, 0],
            PFCands[:, :, 1],
            PFCands[:, :, 2],
            PFCands[:, :, 3],
        )
        pt = np.sqrt(px**2 + py**2) 
        eta = np.arcsinh(np.divide(pz, pt, out=np.zeros_like(pz), where=pt != 0))
        phi = np.arctan2(py, px)

        jet = PFCands[:, :, :4].sum(axis=1)
        jet_eta = np.arcsinh(jet[:, 2] / np.sqrt(jet[:, 0] ** 2 + jet[:, 1] ** 2))
        jet_phi = np.arctan2(jet[:, 1], jet[:, 0])

        eta_rel = eta - jet_eta[:, None]
        phi_rel = (phi - jet_phi[:, None] + np.pi) % (2 * np.pi) - np.pi

        mask = PFCands[:, :, 3] > 0

        if self.padding=='ghosts':
            # add soft 'ghost' particles to zero padded entries:
            pt_min = pt[pt>0].min(0)
            eta_min = eta_rel[pt>0].min(0)
            eta_max = eta_rel[pt>0].max(0)
            phi_min = phi_rel[pt>0].min(0)
            phi_max = phi_rel[pt>0].max(0)

            pt_ghost = np.random.uniform(0, pt_min, size=mask.shape)
            eta_ghost = np.random.uniform(eta_min, eta_max, size=mask.shape)
            phi_ghost = np.random.uniform(phi_min, phi_max, size=mask.shape)

            pt = np.where(mask, pt, pt_ghost)
            eta_rel = np.where(mask, eta_rel, eta_ghost)
            phi_rel = np.where(mask, phi_rel, phi_ghost)
        
            mask = pt > 0

        feats = {}
        feats["px"] = (px * mask)[:, :, None]
        feats["py"] = (py * mask)[:, :, None]
        feats["pz"] = (pz * mask)[:, :, None]
        feats["e"] = (e * mask)[:, :, None]
        feats["pt"] = (pt * mask)[:, :, None]
        feats["eta"] = (eta * mask)[:, :, None]
        feats["phi"] = (phi * mask)[:, :, None]
        feats["eta_rel"] = (eta_rel * mask)[:, :, None]
        feats["phi_rel"] = (phi_rel * mask)[:, :, None]

        d0 = PFCands[:, :, 4]
        d0Err = PFCands[:, :, 5]
        dz = PFCands[:, :, 6]
        dzErr = PFCands[:, :, 7]

        feats["d0"] = (d0 * mask)[:, :, None]
        feats["dz"] = (dz * mask)[:, :, None]
        feats["d0Err"] = (d0Err * mask)[:, :, None]
        feats["dzErr"] = (dzErr * mask)[:, :, None]

        return feats, mask

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        log.info(f"Loading metadata from {metadata_file}.")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata

    def _extract_metadata(self, continuous, mask):

        mask_bool = mask.squeeze(-1) > 0
        nums = mask.sum(dim=1).squeeze()
        hist, _ = np.histogram(
            nums, bins=np.arange(0, mask.shape[1] + 2, 1), density=True
        )

        metadata =  {
            "num_jets_sample": mask.shape[0],
            "num_particles_sample": nums.sum().item(),
            "max_num_particles_per_jet": mask.shape[1],
            }
        
        if continuous is not None:
            metadata["mean"] = continuous[mask_bool].mean(0).tolist()
            metadata["std"] = continuous[mask_bool].std(0).tolist()
            metadata["min"] = continuous[mask_bool].min(0).values.tolist()
            metadata["max"] = continuous[mask_bool].max(0).values.tolist()

        metadata["categorical_probs"] = hist.tolist()
        return metadata


class AOJDataModule(L.LightningDataModule):
    """DataModule for handling source-target-context coupling for particle cloud data."""

    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.config = config
        self.transform = config.data.transform
        self.features = {
            "continuous": config.data.continuous_features,
            "discrete": config.data.discrete_features,
        }
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.pin_memory = config.data.pin_memory
        self.split_ratios = config.data.split_ratios

        self.num_jets = config.data.num_jets
        self.max_num_particles = config.data.max_num_particles
        self.metadata = {}

        self.aoj_train = AspenOpenJets(
            data_dir=config.data.target_path,
            data_files=config.data.target_train_files,
        )
        self.aoj_predict = AspenOpenJets(
            data_dir=config.data.target_path,
            data_files=config.data.target_test_files,
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
        target, metadata = self.aoj_train(
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
        target, _ = self.aoj_predict(
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
