import numpy as np
import torch
import h5py
import os
import urllib.request
import json
from torch.utils.data import DataLoader, Subset
import lightning.pytorch as L

from utils.helpers import SimpleLogger as log
from utils.tensorclass import TensorMultiModal


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
