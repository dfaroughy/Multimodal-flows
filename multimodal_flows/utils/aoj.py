import torch
import numpy as np
import awkward as ak
import vector
import fastjet
import scipy
import h5py
import os
import urllib.request
import json
from torch.utils.data import DataLoader, Subset
import lightning.pytorch as L
from torch.nn import functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataclasses import dataclass

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.autolayout"] = False
vector.register_awkward()

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
        """Filter and remove bad particle candidates. e.g. pdg_id=2
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
            metadata["min"] = continuous[mask_bool].min(0)[0].tolist()
            metadata["max"] = continuous[mask_bool].max(0)[0].tolist()
            metadata["log_pt_mean"] = [torch.log(continuous[...,0])[mask_bool].mean(0).item()] + continuous[mask_bool].mean(0)[1:].tolist()
            metadata["log_pt_std"] = [torch.log(continuous[...,0])[mask_bool].std(0).item()] + continuous[mask_bool].std(0)[1:].tolist()

        # metadata["categorical_probs"] = hist.tolist()
        return metadata


@dataclass
class ParticleClouds:
    """
    A dataclass to hold particle cloud low-level features
    """

    data: TensorMultiModal = None

    def __post_init__(self):

        self.continuous = self.data.continuous  # (B, D, dim_continuous=3)
        self.discrete = self.data.discrete      # (B, D)
        self.mask = self.data.mask              # (B, D, 1) 
        self.mask_bool = self.mask.squeeze(-1) > 0
        self.multiplicity = torch.sum(self.mask, dim=1)

        if self.data.has_continuous:
            self.pt = self.continuous[..., 0]
            self.eta_rel = self.continuous[..., 1]
            self.phi_rel = self.continuous[..., 2]
            self.px = self.pt * torch.cos(self.phi_rel)
            self.py = self.pt * torch.sin(self.phi_rel)
            self.pz = self.pt * torch.sinh(self.eta_rel)
            self.E = self.pt * torch.cosh(self.eta_rel)
            
        if self.data.has_discrete:
            self._flavored_kinematics("Photon", selection=self.discrete == 1)
            self._flavored_kinematics("NeutralHadron", selection=self.discrete == 2)
            self._flavored_kinematics("NegativeHadron", selection=self.discrete == 3)
            self._flavored_kinematics("PositiveHadron", selection=self.discrete == 4)
            self._flavored_kinematics("Electron", selection=self.discrete == 5)
            self._flavored_kinematics("Positron", selection=self.discrete == 6)
            self._flavored_kinematics("Muon", selection=self.discrete == 7)
            self._flavored_kinematics("AntiMuon", selection=self.discrete == 8)
            self._flavored_kinematics("Hadron", selection=(self.discrete >= 2) & (self.discrete <= 4))
            self._flavored_kinematics("Lepton", selection=(self.discrete > 4))
            self._flavored_kinematics("Neutral", selection=(self.discrete <= 2))
            self._flavored_kinematics("Charged", selection=(self.discrete > 2))
            self._flavored_kinematics("Negative", selection=(self.discrete == 3) | (self.discrete == 5) | (self.discrete == 7),)
            self._flavored_kinematics("Positive", selection=(self.discrete == 4) | (self.discrete == 6) | (self.discrete == 8),)

            # get particle charges:

            self.charge = torch.zeros_like(self.pt)
            self.charge[self.isPositive] = 1
            self.charge[self.isNegative] = -1

    def _flavored_kinematics(self, name, selection):
        if self.data.has_discrete:
            setattr(self, f"is{name}", selection * self.mask_bool)
            setattr(self, f"num_{name}", torch.sum(getattr(self, f"is{name}"), dim=1))

        if self.data.has_continuous:
            setattr(self, f"pt_{name}", self.pt[getattr(self, f"is{name}")])
            setattr(self, f"eta_{name}", self.eta_rel[getattr(self, f"is{name}")])
            setattr(self, f"phi_{name}", self.phi_rel[getattr(self, f"is{name}")])

    def __len__(self):
        return len(self.data)


    @property
    def has_continuous(self):
        if self.data.has_continuous:
            return True
        return False

    @property
    def has_discrete(self):
        if self.data.has_discrete:
            return True
        return False

    @property
    def has_hybrid(self):
        if self.data.has_discrete and self.data.has_continuous:
            return True
        return False

    def histplot(
            self,
            feature,
            apply_map="mask_bool",
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None,
            figsize=(3, 3),
            fontsize=10,
            ax=None,
            **kwargs,
        ):
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)

            x = getattr(self, feature)

            if "num" in feature:
                apply_map = None

            if apply_map == "mask_bool":
                x = x[self.mask_bool]

            elif apply_map is not None:
                x = apply_map(x)

            if isinstance(x, torch.Tensor):
                x.detach().cpu().numpy()

            sns.histplot(x, element="step", ax=ax, **kwargs)
            ax.set_xlabel(
                "particle-level " + feature if xlabel is None else xlabel,
                fontsize=fontsize,
            )
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)


@dataclass
class JetFeatures:
    """
    A dataclass to hold jet features and substructure.
    """

    data: TensorMultiModal = None

    def __post_init__(self):
        self.constituents = ParticleClouds(self.data)
        self.numParticles = torch.sum(self.constituents.mask, dim=1)

        if self.constituents.has_continuous:
            self.px = self.constituents.px.sum(axis=-1)
            self.py = self.constituents.py.sum(axis=-1)
            self.pz = self.constituents.pz.sum(axis=-1)
            self.E = self.constituents.E.sum(axis=-1)
            self.pt = torch.sqrt(self.px**2 + self.py**2)
            self.m = torch.sqrt(self.E**2 - self.pt**2 - self.pz**2)
            self.eta = 0.5 * torch.log((self.pt + self.pz) / (self.pt - self.pz))
            self.phi = torch.atan2(self.py, self.px)

            self._substructure(R=0.8, beta=1.0, use_wta_scheme=True)

        if self.constituents.has_discrete:
            self.charge = self._jet_charge(kappa=0.0)

        if self.constituents.has_continuous and self.constituents.has_discrete:
            self.jet_charge = self._jet_charge(kappa=1.0)

    def histplot(
        self,
        features="pt",
        apply_map=None,
        xlim=None,
        ylim=None,
        xlabel=None,
        ylabel=None,
        figsize=(3, 3),
        fontsize=10,
        ax=None,
        **kwargs,
    ):
        x = getattr(self, features)

        if apply_map is not None:
            x = apply_map(x)

        if isinstance(x, torch.Tensor):
            x.cpu().numpy()

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        sns.histplot(x=x, element="step", ax=ax, **kwargs)
        ax.set_xlabel(
            "jet-level " + features if xlabel is None else xlabel,
            fontsize=fontsize,
        )
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)




    # metrics:

    def Wassertein1D(self, feature, reference):
        x = getattr(self, feature)
        y = getattr(reference, feature)
        return scipy.stats.wasserstein_distance(x, y)

    # helper methods:


    def _jet_charge(self, kappa):
        """jet charge defined as Q_j^kappa = Sum_i Q_i * (pT_i / pT_jet)^kappa"""
        # charge = map_tokens_to_basis(self.constituents.discrete)[..., -1]
        if kappa > 0:
            jet_charge = self.constituents.charge * (self.constituents.pt) ** kappa
            return jet_charge.sum(axis=1) / (self.pt**kappa)
        else:
            return self.constituents.charge.sum(axis=1)

    def _get_flavor_counts(self, return_fracs=False, vocab_size=9):
        num_jets = len(self.constituents)
        tokens = self.constituents.discrete
        mask = self.constituents.mask_bool

        flat_indices = torch.arange(num_jets, device=tokens.device).unsqueeze(1) * (vocab_size + 1) + tokens * mask
        flat_indices = flat_indices[mask]  # Use the mask to remove invalid (padded) values

        token_counts = torch.bincount(flat_indices, minlength=num_jets * (vocab_size + 1))
        count = token_counts.view(num_jets, vocab_size + 1)  # Reshape to (B, n + 1)
        return count


    def _substructure(self, R=0.8, beta=1.0, use_wta_scheme=True):
        constituents_ak = ak.zip(
            {
                "pt": np.array(self.constituents.pt),
                "eta": np.array(self.constituents.eta_rel),
                "phi": np.array(self.constituents.phi_rel),
                "mass": np.zeros_like(np.array(self.constituents.pt)),
            },
            with_name="Momentum4D",
        )

        constituents_ak = ak.mask(constituents_ak, constituents_ak.pt > 0)
        constituents_ak = ak.drop_none(constituents_ak)

        self._constituents_ak = constituents_ak[ak.num(constituents_ak) >= 3]

        if use_wta_scheme:
            jetdef = fastjet.JetDefinition(
                fastjet.kt_algorithm, R, fastjet.WTA_pt_scheme
            )
        else:
            jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, R)

        print("Clustering jets with fastjet")
        print("Jet definition:", jetdef)
        print("Calculating N-subjettiness")

        self._cluster = fastjet.ClusterSequence(self._constituents_ak, jetdef)
        self.d0 = self._calc_d0(R, beta)
        self.c1 = self._cluster.exclusive_jets_energy_correlator(njets=1, func="c1")
        self.d2 = self._cluster.exclusive_jets_energy_correlator(njets=1, func="d2")
        self.tau1 = self._calc_tau1(beta)
        self.tau2 = self._calc_tau2(beta)
        self.tau3 = self._calc_tau3(beta)
        self.tau21 = np.ma.divide(self.tau2, self.tau1)
        self.tau32 = np.ma.divide(self.tau3, self.tau2)

    def _calc_deltaR(self, particles, jet):
        jet = ak.unflatten(ak.flatten(jet), counts=1)
        return particles.deltaR(jet)

    def _calc_d0(self, R, beta):
        """Calculate the d0 values."""
        return ak.sum(self._constituents_ak.pt * R**beta, axis=1)

    def _calc_tau1(self, beta):
        """Calculate the tau1 values."""
        excl_jets_1 = self._cluster.exclusive_jets(n_jets=1)
        delta_r_1i = self._calc_deltaR(self._constituents_ak, excl_jets_1[:, :1])
        pt_i = self._constituents_ak.pt
        return ak.sum(pt_i * delta_r_1i**beta, axis=1) / self.d0

    def _calc_tau2(self, beta):
        """Calculate the tau2 values."""
        excl_jets_2 = self._cluster.exclusive_jets(n_jets=2)
        delta_r_1i = self._calc_deltaR(self._constituents_ak, excl_jets_2[:, :1])
        delta_r_2i = self._calc_deltaR(self._constituents_ak, excl_jets_2[:, 1:2])
        pt_i = self._constituents_ak.pt

        # add new axis to make it broadcastable
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** beta,
                    delta_r_2i[..., np.newaxis] ** beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        return ak.sum(pt_i * min_delta_r, axis=1) / self.d0

    def _calc_tau3(self, beta):
        """Calculate the tau3 values."""
        excl_jets_3 = self._cluster.exclusive_jets(n_jets=3)
        delta_r_1i = self._calc_deltaR(self._constituents_ak, excl_jets_3[:, :1])
        delta_r_2i = self._calc_deltaR(self._constituents_ak, excl_jets_3[:, 1:2])
        delta_r_3i = self._calc_deltaR(self._constituents_ak, excl_jets_3[:, 2:3])
        pt_i = self._constituents_ak.pt

        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** beta,
                    delta_r_2i[..., np.newaxis] ** beta,
                    delta_r_3i[..., np.newaxis] ** beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        return ak.sum(pt_i * min_delta_r, axis=1) / self.d0


class EnergyCorrelationFunctions:

    def __init__(self, data):
        self.data = data
        self.mask_3_parts = data.mask.sum(dim=1).squeeze(-1) >= 3 

    def get_flavor(self, token):
        flavor = self.data.clone()
        if token == 4567:
            flavor_mask = flavor.discrete >= 4
        elif token == 23:
            flavor_mask = (flavor.discrete == 2) | (flavor.discrete == 3) 
        elif token == 45:
            flavor_mask = (flavor.discrete == 4) | (flavor.discrete == 5)
        elif token == 67:
            flavor_mask = (flavor.discrete == 6) | (flavor.discrete == 7)
        elif token == 123:
            flavor_mask = (flavor.discrete >= 1) & (flavor.discrete <= 3)
        elif token == 234567:
            flavor_mask = (flavor.discrete >= 2) 
        elif token == 10:
            flavor_mask = (flavor.discrete == 0) | (flavor.discrete == 1)   
        elif token == 357:
            flavor_mask = (flavor.discrete == 3) | (flavor.discrete == 5) | (flavor.discrete == 7)
        elif token == 246:
            flavor_mask = (flavor.discrete == 2) | (flavor.discrete == 4) | (flavor.discrete == 6)
        else:
            flavor_mask = flavor.discrete == token
        
        flavor.apply_mask(flavor_mask)
        flavor.mask *= flavor_mask 
        del flavor.discrete
        return flavor
        
    def compute_ecf(self, flavor_i, flavor_j=None, beta=1.0):

        flavor = {'photon': 0, 
                  'h0': 1,
                  'h-': 2,
                  'h+': 3,
                  'e-': 4,
                  'e+': 5,
                  'mu-': 6,
                  'mu+': 7,
                  'hadron': 123,
                  'lepton': 4567,
                  'positive':357,
                  'negative':246,
                  'charged': 234567,
                  'neutral': 10,
                  'h+/-': 23,
                  'e+/-': 45,
                  'mu+/-': 67,
                  }
                  
        i = flavor[flavor_i]
        j = flavor[flavor_j] if flavor_j is not None else None

        if flavor_j is None:
            jets = self.get_flavor(i).continuous
            return self._auto_ecf(jets, beta)
        else:
            jets_i = self.get_flavor(i).continuous
            jets_j = self.get_flavor(j).continuous
            return self._cross_ecf(jets_i, jets_j, beta)

    def _auto_ecf(self, tensor, beta=1.0):

        auto_ecf = []
        jet_pT2 = []

        for jet in tensor:
            
            jet = jet[jet[:, 0] != 0]

            if len(jet) < 2:
                auto_ecf.append(0.0)
                jet_pT2.append(0.0)
                continue

            pT = jet[:, 0]
            eta = jet[:, 1]
            phi = jet[:, 2]

            pT2 = pT.sum() ** 2

            # Compute pairwise differences
            delta_eta = eta.unsqueeze(1) - eta.unsqueeze(0)
            delta_phi = phi.unsqueeze(1) - phi.unsqueeze(0)
            delta_phi = torch.remainder(delta_phi + torch.pi, 2 * torch.pi) - torch.pi

            # Compute pairwise distances
            R_ij = torch.sqrt(delta_eta**2 + delta_phi**2) ** beta

            ecf_matrix = (pT.unsqueeze(1) * pT.unsqueeze(0)) * R_ij / 2.0
            ecf = ecf_matrix.sum() / pT2

            auto_ecf.append(ecf.item())
            jet_pT2.append(pT2.item())

        auto_ecf = torch.tensor(auto_ecf)[self.mask_3_parts]
        jet_pT2 = torch.tensor(jet_pT2)[self.mask_3_parts]

        return auto_ecf, jet_pT2


    def _cross_ecf(self, tensor_1, tensor_2, beta=1.0):

        cross_ecf = []
        jet_pT2 = []

        for idx, jet in enumerate(tensor_1):

            j0 = jet[jet[:, 0] != 0]
            j1 = tensor_2[idx][tensor_2[idx][:, 0] != 0]

            if len(jet) == 0  or len(jet) == 0:
                cross_ecf.append(0.0)
                jet_pT2.append(0.0)
                continue

            pT_0, eta_0, phi_0 = j0[:, 0], j0[:, 1], j0[:, 2]
            pT_1, eta_1, phi_1 = j1[:, 0], j1[:, 1], j1[:, 2]

            pT2 = pT_0.sum() * pT_1.sum()

            delta_eta = eta_0.unsqueeze(1) - eta_1.unsqueeze(0)
            delta_phi = phi_0.unsqueeze(1) - phi_1.unsqueeze(0)
            delta_phi = torch.remainder(delta_phi + np.pi, 2 * np.pi) - np.pi

            R_ij = torch.sqrt(delta_eta**2 + delta_phi**2) ** beta

            ecf_matrix = (pT_0.unsqueeze(1) * pT_1.unsqueeze(0)) * R_ij
            ecf = ecf_matrix.sum() / pT2

            cross_ecf.append(ecf.item())
            jet_pT2.append(pT2.item())

        cross_ecf = torch.tensor(cross_ecf)[self.mask_3_parts]
        jet_pT2 = torch.tensor(jet_pT2)[self.mask_3_parts]

        return cross_ecf, jet_pT2


class JetChargeDipole:

    """
    Compute pT-weighted jet charge  Q_kappa  and
    the 2-point electric-dipole moment  d2  for every jet.
    """

    def __init__(self, data):

        """
        data: an object with attributes
              .continuous  (pT, eta, phi) padded with zeros
              .charge      integer charges (−1, 0, +1)
              .mask        boolean mask of real particles
        """
        self.x = data.constituents.continuous      # (n, D, 3)
        self.Q = data.constituents.charge          # (n, D)
        self.mask = data.constituents.mask_bool    # (n, D)

        # option: keep only jets that have ≥2 (for d2) or ≥1 (for Q) particles

        n_part = self.mask.sum(dim=1)
        self.valid_Q  = n_part >= 1
        self.valid_d2 = n_part >= 2

    def _delta_R(self, eta, phi):
        d_eta = eta.unsqueeze(1) - eta.unsqueeze(0)
        d_phi = torch.remainder(phi.unsqueeze(1) - phi.unsqueeze(0) + np.pi,
                                2 * np.pi) - np.pi
        return torch.sqrt(d_eta**2 + d_phi**2)

    def charge_and_dipole(self, kappa: float = 1.0, beta: float = 1.0):
        """
        Compute the pT-weighted jet charge  Q_kappa  and the electric–dipole
        moment  d2  for every jet in the batch.

        Returns
        -------
        Q_kappa : 1-D tensor, length = n_valid_jets
        d2      : 1-D tensor, length = n_valid_jets
                (jets with <2 particles get filtered out, like _auto_ecf)
        """

        Q0_list, Qkappa_list, d2_list = [],[],[]     # results for *all* jets
        mask_2_parts = (self.mask.sum(dim=1) >= 2)   # ≥2 real particles

        for idx, jet in enumerate(self.x):        # iterate over jets   (D,3) view

            pT, eta, phi = jet[:, 0], jet[:, 1], jet[:, 2]
            mask = pT > 0
            Q = self.Q[idx][mask].float() 
            pT = pT[mask]
            eta = eta[mask]
            phi = phi[mask]

            # -------------------------------------------------
            #   Jet charge   Q_kappa
            # -------------------------------------------------

            jet_pT = pT.sum()
            
            if jet_pT == 0:
                Qkappa = torch.nan
                Q0 = torch.nan
            else:
                Qkappa = (Q * pT**kappa).sum() / jet_pT
                Q0 = Q.sum() 

            # -------------------------------------------------
            #   Electric-dipole   d2
            # -------------------------------------------------

            if len(jet) < 2:
                d2 = torch.nan
            else:
                # pair-wise ΔR
                d_eta = eta.unsqueeze(1) - eta.unsqueeze(0)
                d_phi = torch.remainder(phi.unsqueeze(1) - phi.unsqueeze(0) + torch.pi,
                                        2 * torch.pi) - torch.pi
                R_ij  = torch.sqrt(d_eta**2 + d_phi**2).pow(beta)   # (N,N)

                weight   = (Q * pT).unsqueeze(1) * (Q * pT).unsqueeze(0)
                dip_mat  = weight * R_ij / 2.0          # divide-by-2 like _auto_ecf
                d2       = dip_mat.sum() / jet_pT**2

            Q0_list.append(Q0)
            Qkappa_list.append(Qkappa)
            d2_list.append(d2)

        # tensor-ise and filter exactly like _auto_ecf
        Q0 = torch.tensor(Q0_list)
        Qkappa  = torch.tensor(Qkappa_list)
        d2 = torch.tensor(d2_list)

        Q0  = Q0[mask_2_parts]
        Qkappa = Qkappa[mask_2_parts]
        d2 = d2[mask_2_parts]

        return Q0, Qkappa, d2
