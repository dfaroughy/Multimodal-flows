import torch
import numpy as np
import awkward as ak
import vector
from torch.nn import functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataclasses import dataclass
import fastjet
import scipy

from tensorclass import TensorMultiModal

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.autolayout"] = False
vector.register_awkward()


def jet_set_to_seq(part_set: TensorMultiModal, vocab_size: int):
    """Convert a particle set to a sequence with start, end and pad tokens.
    
    Args:
        particle_seq (TensorMultiModal): The particle set.
        vocab_size (int): The size of the vocabulary, used to determine the token values.
        
    Returns:
        np.ndarray: The sequence with start, end and pad tokens.
                    start_token = vocab_size + 1
                    end_token = vocab_size + 2
                    pad_token = vocab_size + 3
    """

    particle_set = part_set.clone()
    start_token = vocab_size + 1
    end_token = vocab_size + 2
    pad_token = vocab_size + 3
    
    if not hasattr(particle_set, 'discrete'):
        raise ValueError("The particle_seq must have a 'discrete' attribute.")

    seq = particle_set.discrete.squeeze(-1).numpy()  # (N, D)
    N, _ = seq.shape

    start = np.full((N, 1), start_token, dtype=int)
    extra_pad = np.full((N, 1), pad_token, dtype=int)
    seq[seq==0] = pad_token 
    seq = np.concatenate([start, seq, extra_pad], axis=1)

    idx_eos = (seq != pad_token).sum(axis=1)

    for i, jet in enumerate(seq):
        jet[idx_eos[i]] = end_token
    
    particle_set.discrete = torch.tensor(seq).long()
    particle_set.mask = (particle_set.discrete != pad_token).long()

    return particle_set




@dataclass
class ParticleClouds:
    """
    A dataclass to hold particle clouds data.
    """

    data: TensorMultiModal = None

    def __post_init__(self):
        self.continuous = self.data.continuous
        self.discrete = self.data.discrete
        self.mask = self.data.mask
        self.mask_bool = self.mask.squeeze(-1) > 0
        self.multiplicity = torch.sum(self.mask, dim=1)

        if self.data.has_continuous:
            self.pt = self.continuous[..., 0]
            self.eta_rel = self.continuous[..., 1]
            self.phi_rel = self.continuous[..., 2]
            self.px = self.pt * torch.cos(self.phi_rel)
            self.py = self.pt * torch.sin(self.phi_rel)
            self.pz = self.pt * torch.sinh(self.eta_rel)
            self.e = self.pt * torch.cosh(self.eta_rel)
            self.has_displaced_vertex = self.continuous.shape[-1] > 3

            if self.has_displaced_vertex:
                self.d0 = self.continuous[..., 3]
                self.d0Err = self.continuous[..., 4]
                self.dz = self.continuous[..., 5]
                self.dzErr = self.continuous[..., 6]
                self.d0_ratio = np.divide(
                    self.d0,
                    self.d0Err,
                    out=np.zeros_like(self.d0),
                    where=self.d0Err != 0,
                )
                self.dz_ratio = np.divide(
                    self.dz,
                    self.dzErr,
                    out=np.zeros_like(self.dz),
                    where=self.dzErr != 0,
                )

        if self.data.has_discrete:
            self._flavored_kinematics(
                "Photon", selection=self.discrete.squeeze(-1) == 1
            )
            self._flavored_kinematics(
                "NeutralHadron", selection=self.discrete.squeeze(-1) == 2
            )
            self._flavored_kinematics(
                "NegativeHadron", selection=self.discrete.squeeze(-1) == 3
            )
            self._flavored_kinematics(
                "PositiveHadron", selection=self.discrete.squeeze(-1) == 4
            )
            self._flavored_kinematics(
                "Electron", selection=self.discrete.squeeze(-1) == 5
            )
            self._flavored_kinematics(
                "Positron", selection=self.discrete.squeeze(-1) == 6
            )
            self._flavored_kinematics("Muon", selection=self.discrete.squeeze(-1) == 7)
            self._flavored_kinematics(
                "AntiMuon", selection=self.discrete.squeeze(-1) == 8
            )
            self._flavored_kinematics(
                "ChargedHadron",
                selection=(self.discrete.squeeze(-1) == 3)
                | (self.discrete.squeeze(-1) == 4),
            )
            self._flavored_kinematics(
                "Electrons",
                selection=(self.discrete.squeeze(-1) == 5)
                | (self.discrete.squeeze(-1) == 6),
            )
            self._flavored_kinematics(
                "Muons",
                selection=(self.discrete.squeeze(-1) == 7)
                | (self.discrete.squeeze(-1) == 8),
            )
            self._flavored_kinematics(
                "Lepton", selection=(self.discrete.squeeze(-1) > 4)
            )
            self._flavored_kinematics(
                "Neutral", selection=(self.discrete.squeeze(-1) == 1 ) |
                (self.discrete.squeeze(-1) == 2)
            )
            self._flavored_kinematics(
                "Charged", selection=(self.discrete.squeeze(-1) > 2)
            )
            self._flavored_kinematics(
                "Negative",
                selection=(self.discrete.squeeze(-1) == 3)
                | (self.discrete.squeeze(-1) == 5)
                | (self.discrete.squeeze(-1) == 7),
            )
            self._flavored_kinematics(
                "Positive",
                selection=(self.discrete.squeeze(-1) == 4)
                | (self.discrete.squeeze(-1) == 6)
                | (self.discrete.squeeze(-1) == 8),
            )

            # get particle charges:

            self.charge = torch.zeros_like(self.mask)
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

            if self.has_displaced_vertex:
                setattr(self, f"d0_{name}", self.d0[getattr(self, f"is{name}")])
                setattr(self, f"d0Err_{name}", self.d0Err[getattr(self, f"is{name}")])
                setattr(self, f"dz_{name}", self.dz[getattr(self, f"is{name}")])
                setattr(self, f"dzErr_{name}", self.dzErr[getattr(self, f"is{name}")])
                setattr(
                    self, f"d0_ratio_{name}", self.d0_ratio[getattr(self, f"is{name}")]
                )
                setattr(
                    self, f"dz_ratio_{name}", self.dz_ratio[getattr(self, f"is{name}")]
                )

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

    def display_cloud(
        jets,
        idx,
        scale_marker=1.0,
        ax=None,
        figsize=(3.2, 3),
        facecolor="whitesmoke",
        color={
            "e": "firebrick",
            "mu": "hotpink",
            "h": "blue",
            "h0": "forestgreen",
            "a": "gold",
        },
        title=None,
        title_box_anchor=(1.025, 1.125),
        savefig=None,
        legend=False,
        display_N=False,
        xlim=None,
        ylim=None,
        alpha=0.5,
        xticks=[],
        yticks=[],
        edgecolor=None,
        lw=0.75,
    ):
        eta = jets.eta_rel[idx]
        phi = jets.phi_rel[idx]
        pt = jets.pt[idx] * scale_marker
        N = jets.mask[idx].squeeze(-1).sum().item()

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            eta[jets.isPhoton[idx]],
            phi[jets.isPhoton[idx]],
            marker="o",
            s=pt[jets.isPhoton[idx]],
            color=color["a"],
            alpha=alpha,
            label=r"$\gamma$",
            edgecolors=edgecolor,
            linewidth=lw,
        )
        ax.scatter(
            eta[jets.isNeutralHadron[idx]],
            phi[jets.isNeutralHadron[idx]],
            marker="o",
            s=pt[jets.isNeutralHadron[idx]],
            color=color["h0"],
            alpha=alpha,
            label=r"$h^{0}$",
            edgecolors=edgecolor,
            linewidth=lw,
        )
        ax.scatter(
            eta[jets.isNegativeHadron[idx]],
            phi[jets.isNegativeHadron[idx]],
            marker="^",
            s=pt[jets.isNegativeHadron[idx]],
            color=color["h"],
            alpha=alpha,
            label=r"$h^{-}$",
            edgecolors=edgecolor,
            linewidth=lw,
        )
        ax.scatter(
            eta[jets.isPositiveHadron[idx]],
            phi[jets.isPositiveHadron[idx]],
            marker="v",
            s=pt[jets.isPositiveHadron[idx]],
            color=color["h"],
            alpha=alpha,
            label=r"$h^{+}$",
            edgecolors=edgecolor,
            linewidth=lw,
        )
        ax.scatter(
            eta[jets.isElectron[idx]],
            phi[jets.isElectron[idx]],
            marker="^",
            s=pt[jets.isElectron[idx]],
            color=color["e"],
            alpha=alpha,
            label=r"$e^{-}$",
            edgecolors=edgecolor,
            linewidth=lw,
        )
        ax.scatter(
            eta[jets.isPositron[idx]],
            phi[jets.isPositron[idx]],
            marker="v",
            s=pt[jets.isPositron[idx]],
            color=color["e"],
            alpha=alpha,
            label=r"$e^{+}$",
            edgecolors=edgecolor,
            linewidth=lw,
        )
        ax.scatter(
            eta[jets.isMuon[idx]],
            phi[jets.isMuon[idx]],
            marker="^",
            s=pt[jets.isMuon[idx]],
            color=color["mu"],
            alpha=alpha,
            label=r"$\mu^{-}$",
            edgecolors=edgecolor,
            linewidth=lw,
        )
        ax.scatter(
            eta[jets.isAntiMuon[idx]],
            phi[jets.isAntiMuon[idx]],
            marker="v",
            s=pt[jets.isAntiMuon[idx]],
            color=color["mu"],
            alpha=alpha,
            label=r"$\mu^{+}$",
            edgecolors=edgecolor,
            linewidth=lw,
        )

        # Define custom legend markers
        h1 = Line2D(
            [0],
            [0],
            marker="o",
            markersize=6,
            alpha=0.5,
            color="gold",
            linestyle="None",
        )
        h2 = Line2D(
            [0],
            [0],
            marker="o",
            markersize=6,
            alpha=0.5,
            color="forestgreen",
            linestyle="None",
        )
        h3 = Line2D(
            [0],
            [0],
            marker="^",
            markersize=6,
            alpha=0.5,
            color="blue",
            linestyle="None",
        )
        h4 = Line2D(
            [0],
            [0],
            marker="v",
            markersize=6,
            alpha=0.5,
            color="blue",
            linestyle="None",
        )
        h5 = Line2D(
            [0],
            [0],
            marker="^",
            markersize=6,
            alpha=0.5,
            color="firebrick",
            linestyle="None",
        )
        h6 = Line2D(
            [0],
            [0],
            marker="v",
            markersize=6,
            alpha=0.5,
            color="firebrick",
            linestyle="None",
        )
        h7 = Line2D(
            [0],
            [0],
            marker="^",
            markersize=6,
            alpha=0.5,
            color="hotpink",
            linestyle="None",
        )
        h8 = Line2D(
            [0],
            [0],
            marker="v",
            markersize=6,
            alpha=0.5,
            color="hotpink",
            linestyle="None",
        )

        if legend:
            ax.legend(
                [h1, h2, h3, h4, h5, h6, h7, h8],
                [
                    r"$\gamma$",
                    r"$h^0$",
                    r"$h^-$",
                    r"$h^+$",
                    r"$e^-$",
                    r"$e^+$",
                    r"$\mu^{-}$",
                    r"$\mu^{+}$",
                ],
                loc="center",
                bbox_to_anchor=(0.5, -0.25),
                markerscale=2,
                scatterpoints=1,
                fontsize=14,
                frameon=False,
                ncols=8,
                handletextpad=0.5,
                columnspacing=1.0,
            )

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=14)
        if display_N:
            ax.text(
                0.975,
                0.975,
                rf"$N={N}$",
                fontsize=14,
                ha="right",
                va="top",
                transform=ax.transAxes,
            )
        ax.set_facecolor(facecolor)  # Set the axis background color


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
            self.e = self.constituents.e.sum(axis=-1)
            self.pt = torch.sqrt(self.px**2 + self.py**2)
            self.m = torch.sqrt(self.e**2 - self.pt**2 - self.pz**2)
            self.eta = 0.5 * torch.log((self.pt + self.pz) / (self.pt - self.pz))
            self.phi = torch.atan2(self.py, self.px)

            self._substructure(R=0.8, beta=1.0, use_wta_scheme=True)

        if self.constituents.has_discrete:
            counts = self._get_flavor_counts()
            self.numPhotons = counts[..., 1]
            self.numNeutralHadrons = counts[..., 2]
            self.numNegativeHadrons = counts[..., 3]
            self.numPositiveHadrons = counts[..., 4]
            self.numElectrons = counts[..., 5]
            self.numPositrons = counts[..., 6]
            self.numMuons = counts[..., 7]
            self.numAntiMuons = counts[..., 8]
            self.numChargedHadrons = self.numPositiveHadrons + self.numNegativeHadrons
            self.numHadrons = self.numNeutralHadrons + self.numChargedHadrons
            self.numLeptons = (
                self.numElectrons
                + self.numPositrons
                + self.numMuons
                + self.numAntiMuons
            )
            self.numNeutrals = self.numPhotons + self.numNeutralHadrons
            self.numCharged = self.numChargedHadrons + self.numLeptons

            self.charge = self._jet_charge(kappa=0.0)

        if self.constituents.has_continuous and self.constituents.has_discrete:
            self.jet_charge = self._jet_charge(kappa=1.0)

    # plotting methods:

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

    def KLmetric1D(self, feature, reference, num_bins=100, use_quantiles=True):
        h1 = (
            self._histogram(
                feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles
            )
            + 1e-8
        )
        h2 = (
            reference._histogram(
                feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles
            )
            + 1e-8
        )
        return scipy.stats.entropy(h1, h2)

    def Wassertein1D(self, feature, reference):
        x = getattr(self, feature)
        y = getattr(reference, feature)
        return scipy.stats.wasserstein_distance(x, y)

    # helper methods:

    def _histogram(
        self, features="pt", density=True, num_bins=100, use_quantiles=False
    ):
        x = getattr(self, features)
        bins = (
            np.quantile(x, np.linspace(0.001, 0.999, num_bins))
            if use_quantiles
            else num_bins
        )
        return np.histogram(x, density=density, bins=bins)[0]

    def _jet_charge(self, kappa):
        """jet charge defined as Q_j^kappa = Sum_i Q_i * (pT_i / pT_jet)^kappa"""
        # charge = map_tokens_to_basis(self.constituents.discrete)[..., -1]
        if kappa > 0:
            jet_charge = self.constituents.charge * (self.constituents.pt) ** kappa
            # jet_charge = charge * (self.constituents.pt) ** kappa

            return jet_charge.sum(axis=1) / (self.pt**kappa)
        else:
            return self.constituents.charge.sum(axis=1)

    def _get_flavor_counts(self, return_fracs=False, vocab_size=8):
        num_jets = len(self.constituents)
        tokens = self.constituents.discrete.squeeze(-1)
        mask = self.constituents.mask_bool

        flat_indices = (
            torch.arange(num_jets, device=tokens.device).unsqueeze(1) * (vocab_size + 1)
        ) + tokens * mask
        flat_indices = flat_indices[
            mask
        ]  # Use the mask to remove invalid (padded) values

        token_counts = torch.bincount(
            flat_indices, minlength=num_jets * (vocab_size + 1)
        )
        count = token_counts.view(num_jets, vocab_size + 1)  # Reshape to (B, n + 1)
        return count

    def plot_flavor_count_per_jet(
        self,
        marker=".",
        color="darkred",
        markersize=2,
        ax=None,
        figsize=(3, 2),
        label=None,
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        count = self._get_flavor_counts(return_fracs=False).float()
        mean = count.mean(dim=0).cpu().numpy().tolist()
        std = count.std(dim=0).cpu().numpy().tolist()
        print(count)

        if isinstance(color, str):
            color = [color] * len(mean)

        labels = [
            r"$\gamma$",
            r"$h^0$",
            r"$h^-$",
            r"$h^+$",
            r"$e^{-}$",
            r"$e^{+}$",
            r"$\mu^{-}$",
            r"$\mu^{+}$",
        ]
        binwidth = 1
        x_positions = np.arange(len(labels)) + 1

        for i, (mu, sig) in enumerate(zip(mean, std)):
            ax.add_patch(
                plt.Rectangle(
                    (i - binwidth / 2, mu - sig),
                    binwidth,
                    2 * sig,
                    color=color[i],
                    alpha=0.15,
                    linewidth=0.0,
                )
            )
            ax.plot(i, mu, marker, color=color[i], markersize=markersize, label=label)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 5000)
        ax.set_ylabel(r"$\langle N \rangle \pm 1\sigma$", fontsize=12)
        ax.set_xlabel(r"particle flavor", fontsize=12)

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


def map_basis_to_tokens(tensor):
    """
    works for jwetclass & aoj datastes. Maps a tensor with shape (N, M, D)
    to a space of 8 tokens based on particle type and charge.

    Args:
        tensor (torch.Tensor): Input tensor of shape (N, M, D) where D=6.

    Returns:
        torch.Tensor: A tensor of shape (N, M) containing the token mappings.
    """

    if tensor.shape[-1] != 6:
        raise ValueError("The last dimension of the input tensor must be 6.")
    one_hot = tensor[..., :-1]  # Shape: (N, M, 5)
    charge = tensor[..., -1]  # Shape: (N, M)
    flavor_charge_combined = one_hot.argmax(dim=-1) * 10 + charge  # Shape: (N, M)
    map_rules = {
        0: 0,  # Photon (1, 0, 0, 0, 0; 0)
        10: 1,  # Neutral hadron (0, 1, 0, 0, 0; 0)
        19: 2,  # Negatively charged hadron (0, 0, 1, 0, 0; -1)
        21: 3,  # Positively charged hadron (0, 0, 1, 0, 0; 1)
        29: 4,  # Negatively charged electron (0, 0, 0, 1, 0; -1)
        31: 5,  # Positively charged electron (0, 0, 0, 1, 0; 1)
        39: 6,  # Negatively charged muon (0, 0, 0, 0, 1; -1)
        41: 7,  # Positively charged muon (0, 0, 0, 0, 1; 1)
    }
    tokens = torch.full_like(
        flavor_charge_combined, -1, dtype=torch.int64
    )  # Initialize with invalid token
    for key, value in map_rules.items():
        tokens[flavor_charge_combined == key] = value
    return tokens.unsqueeze(-1)


def map_tokens_to_basis(tokens):
    """
    Maps a tensor of tokens (integers 0-7) back to the original basis representation.

    Args:
        tokens (torch.Tensor): A tensor of shape (N, M) containing token values (0-7).

    Returns:
        torch.Tensor: A tensor of shape (N, M, 6) with the original basis representation.
    """
    token_to_basis = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],  # Photon 0
            [0, 1, 0, 0, 0, 0],  # Neutral hadron 1
            [0, 0, 1, 0, 0, -1],  # Negatively charged hadron 2
            [0, 0, 1, 0, 0, 1],  # Positively charged hadron 3
            [0, 0, 0, 1, 0, -1],  # Negatively charged electron 4
            [0, 0, 0, 1, 0, 1],  # Positively charged electron 5
            [0, 0, 0, 0, 1, -1],  # Negatively charged muon 6
            [0, 0, 0, 0, 1, 1],  # Positively charged muon 7
        ],
        dtype=torch.float32,
    )
    basis_tensor = token_to_basis[tokens.squeeze(-1)]
    return basis_tensor


def map_basis_to_onehot(tensor):
    return F.one_hot(map_basis_to_tokens(tensor).squeeze(-1), num_classes=8)


def map_onehot_to_basis(onehot):
    return map_tokens_to_basis(onehot.argmax(dim=-1).unsqueeze(-1))
