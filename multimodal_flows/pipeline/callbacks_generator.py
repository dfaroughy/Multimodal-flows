import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

from pipeline.configs import ExperimentConfigs
from pipeline.helpers import SimpleLogger as log
from pipeline.helpers import setup_logging_dir as makedir
from datamodules.utils import JetFeatures
from tensorclass import TensorMultiModal


class JetGeneratorCallback(Callback):
    def __init__(self, config: ExperimentConfigs):
        super().__init__()

        self.config = config
        self.transform = config.data.transform
        self.batched_paths = []
        self.batched_target_states = []

    def on_predict_start(self, trainer, pl_module):
        self.data_dir = makedir(Path(self.config.path) / "data", exist_ok=False)
        self.metric_dir = makedir(Path(self.config.path) / "metrics", exist_ok=False)
        self.plots_dir = makedir(Path(self.config.path) / "plots", exist_ok=False)
        self.config.save(self.data_dir, name="gen_config.yaml")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is not None:
            self.batched_paths.append(outputs[0])
            self.batched_target_states.append(outputs[1])

    def on_predict_end(self, trainer, pl_module):
        rank = trainer.global_rank

        self._save_results_local(rank)
        trainer.strategy.barrier()  # wait for all ranks to finish

        if trainer.is_global_zero:
            self._gather_results_global(trainer)
            self._clean_temp_files()

    ############
    ### helpers:
    ############

    def _save_results_local(self, rank):
        random = np.random.randint(0, 1000)

        paths = TensorMultiModal.cat(self.batched_paths, dim=1)
        paths.save_to(f"{self.data_dir}/temp_paths_{rank}_{random}.h5")

        test = TensorMultiModal.cat(self.batched_target_states)
        test.save_to(f"{self.data_dir}/temp_test_{rank}_{random}.h5")

    @rank_zero_only
    def _gather_results_global(self, trainer):
        paths_files = self.data_dir.glob("temp_paths_*_*.h5")
        test_files = self.data_dir.glob("temp_test_*_*.h5")

        paths_list = [TensorMultiModal.load_from(str(f)) for f in paths_files]
        paths = TensorMultiModal.cat(paths_list, dim=1)
        paths = self._postprocess(paths, transform=self.transform)
        paths.save_to(f"{self.data_dir}/paths_sample.h5")
        gen_states = paths[-1]
        print(
            gen_states.shape,
            gen_states.continuous,
            gen_states.discrete.shape,
            gen_states.available_modes(),
        )
        gen_jets = JetFeatures(gen_states)

        test_list = [TensorMultiModal.load_from(str(f)) for f in test_files]
        test = TensorMultiModal.cat(test_list)
        test = self._postprocess(test, transform=None)
        test.save_to(f"{self.data_dir}/test_sample.h5")
        test_jets = JetFeatures(test)

        metrics = self.compute_performance_metrics(gen_jets, test_jets)

        if hasattr(self.config, "comet_logger"):
            figures = self.get_results_plots(gen_jets, test_jets)
            for key in figures.keys():
                trainer.logger.experiment.log_figure(
                    figure=figures[key], figure_name=key
                )
            df = pd.DataFrame(metrics)
            trainer.logger.experiment.log_table(
                f"{self.metric_dir}/performance_metrics.csv", df
            )

    def _postprocess(self, paths: TensorMultiModal, transform=None):
        metadata = self._load_metadata(self.config.path)

        if transform == "standardize":
            mean = torch.tensor(metadata["target"]["mean"])
            std = torch.tensor(metadata["target"]["std"])
            paths.continuous = paths.continuous * std + mean

        elif transform == "normalize":
            min_val = torch.tensor(metadata["target"]["min"])
            max_val = torch.tensor(metadata["target"]["max"])
            paths.continuous = paths.continuous * (max_val - min_val) + min_val

        if transform == "log_pt":
            mean = torch.tensor(metadata["target"]["mean"])
            std = torch.tensor(metadata["target"]["std"])
            paths.continuous = paths.continuous * std + mean
            paths.continuous[:, :, 0] = torch.exp(paths.continuous[:, :, 0]) - 1e-6

        if self.config.data.discrete_features == "onehot":
            paths.discrete = paths.continuous.clone()[
                ..., -self.config.data.vocab_size :
            ]
            paths.discrete = torch.argmax(paths.discrete, dim=-1).unsqueeze(-1)
            paths.continuous = paths.continuous[..., : -self.config.data.vocab_size]
            if paths.continuous.shape[-1] == 0:
                del paths.continuous

        paths.apply_mask()
        return paths

    def _clean_temp_files(self):
        for f in self.data_dir.glob("temp_paths_*_*.h5"):
            f.unlink()
        for f in self.data_dir.glob("temp_test_*_*.h5"):
            f.unlink()

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        log.info(f"Loading metadata from {metadata_file}.")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata

    def compute_performance_metrics(self, gen_jets, test_jets):
        continuous_metrics = {}
        discrete_metrics = {}

        if gen_jets.constituents.has_continuous:
            continuous_metrics = {
                "obs": [
                    "pt",
                    "m",
                    "tau21",
                    "tau32",
                    "d2",
                ],
                "W1": [
                    test_jets.Wassertein1D("pt", gen_jets),
                    test_jets.Wassertein1D("m", gen_jets),
                    test_jets.Wassertein1D("tau21", gen_jets),
                    test_jets.Wassertein1D("tau32", gen_jets),
                    test_jets.Wassertein1D("d2", gen_jets),
                ],
            }

        if gen_jets.constituents.has_discrete:
            discrete_metrics = {
                "obs": [
                    "total charge",
                    "Photon  multiplicity",
                    "Neutral Hadron multiplicity",
                    "Charged Hadron multiplicity",
                    "Lepton multiplicity",
                ],
                "W1": [
                    test_jets.Wassertein1D("charge", gen_jets),
                    test_jets.Wassertein1D("numPhotons", gen_jets),
                    test_jets.Wassertein1D("numNeutralHadrons", gen_jets),
                    test_jets.Wassertein1D("numChargedHadrons", gen_jets),
                    test_jets.Wassertein1D("numLeptons", gen_jets),
                ],
            }

        return {**continuous_metrics, **discrete_metrics}

    def get_results_plots(self, gen_jets, test_jets):
        hybrid_plots = {}
        continuous_plots = {}
        discrete_plots = {
            "particle multipicity": self.plot_feature(
                "multiplicity",
                gen_jets.constituents,
                test_jets.constituents,
                apply_map=lambda x: x.squeeze(-1),
                discrete=True,
                xlabel=r"$N$",
            )
        }

        if gen_jets.constituents.has_continuous:
            continuous_plots = {
                "particle transverse momentum": self.plot_feature(
                    "pt",
                    gen_jets.constituents,
                    test_jets.constituents,
                    apply_map="mask_bool",
                    xlabel=r"$p_t$ [GeV]",
                    binrange=(0, 400),
                    binwidth=5,
                    log=True,
                    suffix_file="_part",
                ),
                "particle rapidity": self.plot_feature(
                    "eta_rel",
                    gen_jets.constituents,
                    test_jets.constituents,
                    apply_map="mask_bool",
                    xlabel=r"$\eta^{\rm rel}$",
                    log=True,
                    suffix_file="_part",
                ),
                "particle azimuth": self.plot_feature(
                    "phi_rel",
                    gen_jets.constituents,
                    test_jets.constituents,
                    apply_map="mask_bool",
                    xlabel=r"$\phi^{\rm rel}$",
                    log=True,
                    suffix_file="_part",
                ),
                "jet transverse momentum": self.plot_feature(
                    "pt",
                    gen_jets,
                    test_jets,
                    binrange=(0, 800),
                    binwidth=8,
                    xlabel=r"$p_t$ [GeV]",
                ),
                "jet rapidity": self.plot_feature(
                    "eta",
                    gen_jets,
                    test_jets,
                    xlabel=r"$\eta$",
                ),
                "jet azimuth": self.plot_feature(
                    "phi",
                    gen_jets,
                    test_jets,
                    xlabel=r"$\phi$",
                ),
                "jet mass": self.plot_feature(
                    "m",
                    gen_jets,
                    test_jets,
                    binrange=(0, 250),
                    binwidth=2,
                    xlabel=r"$m$ [GeV]",
                ),
                "energy correlation function": self.plot_feature(
                    "d2",
                    gen_jets,
                    test_jets,
                    xlabel=r"$D_2$",
                ),
                "21-subjetiness ratio": self.plot_feature(
                    "tau21",
                    gen_jets,
                    test_jets,
                    xlabel=r"$\tau_{21}$",
                ),
                "23-subjetiness ratio": self.plot_feature(
                    "tau32",
                    gen_jets,
                    test_jets,
                    xlabel=r"$\tau_{32}$",
                ),
            }

        if gen_jets.constituents.has_discrete:
            discrete_plots = {
                "all hadrons": self.plot_feature(
                    "numHadrons",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{\rm h}$",
                    discrete=True,
                ),
                "all leptons": self.plot_feature(
                    "numLeptons",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{\ell}$",
                    discrete=True,
                ),
                "photon N": self.plot_feature(
                    "numPhotons",
                    gen_jets,
                    test_jets,
                    discrete=True,
                    xlabel=r"$N_{\gamma}$",
                ),
                "neutral hadron N": self.plot_feature(
                    "numNeutralHadrons",
                    gen_jets,
                    test_jets,
                    discrete=True,
                    xlabel=r"$N_{h^0}$",
                ),
                "negative hadron N": self.plot_feature(
                    "numNegativeHadrons",
                    gen_jets,
                    test_jets,
                    discrete=True,
                    xlabel=r"$N_{h^-}$",
                ),
                "positive hadron N": self.plot_feature(
                    "numPositiveHadrons",
                    gen_jets,
                    test_jets,
                    discrete=True,
                    xlabel=r"$N_{h^+}$",
                ),
                "electron N": self.plot_feature(
                    "numElectrons",
                    gen_jets,
                    test_jets,
                    discrete=True,
                    xlabel=r"$N_{e^-}$",
                ),
                "positron N": self.plot_feature(
                    "numPositrons",
                    gen_jets,
                    test_jets,
                    discrete=True,
                    xlabel=r"$N_{e^+}$",
                ),
                "muon N": self.plot_feature(
                    "numMuons",
                    gen_jets,
                    test_jets,
                    discrete=True,
                    xlabel=r"$N_{\mu^-}$",
                ),
                "antimuon N": self.plot_feature(
                    "numAntiMuons",
                    gen_jets,
                    test_jets,
                    discrete=True,
                    xlabel=r"$N_{\mu^+}$",
                ),
                "electric charges": self.plot_charges(gen_jets, test_jets),
                "flavor counts avg": self.plot_flavor_counts_per_jet(
                    gen_jets, test_jets
                ),
            }

        if gen_jets.constituents.has_hybrid:
            hybrid_plots = {
                "gen clouds": self.display_clouds(gen_jets.constituents),
                "photon kin": self.plot_flavored_kinematics(
                    "Photon", gen_jets, test_jets
                ),
                "neutral hadron kin": self.plot_flavored_kinematics(
                    "NeutralHadron", gen_jets, test_jets
                ),
                "negative hadron kin": self.plot_flavored_kinematics(
                    "NegativeHadron", gen_jets, test_jets
                ),
                "positive hadron kin": self.plot_flavored_kinematics(
                    "PositiveHadron", gen_jets, test_jets
                ),
                "electron kin": self.plot_flavored_kinematics(
                    "Electron", gen_jets, test_jets
                ),
                "positron kin": self.plot_flavored_kinematics(
                    "Positron", gen_jets, test_jets
                ),
                "muon kin": self.plot_flavored_kinematics("Muon", gen_jets, test_jets),
                "antimuon kin": self.plot_flavored_kinematics(
                    "AntiMuon", gen_jets, test_jets
                ),
            }

        return {**continuous_plots, **discrete_plots, **hybrid_plots}

    def plot_feature(
        self,
        feat,
        gen,
        test,
        apply_map=None,
        xlabel=None,
        log=False,
        binwidth=None,
        binrange=None,
        discrete=False,
        suffix_file="",
    ):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        gen.histplot(
            feat,
            apply_map=apply_map,
            xlabel=xlabel,
            ax=ax,
            stat="density",
            color="crimson",
            log_scale=(False, log),
            binrange=binrange,
            binwidth=binwidth,
            fill=False,
            label="gen",
            discrete=discrete,
        )
        test.histplot(
            feat,
            apply_map=apply_map,
            xlabel=xlabel,
            ax=ax,
            stat="density",
            color="k",
            log_scale=(False, log),
            binrange=binrange,
            binwidth=binwidth,
            fill=False,
            label="target",
            discrete=discrete,
        )
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{feat}{suffix_file}.png")
        return fig

    def plot_flavor_counts_per_jet(
        self,
        gen,
        test,
    ):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        gen.plot_flavor_count_per_jet(ax=ax, color="crimson")
        test.plot_flavor_count_per_jet(ax=ax, color="k")
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "flavor_counts.png")
        return fig

    def plot_flavored_kinematics(self, flavor, gen, test):
        flavor_labels = {
            "Electron": "{\,e^-}",
            "Positron": "{\,e^+}",
            "Muon": "{\,\mu^-}",
            "AntiMuon": "{\,\mu^+}",
            "Photon": "\gamma",
            "NeutralHadron": "{\,h^0}",
            "NegativeHadron": "{\,h^-}",
            "PositiveHadron": "{\,h^+}",
        }

        fig, ax = plt.subplots(1, 3, figsize=(6, 2))
        test.constituents.histplot(
            f"pt_{flavor}",
            apply_map=None,
            ax=ax[0],
            fill=False,
            bins=100,
            lw=1,
            color="k",
            log_scale=(True, True),
            stat="density",
            xlim=(1e-2, 800),
            label="AOJ",
        )
        gen.constituents.histplot(
            f"pt_{flavor}",
            apply_map=None,
            ax=ax[0],
            fill=False,
            bins=100,
            lw=1,
            color="crimson",
            log_scale=(True, True),
            stat="density",
            xlim=(1e-2, 800),
            label="generated",
        )
        test.constituents.histplot(
            f"eta_{flavor}",
            apply_map=None,
            ax=ax[1],
            fill=False,
            bins=100,
            color="k",
            lw=1,
            log_scale=(False, True),
            stat="density",
            xlim=(-1.2, 1.2),
        )
        gen.constituents.histplot(
            f"eta_{flavor}",
            apply_map=None,
            ax=ax[1],
            fill=False,
            bins=100,
            color="crimson",
            lw=1,
            log_scale=(False, True),
            stat="density",
            xlim=(-1.2, 1.2),
        )
        test.constituents.histplot(
            f"phi_{flavor}",
            apply_map=None,
            ax=ax[2],
            fill=False,
            bins=100,
            color="k",
            lw=1,
            log_scale=(False, True),
            stat="density",
            xlim=(-1.2, 1.2),
        )
        gen.constituents.histplot(
            f"phi_{flavor}",
            apply_map=None,
            ax=ax[2],
            fill=False,
            bins=100,
            color="crimson",
            lw=1,
            log_scale=(False, True),
            stat="density",
            xlim=(-1.2, 1.2),
        )
        ax[0].set_xlabel(rf"$p_T^{flavor_labels[flavor]}$")
        ax[1].set_xlabel(rf"$\eta^{flavor_labels[flavor]}$")
        ax[2].set_xlabel(rf"$\phi^{flavor_labels[flavor]}$")
        ax[0].set_ylabel("density")

        plt.tight_layout()
        plt.legend(fontsize=10)
        plt.savefig(self.plots_dir / f"kinematics_{flavor}.png")
        return fig

    def plot_charges(self, gen, test):
        fig, ax = plt.subplots(1, 3, figsize=(6, 2))
        test.histplot(
            "numNeutrals",
            ax=ax[0],
            fill=False,
            discrete=True,
            lw=1,
            color="k",
            stat="density",
            xlabel=r"$N_{Q=0}$",
        )
        gen.histplot(
            "numNeutrals",
            ax=ax[0],
            fill=False,
            discrete=True,
            lw=1,
            color="crimson",
            stat="density",
            xlabel=r"$N_{Q=0}$",
        )
        test.histplot(
            "numCharged",
            ax=ax[1],
            fill=False,
            discrete=True,
            color="k",
            lw=1,
            stat="density",
            xlabel=r"$N_{Q=\pm1}$",
        )
        gen.histplot(
            "numCharged",
            ax=ax[1],
            fill=False,
            discrete=True,
            color="crimson",
            lw=1,
            stat="density",
            xlabel=r"$N_{Q=\pm1}$",
        )
        test.histplot(
            "charge",
            ax=ax[2],
            fill=False,
            discrete=True,
            color="k",
            lw=1,
            stat="density",
            xlabel=r"$Q_{\rm jet}^{\kappa=0}$",
        )
        gen.histplot(
            "charge",
            ax=ax[2],
            fill=False,
            discrete=True,
            color="crimson",
            lw=1,
            stat="density",
            xlabel=r"$Q_{\rm jet}^{\kappa=0}$",
        )
        ax[0].set_xticks([0, 20, 40, 60])
        ax[1].set_xticks([0, 20, 40, 60, 80])
        ax[2].set_xticks([-20, -10, 0, 10, 20])
        ax[0].set_ylabel("density")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "charges.png")
        return fig

    def display_clouds(self, jets):
        fig, ax = plt.subplots(2, 5, figsize=(16, 6))
        jets.display_cloud(0, scale_marker=10, ax=ax[0, 0])
        jets.display_cloud(1, scale_marker=10, ax=ax[0, 1])
        jets.display_cloud(2, scale_marker=10, ax=ax[0, 2])
        jets.display_cloud(3, scale_marker=10, ax=ax[0, 3])
        jets.display_cloud(4, scale_marker=10, ax=ax[0, 4])
        jets.display_cloud(5, scale_marker=10, ax=ax[1, 0])
        jets.display_cloud(6, scale_marker=10, ax=ax[1, 1])
        jets.display_cloud(7, scale_marker=10, ax=ax[1, 2])
        jets.display_cloud(8, scale_marker=10, ax=ax[1, 3])
        jets.display_cloud(9, scale_marker=10, ax=ax[1, 4])
        plt.tight_layout()
        plt.savefig(self.plots_dir / "gen_particle_clouds.png", dpi=500)
        return fig
