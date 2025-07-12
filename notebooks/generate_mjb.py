import numpy as np
import torch
from argparse import ArgumentParser
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning as L
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from transformers import GPT2Config

from utils.tensorclass import TensorMultiModal
from utils.aoj import AspenOpenJets 
from utils.datasets import MultiModalDataset, DataCoupling, data_coupling_collate_fn
from utils.callbacks import GPTGeneratorCallback
from model.MJB import MarkovJumpBridge


###############################################################################

parser = ArgumentParser()

parser.add_argument("--num_nodes", "-N", type=int, default=1)
parser.add_argument("--dir", type=str, default='/home/df630/Multimodal-flows')
parser.add_argument("--project", "-proj", type=str, default='jet_sequences')
parser.add_argument("--comet_workspace", type=str, default='dfaroughy')
parser.add_argument("--comet_api_key", type=str, default='8ONjCXJ1ogsqG1UxQzKxYn7tz')
parser.add_argument("--experiment_id", "-id", type=str, default=None)

parser.add_argument("--num_jets", "-n", type=int, default=100_000)
parser.add_argument("--num_timesteps", "-steps", type=int, default=100)
parser.add_argument("--batch_size", "-bs", type=int, default=256)
parser.add_argument("--tag", "-t", type=str, default='')

config = parser.parse_args()

###############################################################################

import matplotlib.pyplot as plt
import seaborn as sns

def plot_flavor_feats(sample, particle_set, path_dir=None):

    #...Low-level feats

    fig, ax = plt.subplots(2, 4, figsize=(10,3.5))

    sns.histplot((particle_set.discrete == 1).sum(dim=1), discrete=True, stat="density", element="step", fill=True, ax=ax[0, 0], lw=0.0)
    sns.histplot((sample == 1).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 0], lw=0.75)
    ax[0, 0].set_xlabel(r"$N_{\gamma}$")

    sns.histplot((particle_set.discrete == 2).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[0, 1],lw=0.0)
    sns.histplot((sample == 2).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 1],lw=0.75)
    ax[0, 1].set_xlabel(r"$N_{h^0}$")

    sns.histplot((particle_set.discrete == 3).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[0, 2],lw=0.0)
    sns.histplot((sample == 3).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 2],lw=0.75)
    ax[0, 2].set_xlabel(r"$N_{h^{-}}$")

    sns.histplot((particle_set.discrete == 4).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[0, 3],lw=0.0)
    sns.histplot((sample == 4).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 3],lw=0.75)
    ax[0, 3].set_xlabel(r"$N_{h^{+}}$")

    sns.histplot((particle_set.discrete == 5).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 0],lw=0.0)
    sns.histplot((sample == 5).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 0],lw=0.75)
    ax[1, 0].set_xlabel(r"$N_{e^{-}}$")

    sns.histplot((particle_set.discrete == 6).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 1],lw=0.0)
    sns.histplot((sample == 6).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 1],lw=0.75)
    ax[1, 1].set_xlabel(r"$N_{e^{+}}$")

    sns.histplot((particle_set.discrete == 7).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 2],lw=0.0)
    sns.histplot((sample == 7).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 2],lw=0.75)
    ax[1, 2].set_xlabel(r"$N_{\mu^{-}}$")

    sns.histplot((particle_set.discrete == 8).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 3],lw=0.0)
    sns.histplot((sample == 8).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 3],lw=0.75)
    ax[1, 3].set_xlabel(r"$N_{\mu^{+}}$")

    plt.tight_layout()
    for a in ax.flatten():
        a.legend([], [], frameon=False)

    plt.savefig(f'{path_dir}/jet_flavor_low_level.png', dpi=500, bbox_inches='tight')

    #...High-level feats

    fig, ax = plt.subplots(2, 3, figsize=(8,3.5))

    sns.histplot((particle_set.discrete > 0).sum(dim=1), discrete=True, stat="density", element="step", fill=True, ax=ax[0, 0], lw=0.0)
    sns.histplot((sample > 0).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 0], lw=0.75)
    ax[0, 0].set_xlabel(r"$N$")

    sns.histplot(((particle_set.discrete > 1) & (particle_set.discrete < 5)).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[0, 1],lw=0.0)
    sns.histplot(((sample > 1) & (sample < 5)).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 1],lw=0.75)
    ax[0, 1].set_xlabel(r"$N_{\rm had}$")

    sns.histplot((particle_set.discrete > 4).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[0, 2],lw=0.0)
    sns.histplot((sample > 4).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 2],lw=0.75)
    ax[0, 2].set_xlabel(r"$N_{\rm lep}$")

    sns.histplot(((particle_set.discrete == 1) | (particle_set.discrete == 2)).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 0],lw=0.0)
    sns.histplot(((sample == 1) | (sample == 2)).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 0],lw=0.75)
    ax[1, 0].set_xlabel(r"$N_{0}$")

    sns.histplot(((particle_set.discrete == 3) | (particle_set.discrete == 5) | (particle_set.discrete == 7)).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 1],lw=0.0)
    sns.histplot(((sample == 3) | (sample == 5) | (sample == 7)).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 1],lw=0.75)
    ax[1, 1].set_xlabel(r"$N_{-}$")

    sns.histplot(((particle_set.discrete == 4) | (particle_set.discrete == 6) | (particle_set.discrete == 8)).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 2],lw=0.0)
    sns.histplot(((sample == 4) | (sample == 6) | (sample == 8)).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 2],lw=0.75)
    ax[1, 2].set_xlabel(r"$N_{+}$")


    plt.tight_layout()
    for a in ax.flatten():
        a.legend([], [], frameon=False)

    plt.savefig(f'{path_dir}/jet_flavor_high_level.png', dpi=500, bbox_inches='tight')

###############################################################################

mjb = MarkovJumpBridge.load_from_checkpoint(f"/home/df630/Multimodal-flows/jet_sequences/{config.experiment_id}/checkpoints/best.ckpt")

#...dataset & dataloaders:

noise = torch.randint(mjb.vocab_size, (config.num_jets, mjb.max_num_particles, 1))
mask = torch.ones_like(noise).long()
t0 = torch.full((len(noise),), mjb.time_eps)  # (B) t_0=eps

source = TensorMultiModal(time=t0, discrete=noise, mask=mask)
source = source.to(mjb.device)
data = DataCoupling(source=source, target=TensorMultiModal())

#...sample dynamics:

sample = mjb.simulate_dynamics(data)
sample = sample.target.detach().cpu()
sample = sample.discrete.squeeze(-1)

aoj_test = AspenOpenJets(data_dir="/home/df630/Multimodal-Bridges/data/aoj", data_files="RunG_batch1.h5")
test_data, _ = aoj_test(num_jets=config.num_jets,
                        download=False,
                        features={"continuous": None, "discrete": "tokens"},
                        pt_order=True,
                        padding='zeros')

plot_flavor_feats(sample, 
                  test_data, 
                  path_dir=f"{config.dir}/{config.project}/{config.experiment_id}")
