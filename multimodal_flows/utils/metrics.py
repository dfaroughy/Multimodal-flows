import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
from scipy.stats import wasserstein_distance

from utils.tensorclass import TensorMultiModal


def flavor_mutliplicities(sample):
    """
    Extract low-level features from the sample tensor.
    """
    feats = {
        # 'pads': (sample == 0).sum(dim=1), 
        'photons': (sample == 1).sum(dim=1),
        'h0': (sample == 2).sum(dim=1),
        'h-': (sample == 3).sum(dim=1),
        'h+': (sample == 4).sum(dim=1),
        'e-': (sample == 5).sum(dim=1),
        'e+': (sample == 6).sum(dim=1),
        'mu-': (sample == 7).sum(dim=1),
        'mu+': (sample == 8).sum(dim=1),   
        'multiplicity': (sample > 0).sum(dim=1),  # total number of particles
        'hadrons': ((sample >= 2) & (sample <= 4)).sum(dim=1),  # hadrons
        'leptons': (sample > 4).sum(dim=1),  # leptons
        'neutrals': ((sample == 1) | (sample == 2)).sum(dim=1),  # neutral particles
        'negatives': ((sample == 3) | (sample == 5) | (sample == 7)).sum(dim=1),  # negative particles
        'positives': ((sample == 4) | (sample == 6) | (sample == 8)).sum(dim=1),  # positive particles
        'isospin': (sample == 1).sum(dim=1) - (sample == 4).sum(dim=1),  # isospin
        'net charge': ((sample == 3) | (sample == 5) | (sample == 7)).sum(dim=1) - ((sample == 4) | (sample == 6) | (sample == 8)).sum(dim=1),  # net charge
    }
    return feats


def wasserstein_flavor(sample: Union[TensorMultiModal, torch.Tensor], test: Union[TensorMultiModal, torch.Tensor], path: str=None):
    """
    Compute the Wasserstein distance between the multiplicity distributions of a specific feature.
    
    Args:
        sample (TensorMultiModal): The sample tensor containing discrete particle data.
        particle_set (TensorMultiModal): The particle set tensor containing discrete particle data.
        feature_name (str): The name of the feature to compute the Wasserstein distance for.
        
    Returns:
        float: The computed Wasserstein distance.
    """
    
    if isinstance(sample, TensorMultiModal): feats_sample = flavor_mutliplicities(sample.discrete)
    elif isinstance(sample, torch.Tensor): feats_sample = flavor_mutliplicities(sample)
    elif isinstance(sample, dict): feats_sample = sample

    if isinstance(test, TensorMultiModal): feats_test = flavor_mutliplicities(test.discrete)
    elif isinstance(test, torch.Tensor): feats_test = flavor_mutliplicities(test)
    elif isinstance(test, dict): feats_test = test

    w1 = {}

    for (key, feat) in feats_sample.items():
        w1[key] = wasserstein_distance(feat.cpu().numpy(), feats_test[key].cpu().numpy())

    if path:
        with open(path, 'w') as f:
            for key, dist in w1.items():
                if dist is not None:
                    f.write(f"{key}: {dist:.4f}\n")
    return w1

def plot_flavor_feats(sample: Union[TensorMultiModal, torch.Tensor], test: Union[TensorMultiModal, torch.Tensor], path=None):
    """
    Plot the flavor multiplicities of the sample and particle_set tensors.
    
    Args:
        sample (TensorMultiModal): The sample tensor containing discrete particle data.
        particle_set (TensorMultiModal): The particle set tensor containing discrete particle data.
        path_dir (str, optional): Directory to save the plots. If None, plots are not saved.
    """
    
    if isinstance(sample, TensorMultiModal): feats_sample = flavor_mutliplicities(sample.discrete)
    elif isinstance(sample, torch.Tensor): feats_sample = flavor_mutliplicities(sample)
    elif isinstance(sample, dict): feats_sample = sample

    if isinstance(test, TensorMultiModal): feats_test = flavor_mutliplicities(test.discrete)
    elif isinstance(test, torch.Tensor): feats_test = flavor_mutliplicities(test)
    elif isinstance(test, dict): feats_test = test

    fig, ax = plt.subplots(4, 4, figsize=(12, 10))

    for i, (key, feat) in enumerate(feats_sample.items()):
        sns.histplot(feats_test[key], discrete=True, stat="density", element="step", fill=False, ax=ax[i // 4, i % 4], lw=1, ls='-')
        sns.histplot(feat, discrete=True, stat="density", element="step", fill=False, ax=ax[i // 4, i % 4], lw=1, ls='--')
        ax[i // 4, i % 4].set_xlabel(key)

    plt.tight_layout()

    for a in ax.flatten():
        a.legend([], [], frameon=False)

    if path: plt.savefig(path, dpi=500, bbox_inches='tight')
    else: plt.show()


# def plot_flavor_feats(sample, particle_set, path_dir=None):

#     #...Low-level feats

#     fig, ax = plt.subplots(2, 4, figsize=(10,3.5))

#     sns.histplot((particle_set.discrete == 1).sum(dim=1), discrete=True, stat="density", element="step", fill=True, ax=ax[0, 0], lw=0.0)
#     sns.histplot((sample == 1).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 0], lw=0.75)
#     ax[0, 0].set_xlabel(r"$N_{\gamma}$")

#     sns.histplot((particle_set.discrete == 2).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[0, 1],lw=0.0)
#     sns.histplot((sample == 2).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 1],lw=0.75)
#     ax[0, 1].set_xlabel(r"$N_{h^0}$")

#     sns.histplot((particle_set.discrete == 3).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[0, 2],lw=0.0)
#     sns.histplot((sample == 3).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 2],lw=0.75)
#     ax[0, 2].set_xlabel(r"$N_{h^{-}}$")

#     sns.histplot((particle_set.discrete == 4).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[0, 3],lw=0.0)
#     sns.histplot((sample == 4).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 3],lw=0.75)
#     ax[0, 3].set_xlabel(r"$N_{h^{+}}$")

#     sns.histplot((particle_set.discrete == 5).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 0],lw=0.0)
#     sns.histplot((sample == 5).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 0],lw=0.75)
#     ax[1, 0].set_xlabel(r"$N_{e^{-}}$")

#     sns.histplot((particle_set.discrete == 6).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 1],lw=0.0)
#     sns.histplot((sample == 6).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 1],lw=0.75)
#     ax[1, 1].set_xlabel(r"$N_{e^{+}}$")

#     sns.histplot((particle_set.discrete == 7).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 2],lw=0.0)
#     sns.histplot((sample == 7).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 2],lw=0.75)
#     ax[1, 2].set_xlabel(r"$N_{\mu^{-}}$")

#     sns.histplot((particle_set.discrete == 8).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 3],lw=0.0)
#     sns.histplot((sample == 8).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 3],lw=0.75)
#     ax[1, 3].set_xlabel(r"$N_{\mu^{+}}$")

#     plt.tight_layout()
#     for a in ax.flatten():
#         a.legend([], [], frameon=False)

#     plt.savefig(f'{path_dir}/jet_flavor_low_level.png', dpi=500, bbox_inches='tight')

#     #...High-level feats

#     fig, ax = plt.subplots(2, 3, figsize=(8,3.5))

#     sns.histplot((particle_set.discrete > 0).sum(dim=1), discrete=True, stat="density", element="step", fill=True, ax=ax[0, 0], lw=0.0)
#     sns.histplot((sample > 0).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 0], lw=0.75)
#     ax[0, 0].set_xlabel(r"$N$")

#     sns.histplot(((particle_set.discrete > 1) & (particle_set.discrete < 5)).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[0, 1],lw=0.0)
#     sns.histplot(((sample > 1) & (sample < 5)).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 1],lw=0.75)
#     ax[0, 1].set_xlabel(r"$N_{\rm had}$")

#     sns.histplot((particle_set.discrete > 4).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[0, 2],lw=0.0)
#     sns.histplot((sample > 4).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[0, 2],lw=0.75)
#     ax[0, 2].set_xlabel(r"$N_{\rm lep}$")

#     sns.histplot(((particle_set.discrete == 1) | (particle_set.discrete == 2)).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 0],lw=0.0)
#     sns.histplot(((sample == 1) | (sample == 2)).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 0],lw=0.75)
#     ax[1, 0].set_xlabel(r"$N_{0}$")

#     sns.histplot(((particle_set.discrete == 3) | (particle_set.discrete == 5) | (particle_set.discrete == 7)).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 1],lw=0.0)
#     sns.histplot(((sample == 3) | (sample == 5) | (sample == 7)).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 1],lw=0.75)
#     ax[1, 1].set_xlabel(r"$N_{-}$")

#     sns.histplot(((particle_set.discrete == 4) | (particle_set.discrete == 6) | (particle_set.discrete == 8)).sum(dim=1), discrete=True, stat="density",element="step", fill=True, ax=ax[1, 2],lw=0.0)
#     sns.histplot(((sample == 4) | (sample == 6) | (sample == 8)).sum(dim=1), discrete=True, stat="density",element="step", fill=False, ax=ax[1, 2],lw=0.75)
#     ax[1, 2].set_xlabel(r"$N_{+}$")


#     plt.tight_layout()
#     for a in ax.flatten():
#         a.legend([], [], frameon=False)

#     plt.savefig(f'{path_dir}/jet_flavor_high_level.png', dpi=500, bbox_inches='tight')
