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

