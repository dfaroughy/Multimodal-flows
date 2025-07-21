import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm as tqdm
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from typing import Union

from utils.metrics import flavor_mutliplicities
from utils.tensorclass import TensorMultiModal
from utils.aoj import JetFeatures   

def plot_hist_and_ratio(test, 
                        gen,  
                        gen_ref,
                        ax_hist, 
                        ax_ratio, 
                        apply_map_test=None,
                        apply_map_gen=None,
                        apply_map_gen_ref=None,
                        xlabel=None, 
                        xlim=None,
                        ylim=None,
                        ylabel='',
                        feat=None,
                        num_bins=100, 
                        log_scale=False, 
                        color1=None,
                        color2=None,
                        color3='dodgerblue',
                        color_test='darkslategrey',
                        legend1=None,
                        legend2=None,
                        legend3=None,
                        discrete=False,
                        fill=False,
                        lw=0.75,
                        ls='-',):

    if xlim is not None:    
        bins = np.linspace(xlim[0], xlim[1], num_bins)  # Adjust binning if necessary
    else:
        bins = num_bins

    if apply_map_test=='remove_zeros':
        apply_map_test = lambda test: test[test!=0]

    if apply_map_gen=='remove_zeros':
        apply_map_gen = lambda gen: gen[gen!=0]

    if gen_ref is not None:
        if apply_map_gen_ref=='remove_zeros':
            apply_map_gen_ref = lambda gen_ref: gen_ref[gen_ref!=0]

    if feat is not None:
        test.histplot(feat, apply_map=apply_map_test,  stat='density', fill=fill,bins=bins, alpha=1 if not fill else 0.2, ax=ax_hist, lw=1.25 if not fill else 0.35, color=color_test, label="AOJ (truth)", log_scale=log_scale, discrete=discrete)
        gen.histplot(feat, apply_map=apply_map_gen,  stat='density', fill=False, bins=bins, ax=ax_hist, ls=ls, lw=lw, color=color1, label=legend1, log_scale=log_scale, discrete=discrete)
        x = getattr(test, feat)
        # x = x[x!=0]
        y1 = getattr(gen, feat)
        if gen_ref is not None: 
            gen_ref.histplot(feat, apply_map=apply_map_gen_ref,  stat='density', fill=False, bins=bins, ax=ax_hist, ls=ls, lw=lw, color=color2, label=legend2, log_scale=log_scale, discrete=discrete)
            y2 = getattr(gen_ref, feat)
        # y1 = y1[y1!=0]
        # y2 = y2[y2!=0]

    else:
        sns.histplot(test, stat='density', fill=fill, element='step', bins=bins, ax=ax_hist, alpha=1 if not fill else 0.3, lw=1.25 if not fill else 0.35, color=color_test, label="AOJ (truth)", log_scale=log_scale, discrete=discrete)
        sns.histplot(gen, stat='density', fill=False, element='step' ,bins=bins, ax=ax_hist, ls=ls, lw=lw, color=color1, label=legend1, log_scale=log_scale, discrete=discrete)
        x = test
        y1 = gen
        if gen_ref is not None: 
            sns.histplot(gen_ref, stat='density', fill=False, element='step' ,bins=bins, ax=ax_hist, ls=ls, lw=lw, color=color2, label=legend2, log_scale=log_scale, discrete=discrete)
            y2 = gen_ref

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    if isinstance(y1, torch.Tensor):
        y1 = y1.cpu().numpy()
        if gen_ref is not None: y2 = y2.cpu().numpy()

    hist, _ = np.histogram(x, bins=bins, density=True)
    hist1, _ = np.histogram(y1, bins=bins, density=True)
    ratio1 = np.divide(hist1, hist, out=np.ones_like(hist1, dtype=float), where=hist > 0)
    
    if gen_ref is not None:
        hist2, _ = np.histogram(y2, bins=bins, density=True)
        ratio2 = np.divide(hist2, hist, out=np.ones_like(hist2, dtype=float), where=hist > 0)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot ratio
    ax_ratio.hist(bin_centers, bins=bins, weights=ratio1, histtype="step", color=color1, ls=ls)
    if gen_ref is not None:
        ax_ratio.hist(bin_centers, bins=bins, weights=ratio2, histtype="step", color=color2, ls=ls)
    ax_ratio.axhline(1.0, color='gray', linestyle='--', lw=0.5)

    # Format axes
    ax_hist.set_ylabel(ylabel, fontsize=10)
    
    if legend1 is not None:
        ax_hist.legend(fontsize=8)

    ax_ratio.set_ylabel("Ratio", fontsize=8)

    if xlabel is not None:
        ax_ratio.set_xlabel(xlabel, fontsize=15)

    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_yticks([0.7, 1.0, 1.3])
    ax_ratio.set_yticklabels([0.7, 1.0, 1.3], fontsize=7)

    if xlim is not None:
        ax_hist.set_xlim(*xlim)
        ax_ratio.set_xlim(*xlim)

    if ylim is not None:
        ax_hist.set_ylim(*ylim)


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


def plot_kin_feats(sample, test, path=None):


    fig, ax = plt.subplots(3, 4, figsize=(10, 8))

    # particle level

    bins = np.linspace(0, 700, 100)
    sns.histplot(test.constituents.pt[test.constituents.mask_bool].numpy(), bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[0,0], log_scale=(False, True), lw=0.75, label="AOJ (truth)")
    sns.histplot(sample.constituents.pt[sample.constituents.mask_bool].numpy(), bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[0,0], log_scale=(False, True), lw=0.75, label="MMF - ParticleFormer")
    ax[0,0].set_xlabel(r"$p_T$ [GeV]", fontsize=10)

    bins = np.linspace(-1.1, 1.1, 100)
    sns.histplot(test.constituents.eta_rel[test.constituents.mask_bool].numpy(),  bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[0,1], log_scale=(False, True), lw=0.75)
    sns.histplot(sample.constituents.eta_rel[sample.constituents.mask_bool].numpy(),  bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[0,1], log_scale=(False, True), lw=0.75)
    ax[0,1].set_xlabel(r"$\Delta\eta$", fontsize=10)

    bins = np.linspace(-1.1, 1.1, 100)
    sns.histplot(test.constituents.phi_rel[test.constituents.mask_bool].numpy(),  bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[0,2], log_scale=(False, True), lw=0.75)
    sns.histplot(sample.constituents.phi_rel[sample.constituents.mask_bool].numpy(),  bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[0,2], log_scale=(False, True), lw=0.75)
    ax[0,2].set_xlabel(r"$\Delta\phi$", fontsize=10)

    bins = np.linspace(-1.1, 1.1, 100)
    sns.histplot(test.constituents.multiplicity.numpy(),  bins=bins, discrete=True, stat="density", element="step", fill=False, ax=ax[0,3], lw=0.75)
    sns.histplot(sample.constituents.multiplicity.numpy(),  bins=bins, discrete=True, stat="density", element="step", fill=False, ax=ax[0,3], lw=0.75)
    ax[0,3].set_xlabel(r"$\Delta\phi$", fontsize=10)

    # jet level

    bins = np.linspace(200, 700, 100)
    sns.histplot(test.pt.numpy(), bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[1,0],  lw=0.75)
    sns.histplot(sample.pt.numpy(), bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[1,0],  lw=0.75)
    ax[1,0].set_xlabel(r"$p_T$ [GeV]", fontsize=10)


    bins = np.linspace(-.1, .1, 100)
    sns.histplot(test.eta.numpy(),  bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[1,1],  lw=0.75)
    sns.histplot(sample.eta.numpy(),  bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[1,1], lw=0.75)
    ax[1,1].set_xlabel(r"$\eta$", fontsize=10)

    bins = np.linspace(-.01, .01, 100)
    sns.histplot(test.phi.numpy(),  bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[1,2], lw=0.75)
    sns.histplot(sample.phi.numpy(),  bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[1,2], lw=0.75)
    ax[1,2].set_xlabel(r"$\phi$", fontsize=10)

    bins = np.linspace(0, 250, 100)
    sns.histplot(test.m.numpy(),  bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[1,3], lw=0.75) 
    sns.histplot(sample.m.numpy(),  bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[1,3], lw=0.75)
    ax[1,3].set_xlabel(r"mass [GeV]", fontsize=10)

    # substructure

    bins = np.linspace(0, .3, 100)
    sns.histplot(test.c1, bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[2,0], lw=0.75)
    sns.histplot(sample.c1, bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[2,0], lw=0.75)
    ax[2,0].set_xlabel(r"$C_1$", fontsize=10)

    bins = np.linspace(0, 12, 100)
    sns.histplot(test.d2, bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[2,1], lw=0.75)
    sns.histplot(sample.d2, bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[2,1], lw=0.75)
    ax[2,1].set_xlabel(r"$D_2$", fontsize=10)

    bins = np.linspace(0, 1, 100)
    sns.histplot(test.tau21, bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[2,2],  lw=0.75)
    sns.histplot(sample.tau21, bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[2,2],  lw=0.75)
    ax[2,2].set_xlabel(r"$\tau_{21}$", fontsize=10)

    bins = np.linspace(0, 1, 100)
    sns.histplot(test.tau32, bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[2,3],  lw=0.75)
    sns.histplot(sample.tau32, bins=bins, discrete=False, stat="density", element="step", fill=False, ax=ax[2,3],  lw=0.75)
    ax[2,3].set_xlabel(r"$\tau_{32}$", fontsize=10)

    plt.tight_layout()

    for a in ax.flatten():
        a.legend([], [], frameon=False)

    if path: plt.savefig(path, dpi=500, bbox_inches='tight')
    else: plt.show()