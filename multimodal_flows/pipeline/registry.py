import torch
from datamodules.aoj import AspenOpenJets
from distributions.noise import Noise, SourceUniform, SourceDegenerate, SourceMasked, SourceDataDriven, SourceDirichlet, SourceRandomOneHot
from model.bridges import UniformFlow, SchrodingerBridge, TelegraphBridge, SimplexFlow
from model.thermostats import ConstantThermostat, InverseThermostat, LinearThermostat, InverseSquareThermostat, SigmoidThermostat
from encoders.multimodal_epic import UniModalEPiC, MultiModalEPiC
from encoders.particle_transformer import MultiModalParticleTransformer

registered_datasets = {
    "AspenOpenJets": AspenOpenJets,
    }

registered_distributions = {
    "Noise": Noise,
    "SourceUniform": SourceUniform,
    "SourceDegenerate": SourceDegenerate,
    "SourceMasked": SourceMasked,
    "SourceDataDriven": SourceDataDriven,
    "SourceDirichlet": SourceDirichlet,
    "SourceRandomOneHot":SourceRandomOneHot,
}

registered_bridges = {
    "UniformFlow": UniformFlow,
    "SchrodingerBridge": SchrodingerBridge,
    "TelegraphBridge": TelegraphBridge,
    "SimplexFlow": SimplexFlow,
}

registered_models = {
    "UniModalEPiC": UniModalEPiC,
    "MultiModalEPiC": MultiModalEPiC,
    "MultiModalParticleTransformer": MultiModalParticleTransformer,

}

registered_thermostats = {
    "ConstantThermostat": ConstantThermostat,
    "LinearThermostat": LinearThermostat,
    "InverseThermostat": InverseThermostat,
    "InverseSquareThermostat": InverseSquareThermostat,
    "SigmoidThermostat": SigmoidThermostat,
}

registered_optimizers = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
}

registered_schedulers = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
}