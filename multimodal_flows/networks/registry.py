from networks.ParticleTransformers import ParticleFormer, KinFormer, FlavorFormer
from networks.BraidedTransformer import MultiModalParticleFormer
from networks.EPiC import EPiC

MODEL_REGISTRY = {"ParticleFormer": ParticleFormer,
                  "MultiModalParticleFormer": MultiModalParticleFormer,
                  "KinFormer": KinFormer,
                  "FlavorFormer": FlavorFormer,
                  "EPiC": EPiC
                  }

