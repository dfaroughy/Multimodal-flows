from networks.ParticleTransformers import ParticleFormer, FusedParticleFormer, KinFormer, FlavorFormer 
from networks.MultiModalTransformers import GatedParticleFormer
from networks.EPiC import EPiC

MODEL_REGISTRY = {"ParticleFormer": ParticleFormer,
                  "KinFormer": KinFormer,
                  "FlavorFormer": FlavorFormer,
                  "FusedParticleFormer": FusedParticleFormer,
                  "GatedParticleFormer": GatedParticleFormer,
                  "EPiC": EPiC
                  }

