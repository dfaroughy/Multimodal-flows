from networks.ParticleTransformers import ParticleFormer, FusedParticleFormer, KinFormer, FlavorFormer 
from networks.EPiC import EPiC

MODEL_REGISTRY = {"ParticleFormer": ParticleFormer,
                  "KinFormer": KinFormer,
                  "FlavorFormer": FlavorFormer,
                  "FusedParticleFormer": FusedParticleFormer,
                  "EPiC": EPiC
                  }

