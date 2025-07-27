from networks.ParticleTransformers import ParticleFormer, KinFormer, FlavorFormer, KinFlavorFormer
from networks.MultiModalTransformers import FusedParticleFormer, GatedParticleFormer
from networks.EPiC import EPiC

MODEL_REGISTRY = {"ParticleFormer": ParticleFormer,
                  "KinFormer": KinFormer,
                  "FlavorFormer": FlavorFormer,
                  "KinFlavorFormer": KinFlavorFormer,
                  "FusedParticleFormer": FusedParticleFormer,
                  "GatedParticleFormer": GatedParticleFormer,
                  "EPiC": EPiC
                  }

