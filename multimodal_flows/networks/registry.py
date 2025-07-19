
from networks.ParticleTransformers import ParticleFormer, KinFormer, FlavorFormer
# from networks.EPiC import EPiC

MODEL_REGISTRY = {"ParticleFormer": ParticleFormer,
                  "KinFormer": KinFormer,
                  "FlavorFormer": FlavorFormer,
                #   "EPiC": EPiC
                  }

