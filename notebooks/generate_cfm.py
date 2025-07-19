
import numpy as np
import torch
from model.CFM import ConditionalFlowMatching
import matplotlib.pyplot as plt
import seaborn as sns

from train_cfm import experiment_configs
config = experiment_configs("c9c27af41e7b4f23ae29bd86136d5bcd")

print(config)
# cfm = ConditionalFlowMatching.load_from_checkpoint(f"/home/df630/Multimodal-flows/jet_sequences/{experiment_id}/checkpoints/best.ckpt", config=config)

# print(cfm)

# #...dataset & dataloaders:

# noise = torch.randn((config.num_jets, cfm.max_num_particles, cfm.vocab_size),)
# mask = torch.ones((config.num_jets, cfm.max_num_particles, 1),).long()
# t0 = torch.full((len(noise),), cfm.time_eps)  # (B) t_0=eps

# source = TensorMultiModal(time=t0, continuous=noise, mask=mask)
# source = source.to(cfm.device)
# data = DataCoupling(source=source, target=TensorMultiModal())

# #...sample dynamics:

# sample = cfm.simulate_dynamics(data)
# sample = sample.target.detach().cpu()
# sample = torch.argmax(sample.continuous, dim=-1).squeeze(-1)

# aoj_test = AspenOpenJets(data_dir="/home/df630/Multimodal-Bridges/data/aoj", data_files="RunG_batch1.h5")
# test_data, _ = aoj_test(num_jets=config.num_jets,
#                         download=False,
#                         features={"continuous": None, "discrete": "tokens"},
#                         pt_order=True,
#                         padding='zeros')

# plot_flavor_feats(sample, 
#                   test_data, 
#                   path_dir=f"{config.dir}/{config.project}/{config.experiment_id}")
