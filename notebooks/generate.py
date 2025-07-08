import numpy as np
import torch
import pytorch_lightning as L
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from models import JetGPT2Model
from utils import GeneratorCallback


import numpy as np
import torch
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning as L
import matplotlib.pyplot as plt
from datamodules.aoj import AspenOpenJets 
from transformers import GPT2LMHeadModel, GPT2Config

from torch.utils.data import DataLoader, random_split
from tensorclass import TensorMultiModal
from datamodules.datasets import MultiModalDataset, DataCoupling, data_coupling_collate_fn
from model.multimodal_bridge_matching import MarkovJumpBridge

###############################################################################
parser = ArgumentParser()

parser.add_argument("--dir", type=str, default='/pscratch/sd/d/dfarough')
parser.add_argument("--predict_type", type=str, default='gen')
parser.add_argument("--num_nodes", "-N", type=int, default=1)
parser.add_argument("--project", "-proj", type=str, default='tokenized-jets')
parser.add_argument("--experiment_id", "-id", type=str, default=None)
parser.add_argument("--tag", type=str, default=None)
parser.add_argument("--data_path", type=str, default='/pscratch/sd/d/dfarough/JetClass')
parser.add_argument("--checkpoint", "-ckpt", type=str, default='best.ckpt')
parser.add_argument("--jet_type", "-type", type=str, default=None)
parser.add_argument("--max_seq_length", "-len", type=int, default=200)
parser.add_argument("--top_k", type=int, default=5000)
parser.add_argument("--num_jets", "-n", type=int, default=1000000)
parser.add_argument("--batch_size", "-bs", type=int, default=1024)
parser.add_argument("--plots", "-plt", type=bool, default=False)

config = parser.parse_args()
###############################################################################


import torch
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning as L
from torch.utils.data import DataLoader, random_split
from tensorclass import TensorMultiModal
from datamodules.datasets import MultiModalDataset, DataCoupling, data_coupling_collate_fn
from model.multimodal_bridge_matching import MarkovJumpBridge


experiment_id = '62ddb0e522d945a7982785b40f7ed96a'
vocab_size = 9
num_jets = 3000
max_num_particles = 150
batch_size=100

mjb = MarkovJumpBridge.load_from_checkpoint(f"/home/df630/Multimodal-flows/results/jet_sequences/{config.experiment_id}/checkpoints/best.ckpt", map_location="cpu",)


noise = torch.randint(0, vocab_size, (num_jets, max_num_particles, 1))
mask = torch.ones((num_jets, max_num_particles, 1))
source = TensorMultiModal(discrete=noise, mask=mask)

data = DataCoupling(source=source, target=TensorMultiModal())
dataset = MultiModalDataset(data)
predict_dataloader   = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_coupling_collate_fn)

