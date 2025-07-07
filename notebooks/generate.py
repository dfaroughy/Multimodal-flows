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
parser.add_argument("--project_name", "-proj", type=str, default='tokenized-jets')
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

model = JetGPT2Model.load_from_checkpoint(f"{config.dir}/{config.project_name}/{config.experiment_id}/checkpoints/{config.checkpoint}", map_location="cpu",)

prompts = torch.full((config.num_jets, 1), model.start_token, dtype=torch.long)
prompt_dataloadeer = DataLoader(prompts, batch_size=config.batch_size, shuffle=False)

callback = GeneratorCallback(config=config)

generator = L.Trainer(accelerator="gpu", 
                      devices='auto', 
                      strategy='ddp', 
                      num_nodes=config.num_nodes, 
                      callbacks=[callback])

generator.predict(model, dataloaders=prompt_dataloadeer)
