
import numpy as np
import torch
from argparse import ArgumentParser
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning as L
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from transformers import GPT2Config

from tensorclass import TensorMultiModal
from datamodules.aoj import AspenOpenJets 
from datamodules.utils import jet_set_to_seq
from datamodules.datasets import MultiModalDataset, DataCoupling, data_coupling_collate_fn
from pipeline.callbacks_generator_gpt import GPTGeneratorCallback
from model.GPT import JetFlavorSeqGPT

###############################################################################

parser = ArgumentParser()

parser.add_argument("--num_nodes", "-N", type=int, default=1)
parser.add_argument("--dir", type=str, default='/home/df630/Multimodal-flows')
parser.add_argument("--project", "-proj", type=str, default='jet_sequences')
parser.add_argument("--comet_workspace", type=str, default='dfaroughy')
parser.add_argument("--comet_api_key", type=str, default='8ONjCXJ1ogsqG1UxQzKxYn7tz')
parser.add_argument("--experiment_id", "-id", type=str, default=None)
parser.add_argument("--num_jets", "-n", type=int, default=100_000)
parser.add_argument("--vocab_size", type=int, default=8)
parser.add_argument("--max_seq_length", "-len", type=int, default=150)
parser.add_argument("--batch_size", "-bs", type=int, default=256)
parser.add_argument("--tag", "-t", type=str, default='')


config = parser.parse_args()

###############################################################################

gpt = JetFlavorSeqGPT.load_from_checkpoint(f"/home/df630/Multimodal-flows/jet_sequences/{config.experiment_id}/checkpoints/best.ckpt", map_location="cpu",)

#...dataset & dataloaders:

aoj = AspenOpenJets(data_dir="/home/df630/Multimodal-Bridges/data/aoj", data_files="RunG_batch1.h5")

jets, _ = aoj(num_jets=config.num_jets,
              max_num_particles=config.max_seq_length,
              download=False,
              features={"continuous": None, "discrete": "tokens"},
              pt_order=True,
              padding='zeros')

prompts = torch.full((config.num_jets, 1), gpt.start_token, dtype=torch.long)
attention_mask = torch.ones_like(prompts)

source = TensorMultiModal(discrete=prompts, mask=attention_mask)
data = DataCoupling(source=source, target=TensorMultiModal())
dataset = MultiModalDataset(data)
predict_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=data_coupling_collate_fn)

#...sample from MJB model

gen_callback = GPTGeneratorCallback(config)

generator = L.Trainer(accelerator="gpu", 
                      devices=[0], 
                      callbacks=[gen_callback],
                      )

generator.predict(gpt, dataloaders=predict_dataloader)