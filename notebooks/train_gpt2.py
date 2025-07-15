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
from model.GPT import JetFlavorSeqGPT

###############################################################################

parser = ArgumentParser()

parser.add_argument("--num_nodes", "-N", type=int, default=1)
parser.add_argument("--dir", type=str, default='/home/df630/Multimodal-flows')
parser.add_argument("--project", "-proj", type=str, default='jet_sequences')
parser.add_argument("--comet_workspace", type=str, default='dfaroughy')
parser.add_argument("--comet_api_key", type=str, default='8ONjCXJ1ogsqG1UxQzKxYn7tz')
parser.add_argument("--experiment_id", "-id", type=str, default=None)
parser.add_argument("--tags", type=str, nargs='*')
parser.add_argument("--num_jets", "-n", type=int, default=100_000)
parser.add_argument("--vocab_size", type=int, default=8)
parser.add_argument("--max_seq_length", "-len", type=int, default=150)
parser.add_argument("--batch_size", "-bs", type=int, default=256)
parser.add_argument("--max_epochs", "-epochs", type=int, default=250)
parser.add_argument("--train_frac", type=float, default=0.8)
parser.add_argument("--n_embd", type=int, default=256)
parser.add_argument("--n_inner", type=int, default=256)
parser.add_argument("--n_layer", type=int, default=8)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--activation", "-a", type=str, default='gelu_new')
parser.add_argument("--dropout_att", "-do_att", type=float, default=0.1)
parser.add_argument("--dropout_emb", "-do_emb",type=float, default=0.1)
parser.add_argument("--dropout_res", "-do_res", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=None)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_final", type=float, default=0.0001)

config = parser.parse_args()

###############################################################################

logger = CometLogger(api_key=config.comet_api_key,
                     project=config.project,
                     workspace=config.comet_workspace,
                     offline_directory=config.dir,
                     experiment_key=config.experiment_id if config.experiment_id else None
                     )
                     
logger.experiment.add_tags(config.tags)
logger.experiment.log_parameters(vars(config))                     
                    

#...dataset & dataloaders:
aoj = AspenOpenJets(data_dir="/home/df630/Multimodal-Bridges/data/aoj", data_files="RunG_batch0.h5")

jets, _ = aoj(num_jets=config.num_jets,
              max_num_particles=config.max_seq_length,
              download=False,
              features={"continuous": None, "discrete": "tokens"},
              pt_order=True,
              padding='zeros')

jet_seq = jet_set_to_seq(jets, vocab_size=8)

data = DataCoupling(source=TensorMultiModal(), target=jet_seq)
dataset = MultiModalDataset(data)
train_size = int(config.train_frac * len(dataset))
val_size   = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=data_coupling_collate_fn)
val_dataloader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False, collate_fn=data_coupling_collate_fn)

#...train

gpt = JetFlavorSeqGPT(config)
                    
trainer = L.Trainer(max_epochs=config.max_epochs, 
                    accelerator='gpu', 
                    devices='auto',
                    strategy='ddp',
                    num_nodes=config.num_nodes,
                    callbacks=[L.callbacks.ModelCheckpoint(dirpath=None,
                                                           monitor="val_loss",
                                                           filename="best",
                                                           save_top_k=1,
                                                           mode="min",
                                                           save_last=True,
                                                            )],
                    logger=logger,
                    sync_batchnorm=True,
                    gradient_clip_val=1.0,
                    )

trainer.fit(gpt, train_dataloader, val_dataloader)
