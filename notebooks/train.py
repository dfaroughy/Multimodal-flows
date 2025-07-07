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
num_jets = 200_000
batch_size = 256
max_epochs = 200
logger = CometLogger(api_key='8ONjCXJ1ogsqG1UxQzKxYn7tz', 
                     project_name='jet_sequences',
                     workspace='dfaroughy', 
                     save_dir='/home/df630/Multimodal-flows/results')
###############################################################################

aoj = AspenOpenJets(data_dir="/home/df630/Multimodal-Bridges/data/aoj", data_files="RunG_batch0.h5")

jets, _ = aoj(num_jets=num_jets,
              download=False,
              features={"continuous": None, "discrete": "tokens"},
              pt_order=True,
              padding='zeros',
            )

jets.mask = torch.ones_like(jets.discrete)

data = DataCoupling(source=TensorMultiModal(), target=jets)
dataset = MultiModalDataset(data)

train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_coupling_collate_fn)
val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=data_coupling_collate_fn)

mjb = MarkovJumpBridge(gamma=0.075, 
                       vocab_size=9,
                       num_jets=num_jets,
                       max_num_particles=150,
                       lr_final=0.0001,
                       lr=0.001,
                       max_epochs=max_epochs,
                       time_eps=1e-5,
                       n_embd=256,
                       n_layer=4,
                       n_head=4,
                       activation_function='gelu_new',
                       )

trainer = L.Trainer(max_epochs=max_epochs, 
                    accelerator='gpu', 
                    devices=[0,1,2,3], 
                    strategy='ddp',
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

trainer.fit(mjb, train_dataloader, val_dataloader)
