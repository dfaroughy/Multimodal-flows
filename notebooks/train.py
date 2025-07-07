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

parser.add_argument("--num_nodes", "-N", type=int, default=1)
parser.add_argument("--dir", type=str, default='/pscratch/sd/d/dfarough')
parser.add_argument("--project_name", "-proj", type=str, default='jet_sequences')
parser.add_argument("--comet_workspace", type=str, default='dfaroughy')
parser.add_argument("--comet_api_key", type=str, default='8ONjCXJ1ogsqG1UxQzKxYn7tz')
parser.add_argument("--data_path", type=str, default='/pscratch/sd/d/dfarough/JetClass')
parser.add_argument("--experiment_id", "-id", type=str, default=None)
parser.add_argument("--checkpoint", "-ckpt", type=str, default='last')
parser.add_argument("--tags", type=str, nargs='*')

parser.add_argument("--checkpoint", "-ckpt", type=str, default='last.ckpt')
parser.add_argument("--num_jets", "-n", type=int, default=2_500_000)
parser.add_argument("--max_num_particles", type=int, default=150)
parser.add_argument("--batch_size", "-bs", type=int, default=1024)
parser.add_argument("--max_epochs", "-epochs", type=int, default=100)
parser.add_argument("--train_frac", type=float, default=0.8)

parser.add_argument("--vocab_size", type=int, default=9)
parser.add_argument("--gamma ", type=float, default=0.075)
parser.add_argument("--time_eps ", type=float, default=1e-5)
parser.add_argument("--n_embd ", type=int, default=256)
parser.add_argument("--n_layer ", type=int, default=8)
parser.add_argument("--n_head ", type=int, default=4)
parser.add_argument("--activation_function", type=str, default='new_gelu')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_final", type=float, default=1e-4)


config = parser.parse_args()



###############################################################################

logger = CometLogger(
            api_key=config.comet_api_key,
            project_name=config.project_name,
            workspace=config.comet_workspace,
            save_dir=config.dir,
            experiment_key=config.experiment_id if config.experiment_id else None
        )

#...dataset & dataloaders:

aoj = AspenOpenJets(data_dir=config.dir + '/aoj', data_files="RunG_batch0.h5")

jets, _ = aoj(num_jets=config.num_jets,
              download=False,
              features={"continuous": None, "discrete": "tokens"},
              pt_order=True,
              padding='zeros',
            )

jets.mask = torch.ones_like(jets.discrete)
data = DataCoupling(source=TensorMultiModal(), target=jets)
dataset = MultiModalDataset(data)
train_size = int(config.train_frac * len(dataset))
val_size   = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=data_coupling_collate_fn)
val_dataloader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False, collate_fn=data_coupling_collate_fn)

#...train

mjb = MarkovJumpBridge(gamma=config.gamma, 
                       vocab_size=config.vocab_size,
                       num_jets=config.num_jets,
                       max_num_particles=config.max_num_particles,
                       lr_final=config.lr_final,
                       lr=config.lr,
                       max_epochs=config.max_epochs,
                       time_eps=config.time_eps,
                       n_embd=config.n_embd,
                       n_layer=config.n_layer,
                       n_head=config.n_head,
                       activation_function=config.activation_function,
                       )


trainer = L.Trainer(max_epochs=config.max_epochs, 
                    accelerator='gpu', 
                    devices='auto',
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
