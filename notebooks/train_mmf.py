import torch
import numpy as np
import yaml
import os
import pytorch_lightning as L
import matplotlib.pyplot as plt
import argparse
from argparse import ArgumentParser
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader, random_split
from utils.tensorclass import TensorMultiModal
from utils.aoj import AspenOpenJets 
from utils.helpers import load_from_experiment, set_logger
from utils.datasets import MultiModalDataset, DataCoupling, data_coupling_collate_fn
                            


def experiment_configs(exp_id_path=None):
    if exp_id_path is None:
        config = ArgumentParser()

        config.add_argument("--num_nodes", "-N", type=int, default=1)
        config.add_argument("--dir", type=str, default='/home/df630/Multimodal-flows')
        config.add_argument("--project", "-proj", type=str, default='jet_sequences')
        config.add_argument("--comet_workspace", type=str, default='dfaroughy')
        config.add_argument("--comet_api_key", type=str, default='8ONjCXJ1ogsqG1UxQzKxYn7tz')
        config.add_argument("--experiment_id", "-id", type=str, default=None)
        config.add_argument("--ckpt_path", "-ckpt", type=str, default=None)
        config.add_argument("--tags", type=str, nargs='*')

        config.add_argument("--data_files", "-f", type=str, default='RunG_batch0.h5')
        config.add_argument("--num_jets", "-n", type=int, default=100_000)
        config.add_argument("--max_num_particles", "-d", type=int, default=150)
        config.add_argument("--dim_continuous", type=int, default=3)
        config.add_argument("--vocab_size", "-v", type=int, default=9)
        config.add_argument("--batch_size", "-bs", type=int, default=256)
        config.add_argument("--max_epochs", "-epochs", type=int, default=250)
        config.add_argument("--train_frac", type=float, default=0.8)

        config.add_argument("--model", "-nn", type=str, default='MultiModalParticleFormer')
        config.add_argument("--n_embd", type=int, default=256)
        config.add_argument("--n_inner", type=int, default=1024)
        config.add_argument("--n_layer", type=int, default=2)
        config.add_argument("--n_head", type=int, default=2)
        config.add_argument("--init_gate_val", '-gate', type=float, default=0.9)
        config.add_argument("--dropout", type=float, default=0.0)
        config.add_argument("--qk_layernorm", type=bool, default=True)
        config.add_argument("--bias", type=bool, default=True)
        config.add_argument("--lr", type=float, default=1e-4)
        config.add_argument("--lr_final", type=float, default=1e-5)
        config.add_argument("--warmup_epochs", type=str, default=5)
        
        config.add_argument("--gamma", "-gam", type=float, default=0.1)
        config.add_argument("--sigma", "-sig", type=float, default=1e-5)
        config.add_argument("--loss_weight", type=float, default=1.0)        
        config.add_argument("--time_eps", "-eps", type=float, default=1e-5)
        config.add_argument("--num_timesteps", "-steps", type=int, default=1000)
        config.add_argument("--temperature", type=float, default=1.0)

        return config.parse_args()
    else:
        return load_from_experiment(exp_id_path)


def make_dataloaders(config):

    aoj = AspenOpenJets(data_dir="/home/df630/Multimodal-Bridges/data/aoj", data_files=config.data_files)

    jets, metadata = aoj(num_jets=config.num_jets,
                download=False,
                features={"continuous": ['pt', 'eta_rel', 'phi_rel'], "discrete": "tokens"},
                pt_order=True,
                transform='standardize',
                padding='zeros')

    config.metadata = metadata

    
    gauss_noise = torch.randn_like(jets.continuous) * jets.mask
    cat_noise = torch.randint_like(jets.discrete, 1, config.vocab_size) * jets.mask
    noise = TensorMultiModal(continuous=gauss_noise, discrete=cat_noise, mask=jets.mask.clone())

    data = DataCoupling(source=noise, target=jets)
    dataset = MultiModalDataset(data)
    train_size = int(config.train_frac * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=data_coupling_collate_fn)
    val_dataloader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False, collate_fn=data_coupling_collate_fn)
 
    return train_dataloader, val_dataloader

    
def run_train_experiment(config, lighting_module):

    train, val = make_dataloaders(config)

    if config.experiment_id is not None: # resume training
        config.ckpt_path = f"{config.dir}/{config.project}/{config.experiment_id}/checkpoints/last.ckpt"
        model = lighting_module.load_from_checkpoint(config.ckpt_path, map_location="cpu",)

    else: # new run
        model = lighting_module(config)

    callback = L.callbacks.ModelCheckpoint(dirpath=None,
                                           monitor="val_loss",
                                           filename="best",
                                           save_top_k=10,
                                           mode="min",
                                           save_last=True,
                                            )

    logger = set_logger(config)

    trainer = L.Trainer(max_epochs=config.max_epochs, 
                        accelerator='gpu', 
                        devices='auto',
                        strategy='ddp',
                        num_nodes=config.num_nodes,
                        callbacks=[callback],
                        logger=logger,
                        sync_batchnorm=True,
                        gradient_clip_val=1.0,
                        )
    
    trainer.fit(model, train, val, ckpt_path=config.ckpt_path)


if __name__ == "__main__":

    from model.MMF import MultiModalFlowBridge

    config = experiment_configs()
    run_train_experiment(config, lighting_module=MultiModalFlowBridge)



