import numpy as np
import torch
from argparse import ArgumentParser
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning as L
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split


from model.MMF import MultiModalFlowBridge
from utils.tensorclass import TensorMultiModal
from utils.aoj import AspenOpenJets 
from utils.datasets import (MultiModalDataset, 
                            DataCoupling, 
                            data_coupling_collate_fn, 
                            standardize)


def experiment_configs():

    config = ArgumentParser()

    config.add_argument("--num_nodes", "-N", type=int, default=1)
    config.add_argument("--dir", type=str, default='/home/df630/Multimodal-flows')
    config.add_argument("--project", "-proj", type=str, default='jet_sequences')
    config.add_argument("--comet_workspace", type=str, default='dfaroughy')
    config.add_argument("--comet_api_key", type=str, default='8ONjCXJ1ogsqG1UxQzKxYn7tz')
    config.add_argument("--experiment_id", "-id", type=str, default=None)
    config.add_argument("--tags", type=str, nargs='*')

    config.add_argument("--data_files", "-f", type=str, default='RunG_batch0.h5')
    config.add_argument("--num_jets", "-n", type=int, default=100_000)
    config.add_argument("--max_num_particles", "-d", type=int, default=150)
    config.add_argument("--batch_size", "-bs", type=int, default=256)
    config.add_argument("--max_epochs", "-epochs", type=int, default=250)
    config.add_argument("--train_frac", type=float, default=0.8)

    config.add_argument("--vocab_size", type=int, default=9)
    config.add_argument("--dim_continuous", type=int, default=3)
    config.add_argument("--n_embd", type=int, default=256)
    config.add_argument("--n_inner", type=int, default=1024)
    config.add_argument("--n_layer", type=int, default=2)
    config.add_argument("--n_layer_fused", type=int, default=2)
    config.add_argument("--n_head", type=int, default=2)
    config.add_argument("--dropout", type=float, default=0.0)
    config.add_argument("--qk_layernorm", type=bool, default=True)
    config.add_argument("--bias", type=bool, default=True)
    config.add_argument("--lr", type=float, default=1e-4)
    config.add_argument("--lr_final", type=float, default=1e-5)
    config.add_argument("--gamma", "-gam", type=float, default=0.1)
    config.add_argument("--sigma", "-sig", type=float, default=1e-5)
    config.add_argument("--loss_weight", type=float, default=1.0)

    config.add_argument("--time_eps", "-eps", type=float, default=1e-5)
    config.add_argument("--num_timesteps", "-steps", type=int, default=1000)
    config.add_argument("--temperature", type=float, default=1.0)

    return config.parse_args()


def _set_logger(config):

    logger = CometLogger(api_key=config.comet_api_key,
                        project=config.project,
                        workspace=config.comet_workspace,
                        offline_directory=config.dir,
                        experiment_key=config.experiment_id if config.experiment_id else None
                        )
                        
    logger.experiment.add_tags(config.tags)
    logger.experiment.log_parameters(vars(config))
    return logger


def _make_dataloaders(config):

    aoj = AspenOpenJets(data_dir="/home/df630/Multimodal-Bridges/data/aoj", data_files=config.data_files)

    jets, metadata = aoj(num_jets=config.num_jets,
                download=False,
                features={"continuous": ['pt', 'eta_rel', 'phi_rel'], "discrete": "tokens"},
                pt_order=True,
                padding='zeros')

    config.metadata = metadata
    jets.continuous = (jets.continuous - metadata['mean']) / metadata['std'] 
    jets.apply_mask()
    
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


def run_experiment(config):

    logger = _set_logger(config)

    mmf = MultiModalFlowBridge(config)
    
    callback = L.callbacks.ModelCheckpoint(dirpath=None,
                                            monitor="val_loss",
                                            filename="best",
                                            save_top_k=1,
                                            mode="min",
                                            save_last=True,
                                            )

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

    train, val = _make_dataloaders(config)
    trainer.fit(mmf, train, val)


if __name__ == "__main__":

    config = experiment_configs()
    run_experiment(config)

