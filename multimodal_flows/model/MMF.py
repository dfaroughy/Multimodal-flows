import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.nn import functional as F
from typing import List, Tuple, Dict, Union
from torch.nn.functional import softmax
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.CFM import UniformFlow
from models.MJB import RandomTelegraphBridge

from utils.tensorclass import TensorMultiModal
from utils.datasets import DataCoupling
from utils.thermostats import ConstantThermostat
from model.solvers import DiscreteSolver, ContinuousSolver
from networks.ParticleTransformers import ParticleFormer


class MultiModalFlowBridge(L.LightningModule):
    def __init__(self, config):
                 
        super().__init__()

        self.dim_continuous=config.dim_continuous
        self.vocab_size=config.vocab_size
        self.num_jets=config.num_jets
        self.max_num_particles=config.max_num_particles

        self.max_epochs=config.max_epochs
        self.time_eps=config.time_eps
        self.temperature = config.temperature
        self.num_timesteps = config.num_timesteps
        self.lr_final=config.lr_final
        self.lr=config.lr

        self.gamma=config.gamma
        self.sigma=config.sigma

        thermostat = ConstantThermostat(self.gamma, self.vocab_size)
        
        self.bridge_continuous = UniformFlow(self.sigma)        
        self.bridge_discrete = RandomTelegraphBridge(self.gamma, self.vocab_size, thermostat, self.temperature)        
        self.model = ParticleFormer(config)
        
        self.save_hyperparameters()

    # ...Lightning functions

    def forward(self, state: TensorMultiModal) -> (torch.Tensor, torch.Tensor):
        return self.model(state)
        
    def training_step(self, batch: DataCoupling, batch_idx):

        loss = self.loss(batch)

        self.log("train_loss",
                 loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=len(batch)
                 )

        return {"loss": loss}
        
    def validation_step(self, batch: DataCoupling, batch_idx):

        loss = self.loss(batch)

        self.log("val_loss",
                 loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=len(batch)
                 )

        return {"val_loss": loss}

    def predict_step(self, batch: DataCoupling, batch_idx, dataloader_idx=0) -> TensorMultiModal:
        ''' sample generation
        '''
        traj = self.simulate_dynamics(batch)
        sample = traj.target
        return sample.detach().cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,  # full cycle length
            eta_min=self.lr_final,  # final LR
            last_epoch=-1,         
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   
                "frequency": 1,
                "strict": True,
            },
        }

    # ...Model functions

    def loss(self, batch: DataCoupling) -> TensorMultiModal:

        """ Multi-modal flow MSE + CE loss
        """

        # sample time uniformly in [eps, 1-eps], eps << 1

        time = self.time_eps  + (1. - self.time_eps ) * torch.rand(len(batch), device=self.device)  # (B,)

        # sample continuous and discrete states from hybrid bridge

        state = self.bridge_continuous.sample(time, batch)
        state = self.bridge_discrete.sample(time, batch)
        state = state.to(self.device)

        # compute loss

        vt, logits = self.model(state)     

        targets_continuous = self.bridge_continuous.conditional_drift(state, batch)
        loss_mse =  F.mse_loss(vt, targets_continuous, reduction='none')
        loss_mse = loss_mse * batch.target.mask    

        targets_discrete = batch.target.discrete.to(self.device)        
        loss_ce =  F.cross_entropy(logits.view(-1, self.vocab_size), targets_discrete.view(-1), reduction='none')    # (B*D,)
        loss_ce = loss_ce.view(len(batch), -1) * state.mask    # (B, D)
        
        loss = loss_ce + loss_mse
        loss = loss.sum() / batch.target.mask.sum()

        return loss

    def simulate_dynamics(self, batch: DataCoupling) -> DataCoupling:

        """generate target data from source input using trained dynamics
        returns the final state of the bridge at the end of the time interval
        """
        
        solver_ode = ContinuousSolver(model=self, method='euler')
        solver_jumps = DiscreteSolver(model=self, vocab_size=self.vocab_size, method='tauleap', temperature=self.temperature)
        
        time_steps = torch.linspace(self.time_eps, 1.0 - self.time_eps, self.num_timesteps, device=self.device)
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)

        state = batch.source.clone()
        
        for i, t in enumerate(time_steps):
            is_last_step = (i == len(time_steps) - 1)
            state.time = torch.full((len(state),), t.item(), device=self.device)  
            state = solver_ode.fwd_step(state, delta_t)         
            state = solver_jumps.fwd_step(state, delta_t, is_last_step)

        batch.target = state

        return batch


