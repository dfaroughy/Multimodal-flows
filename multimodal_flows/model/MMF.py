import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.nn import functional as F
from typing import List, Tuple, Dict, Union
from torch.nn.functional import softmax
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.CFM import UniformFlow
from model.MJB import RandomTelegraphBridge

from utils.tensorclass import TensorMultiModal
from utils.datasets import DataCoupling
from utils.thermostats import ConstantThermostat
from model.solvers import HybridSolver
from networks.registry import MODEL_REGISTRY


class MultiModalFlowBridge(L.LightningModule):
    def __init__(self, config):

        """ Hybrid Dynamical generative model for continuous and discrete states
            based on continuous-time Markov jump processes and flow matching.
        """                 
        super().__init__()

        thermostat = ConstantThermostat(config.gamma, config.vocab_size)

        self.model = MODEL_REGISTRY[config.model](config)
        self.bridge_continuous = UniformFlow(config.sigma)        
        self.bridge_discrete = RandomTelegraphBridge(config.gamma, config.vocab_size, thermostat)   
        self.save_hyperparameters(vars(config))
        self.config = config

    # ...Lightning functions

    def forward(self, state: TensorMultiModal) -> (torch.Tensor, torch.Tensor):
        return self.model(state)
        
    def training_step(self, batch: DataCoupling, batch_idx):

        loss_mse, loss_ce = self.loss(batch)
        loss = loss_mse + self.config.loss_weight * loss_ce

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
        self.log("train_loss_ce", loss_ce, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
        self.log("train_loss_mse", loss_mse, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))

        return {"loss": loss}

    def validation_step(self, batch: DataCoupling, batch_idx):

        loss_mse, loss_ce = self.loss(batch)
        loss = loss_mse + self.config.loss_weight * loss_ce

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
        self.log("val_loss_ce", loss_ce, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
        self.log("val_loss_mse", loss_mse, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))

        return {"val_loss": loss}

    def predict_step(self, batch: DataCoupling, batch_idx, dataloader_idx=0) -> TensorMultiModal:
        ''' sample generation
        '''
        traj = self.simulate_dynamics(batch)
        sample = traj.target
        return sample.detach().cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

        cosine_epochs = max(self.config.max_epochs - self.config.warmup_epochs, 1)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=self.config.lr_final,
            last_epoch=-1
        )

        # linear warmup over the first `warmup_epochs` 
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # step per epoch
                "frequency": 1,
                "strict": True,
            },
        }


    # ...Model functions

    def loss(self, batch: DataCoupling) -> TensorMultiModal:

        """ Multi-modal flow MSE + CE loss
        """
        eps = self.config.time_eps
        time = eps  + (1. - eps ) * torch.rand(len(batch), device=self.device)

        # sample continuous and discrete states from hybrid bridge

        xt = self.bridge_continuous.sample(time, batch)
        kt = self.bridge_discrete.sample(time, batch)

        mutlimodal_state = TensorMultiModal(continuous=xt, discrete=kt, mask=batch.target.mask, time=time)
        mutlimodal_state = mutlimodal_state.to(self.device)

        # compute loss

        vt, logits = self.model(mutlimodal_state)     # (B, D, dim_continuous), # (B, D, vocab_size)

        targets_continuous = self.bridge_continuous.conditional_drift(mutlimodal_state, batch)
        loss_mse =  F.mse_loss(vt, targets_continuous, reduction='none')
        loss_mse = loss_mse * mutlimodal_state.mask     # (B, D, dim_continuous)
        loss_mse = loss_mse.sum() / mutlimodal_state.mask.sum()

        targets_discrete = batch.target.discrete.to(self.device)        
        loss_ce =  F.cross_entropy(logits.view(-1, self.config.vocab_size), targets_discrete.view(-1), ignore_index=0, reduction='none')    # (B*D,)
        loss_ce = loss_ce.view(len(batch), -1) * mutlimodal_state.mask.squeeze(-1)    # (B, D)
        loss_ce = loss_ce.sum() / mutlimodal_state.mask.sum()

        return loss_mse, loss_ce

    @torch.no_grad()
    def simulate_dynamics(self, batch: DataCoupling) -> DataCoupling:

        self.model.eval()

        """generate target data from source input using trained dynamics
        returns the final state of the bridge at the end of the time interval
        """

        eps = self.config.time_eps
        steps = self.config.num_timesteps
        time_steps = torch.linspace(eps, 1.0 - eps, steps, device=self.device)
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)

        solver = HybridSolver(model=self, config=self.config)
        state = batch.source.clone()
        
        for i, t in enumerate(time_steps):
            state.time = torch.full((len(state),), t.item(), device=self.device)  
            state = solver.fwd_step(state, delta_t)

        batch.target = state

        return batch


