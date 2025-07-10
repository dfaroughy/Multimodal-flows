import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.nn import functional as F
from typing import List, Tuple, Dict, Union

from torch.optim.lr_scheduler import CosineAnnealingLR

from tensorclass import TensorMultiModal
from datamodules.datasets import DataCoupling
from transformers import GPT2Config

from model.solvers import ContinuousSolver
from model.thermostats import Thermostat
from networks.JetFlavorGPT import FlavorFormer

class ConditionalFlowMatching(L.LightningModule):
    def __init__(self, config):
                 
        super().__init__()

        self.sigma=config.sigma
        self.vocab_size=config.vocab_size
        self.num_jets=config.num_jets
        self.max_num_particles=config.max_num_particles
        self.lr_final=config.lr_final
        self.lr=config.lr
        self.max_epochs=config.max_epochs
        self.time_eps=config.time_eps
        self.num_timesteps = config.num_timesteps if hasattr(config, 'num_timesteps') else 10
        self.path_snapshots_idx = False

        self.bridge_continuous = UniformFlow(self.sigma)        
        self.model = FlavorFormer(config)

        self.save_hyperparameters()

    # ...Lightning functions

    def forward(self, state: TensorMultiModal) -> TensorMultiModal:
        return self.model(state.continuous, state.time.squeeze(-1))
        
    def training_step(self, batch: DataCoupling, batch_idx):

        state = self.sample_bridges(batch)
        state = state.to(self.device)
        drift = self.model(xt=state.continuous,
                           time=state.time.squeeze(-1),
                           )
        loss = self.loss(drift, batch)

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

        state = self.sample_bridges(batch)
        state = state.to(self.device)
        drift = self.model(xt=state.continuous,
                           time=state.time.squeeze(-1),
                           )
        loss = self.loss(drift, batch)
        
        self.log("val_loss",
                 loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=len(batch)
                 )

        return {"val_loss": loss}

    def predict_step(self, batch: DataCoupling, batch_idx):

        """generate target data from source by solving EOMs
        """
        paths = self.simulate_dynamics(batch.source)  # still preprocessed!

        return paths.detach().cpu()

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,    # full cycle length
            eta_min=self.lr_final             # final LR
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

    def sample_bridges(self, batch: DataCoupling) -> TensorMultiModal:

        """sample stochastic bridges
        """

        eps = self.time_eps  # min time resolution
        t = eps + (1 - eps) * torch.rand(len(batch), device=self.device)
        time = self.reshape_time_dim_like(t, batch)

        source_data = torch.randn((len(batch), self.max_num_particles, self.vocab_size), device=self.device)
        mask = torch.ones((len(batch), 
                           self.max_num_particles, 1), 
                           device=self.device).long()

        batch.source = TensorMultiModal(continuous=source_data, mask=mask)
        noisy_data = self.bridge_continuous.sample(time, batch)

        return TensorMultiModal(time=time, continuous=noisy_data, mask=mask)


    def loss(self, drift, batch: DataCoupling) -> torch.Tensor:

        """MSE loss for flow-matching
        """

        targets = batch.target.continuous.to(self.device)
        loss_mse = F.mse_loss(drift, targets, reduction='mean')
        return loss_mse


    def simulate_dynamics(self, state: TensorMultiModal) -> TensorMultiModal:

        """generate target data from source input using trained dynamics
        returns the final state of the bridge at the end of the time interval
        """

        eps = self.time_eps  # min time resolution
        state.time = torch.full((len(state), 1), eps, device=self.device)  # (B,1) t_0=eps
        state.broadcast_time() # (B,1) -> (B,D,1)
        steps = self.num_timesteps
        time_steps = torch.linspace(eps, 1.0 - eps, steps, device=self.device)
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)

        solver = ContinuousSolver(model=self, method='euler',)
        paths = [state.clone()]  # append t=0 source

        for i, t in enumerate(time_steps):
            is_last_step = (i == len(time_steps) - 1)

            state.time = torch.full((len(state), 1), t.item(), device=self.device)            
            state = solver.fwd_step(state, delta_t)
            state.broadcast_time() 
                        
            if isinstance(self.path_snapshots_idx, list):
                for i in self.path_history_idx:
                    paths.append(state.clone())

        paths.append(state)  # append t=1 generated target
        paths = TensorMultiModal.stack(paths, dim=0)
        return paths

    def reshape_time_dim_like(self, t, state: Union[TensorMultiModal, DataCoupling]):

        if isinstance(t, (float, int)):
            return t
        else:
            return t.reshape(-1, *([1] * (state.ndim - 1)))



class UniformFlow:
    """Conditional OT Flow-Matching for continuous states.
    This bridge is a linear interpolation between boundaries states at t=0 and t=1.
    notation:
      - t: time
      - x0: continuous source state at t=0
      - x1: continuous target state at t=1
      - x: continuous state at time t
      - z: delta function regularizer
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, t, batch: DataCoupling):
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        xt = t * x1 + (1.0 - t) * x0
        z = torch.randn_like(xt)
        std = self.sigma
        return xt + std * z

    def drift(self, state: TensorMultiModal, batch: DataCoupling):
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        xt = state.continuous
        A = 0.0
        B = 1.0
        C = -1.0
        return A * xt + B * x1 + C * x0

    def diffusion(self, state: TensorMultiModal):
        return 0.0