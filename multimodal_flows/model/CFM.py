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
from networks.ParticleTransformers import FlavorFormer

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
        self.mean = config.mean
        self.std = config.std
        self.path_snapshots_idx = False

        self.bridge_continuous = UniformFlow(self.sigma)        
        self.model = FlavorFormer(config)

        self.save_hyperparameters()

    # ...Lightning functions

    def forward(self, state: TensorMultiModal) -> TensorMultiModal:
        return self.model(state)
        
    def training_step(self, batch: DataCoupling, batch_idx):

        state = self.sample_bridges(batch)
        state = state.to(self.device)
        vt = self.model(state)
        ut = self.bridge_continuous.conditional_drift(state, batch)
        loss = F.mse_loss(vt, ut, reduction='mean')

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
        vt = self.model(state)
        ut = self.bridge_continuous.conditional_drift(state, batch)
        loss = F.mse_loss(vt, ut, reduction='mean')
        
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
        state = self.bridge_continuous.sample(time, batch)

        return state

    def simulate_dynamics(self, batch: DataCoupling) -> DataCoupling:

        """generate target data from source input using trained dynamics
        returns the final state of the bridge at the end of the time interval
        """
        solver = ContinuousSolver(model=self, method='euler',)
        time_steps = torch.linspace(self.time_eps, 1.0 - self.time_eps, self.num_timesteps, device=self.device)
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)

        state = batch.source.clone()
        state.time = torch.full((len(state), 1), self.time_eps, device=self.device)  # (B,1) t_0=eps
        state.broadcast_time() # (B,1) -> (B,D,1)
        
        for i, t in enumerate(time_steps):
            state.time = torch.full((len(state), 1), t.item(), device=self.device)            
            state = solver.fwd_step(state, delta_t)
            state.broadcast_time() 

        batch.target = state

        return batch

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

    def sample(self, time, batch: DataCoupling) -> TensorMultiModal:
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        xt = time * x1 + (1.0 - time) * x0
        z = torch.randn_like(xt)
        std = self.sigma
        state = xt + std * z
        return TensorMultiModal(time=time, continuous=state, mask=batch.target.mask)

    def conditional_drift(self, state: TensorMultiModal, batch: DataCoupling) -> torch.Tensor:
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        xt = state.continuous
        A = 0.0
        B = 1.0
        C = -1.0
        return A * xt + B * x1 + C * x0

    def diffusion(self, state: TensorMultiModal):
        return 0.0