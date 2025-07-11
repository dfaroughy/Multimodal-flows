import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.nn import functional as F
from typing import List, Tuple, Dict, Union
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.tensorclass import TensorMultiModal
from utils.datasets import DataCoupling
from model.solvers import ContinuousSolver
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
        self.mean = config.mean if hasattr(config, 'mean') else 0.0
        self.std = config.std if hasattr(config, 'std') else 1.0
        self.path_snapshots_idx = False

        self.bridge_continuous = UniformFlow(self.sigma)        
        self.model = FlavorFormer(config)

        self.save_hyperparameters()

    # ...Lightning functions

    def forward(self, state: TensorMultiModal) -> TensorMultiModal:
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

    def loss(self, batch: DataCoupling) -> TensorMultiModal:

        """ Flow-matching MSE loss
        """

        # sample time uniformly in [eps, 1-eps], eps << 1
        time = self.time_eps  + (1. - self.time_eps ) * torch.rand(len(batch), device=self.device)

        # sample continuous state from bridge
        state = self.bridge_continuous.sample(time, batch)
        state = state.to(self.device)

        # compute loss
        vt = self.model(state)
        targets = self.bridge_continuous.conditional_drift(state, batch)

        return F.mse_loss(vt, targets, reduction='mean')

    def simulate_dynamics(self, batch: DataCoupling) -> DataCoupling:

        """generate target data from source input using trained dynamics
        returns the final state of the bridge at the end of the time interval
        """
        solver = ContinuousSolver(model=self, method='euler',)
        time_steps = torch.linspace(self.time_eps, 1.0 - self.time_eps, self.num_timesteps, device=self.device)
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)

        state = batch.source.clone()
        
        for i, t in enumerate(time_steps):
            state.time = torch.full((len(state),), t.item(), device=self.device)            
            state = solver.fwd_step(state, delta_t)
            state.broadcast_time() 

        batch.target = state

        return batch


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

    def sample(self, time: torch.Tensor, batch: DataCoupling) -> TensorMultiModal:

        t = self.reshape_time_dim_like(time, batch) # (B,) -> (B, 1, 1)
        x0 = batch.source.continuous  # (B, D, vocab_size)
        x1 = batch.target.continuous  # (B, D, vocab_size)

        xt = t * x1 + (1.0 - t) * x0   # time-interpolated state
        z = torch.randn_like(xt)       # noise
        state = xt + self.sigma * z    # Dirac -> Gauss smear

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

    def reshape_time_dim_like(self, t, state: Union[TensorMultiModal, DataCoupling]):

        if isinstance(t, (float, int)):
            return t
        else:
            return t.reshape(-1, *([1] * (state.ndim - 1)))


class SchrodingerBridge:
    """ Schrodinger bridge for continuous states
        notation:
        - t: time
        - x0: continuous source state at t=0
        - x1: continuous  target state at t=1
        - x: continuous state at time t
        - z: noise
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, t, batch: DataCoupling):
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        x = t * x1 + (1.0 - t) * x0
        z = torch.randn_like(x)
        std = self.sigma * torch.sqrt(t * (1.0 - t))
        return x + std * z

    def drift(self, state: TensorMultiModal, batch: DataCoupling):
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        xt = state.continuous
        t = state.time
        A = (1 - 2 * t) / (t * (1 - t))
        B = t**2 / (t * (1 - t))
        C = -1 * (1 - t) ** 2 / (t * (1 - t))
        return A * xt + B * x1 + C * x0

    def diffusion(self, state: TensorMultiModal):
        state.broadcast_time()
        t = state.time
        return self.sigma * torch.sqrt(t * (1.0 - t))