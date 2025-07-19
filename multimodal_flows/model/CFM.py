import torch
import torch.nn as nn
import pytorch_lightning as L

from torch.nn import functional as F
from typing import List, Tuple, Dict, Union
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.tensorclass import TensorMultiModal
from utils.datasets import DataCoupling
from model.solvers import ContinuousSolver
from networks.registry import MODEL_REGISTRY

class ConditionalFlowMatching(L.LightningModule):

    """ Dynamical generative model for continuous states
        ODE-based Conditional Flow Matching.
    """

    def __init__(self, config):
                 
        super().__init__()

        self.model = MODEL_REGISTRY[config.model](config)
        self.bridge_continuous = UniformFlow(config.sigma)        
        self.save_hyperparameters(vars(config))
        self.config = config

    # ...Lightning functions

    def forward(self, state: TensorMultiModal) -> torch.Tensor:
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_epochs,    # full cycle length
            eta_min=self.config.lr_final,    # final LR
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

        """ Flow-matching MSE loss
        """
        eps = self.config.time_eps

        # sample time uniformly in [eps, 1-eps], eps << 1
        time = eps  + (1. - eps ) * torch.rand(len(batch), device=self.device)

        # sample continuous state from bridge
        xt = self.bridge_continuous.sample(time, batch)
        state = TensorMultiModal(time=time, continuous=xt, mask=batch.target.mask)
        state = state.to(self.device)

        # compute loss
        vt = self.model(state)
        targets = self.bridge_continuous.conditional_drift(state, batch)

        loss_mse =  F.mse_loss(vt, targets, reduction='none')
        loss_mse = loss_mse * batch.target.mask    
        loss_mse = loss_mse.sum() / batch.target.mask.sum()

        return loss_mse

    def simulate_dynamics(self, batch: DataCoupling) -> DataCoupling:

        """generate target data from source input using trained dynamics
        returns the final state of the bridge at the end of the time interval
        """

        print(self.config.num_jets)
        eps = self.config.time_eps
        steps = self.config.num_timesteps
        time_steps = torch.linspace(eps, 1.0 - eps, steps, device=self.device)
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)

        solver = ContinuousSolver(model=self, config=self.config)
        state = batch.source.clone()
        
        for i, t in enumerate(time_steps):
            state.time = torch.full((len(state),), t.item(), device=self.device)            
            state = solver.fwd_step(state, delta_t)

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

    def sample(self, time: torch.Tensor, batch: DataCoupling) -> torch.Tensor:

        t = self.reshape_time_dim_like(time, batch) # (B,) -> (B, 1, 1)
        x0 = batch.source.continuous  # (B, D, vocab_size)
        x1 = batch.target.continuous  # (B, D, vocab_size)

        xt = t * x1 + (1.0 - t) * x0   # time-interpolated state
        z = torch.randn_like(xt)       # noise
        xt += self.sigma * z    # Dirac -> Gauss smear

        return xt

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
