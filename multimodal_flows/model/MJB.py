import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.nn import functional as F
from typing import List, Tuple, Dict, Union
from torch.nn.functional import softmax
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.tensorclass import TensorMultiModal
from utils.datasets import DataCoupling
from utils.thermostats import ConstantThermostat
from model.solvers import DiscreteSolver
from networks.ParticleTransformers import FlavorFormer


class MarkovJumpBridge(L.LightningModule):
    def __init__(self, config):
                 
        super().__init__()

        self.gamma=config.gamma
        self.vocab_size=config.vocab_size
        self.num_jets=config.num_jets
        self.max_num_particles=config.max_num_particles
        self.lr_final=config.lr_final
        self.lr=config.lr
        self.max_epochs=config.max_epochs
        self.time_eps=config.time_eps
        self.temperature = config.temperature
        self.num_timesteps = config.num_timesteps
        self.path_snapshots_idx = False

        thermostat = ConstantThermostat(self.gamma, self.vocab_size)
        
        self.bridge_discrete = RandomTelegraphBridge(self.gamma, self.vocab_size, thermostat, self.temperature)        
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

    def loss(self, batch: DataCoupling) -> torch.Tensor:

        """ Markov bridge CE loss
        """

        # sample time uniformly in [eps, 1-eps], eps << 1

        time = self.time_eps  + (1. - self.time_eps ) * torch.rand(len(batch), device=self.device)  # (B,)

        # sample continuous state from bridge

        state = self.bridge_discrete.sample(time, batch)
        state = state.to(self.device)

        # compute loss

        logits = self.model(state)                            
        targets = batch.target.discrete.to(self.device)        
        loss_ce =  F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), reduction='none')    # (B*D,)
        loss_ce = loss_ce.view(len(batch), -1) * state.mask    # (B, D)
        loss_ce = loss_ce.sum() / state.mask.sum() 

        return loss_ce

    def simulate_dynamics(self, batch: DataCoupling) -> DataCoupling:

        """generate target data from source input using trained dynamics
        returns the final state of the bridge at the end of the time interval
        """
        
        solver = DiscreteSolver(model=self, vocab_size=self.vocab_size, method='tauleap', temperature=self.temperature)
        time_steps = torch.linspace(self.time_eps, 1.0 - self.time_eps, self.num_timesteps, device=self.device)
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)

        state = batch.source.clone()
        
        for i, t in enumerate(time_steps):
            is_last_step = (i == len(time_steps) - 1)
            state.time = torch.full((len(state),), t.item(), device=self.device)            
            state = solver.fwd_step(state, delta_t, is_last_step)

        batch.target = state

        return batch


class RandomTelegraphBridge:
    """Multivariate Random Telegraph bridge for discrete states
    - t: time
    - k0: discrete source state at t=0
    - k1: discrete  target state at t=1
    - k: discrete state at time t
    """

    def __init__(self, gamma, vocab_size, thermostat_fn, temperature=1.0):
        self.gamma = gamma
        self.vocab_size = vocab_size
        self.thermostat = thermostat_fn
        self.temperature = temperature

    def rate(self, state: TensorMultiModal, logits: torch.Tensor):
        """ input:
            - state.time (t): (B, 1) time tensor
            - state.discrete (k) : (B, D, 1) current state tensor
            - logits: (B, D, vocab_size) logits tensor

            output:
            - rates: (B, D, vocab_size) transition rates tensor

        """

        t = state.time.unsqueeze(1)
        k = state.discrete

        assert t.shape == (len(state), 1)
        assert (k >= 0).all() and (k < self.vocab_size).all(), (
            "Values in `k` outside of bound! k_min={}, k_max={}".format(
                k.min(), k.max()
            )
        )

        qx = softmax(logits, dim=2) # transition probabilities to all states
        qy = torch.gather(qx, 2, k.long())  # current state prob

        # ...Telegraph process rates:

        t = t.squeeze()
        wt = self.thermostat.w_ts(t, 1.0)
        A = 1.0
        B = (wt * self.vocab_size) / (1.0 - wt)
        C = wt
        rate = A + B[:, None, None] * qx + C[:, None, None] * qy
        return rate

    def sample(self, time: torch.Tensor, batch: DataCoupling) -> TensorMultiModal:

        # time: (B,) 
        k0 = batch.source.discrete                   # (B, D, 1)
        k1 = batch.target.discrete                   # (B, D, 1)  

        transition_probs = self.transition_probability(time, k0, k1)
        state = Categorical(transition_probs).sample().to(k1.device) # (B, D)
        state = state.unsqueeze(-1) # (B, D, 1)

        return TensorMultiModal(time=time, discrete=state, mask=batch.target.mask)

    def transition_probability(self, t, k0, k1):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """
        # ...set state configurations:

        k = torch.arange(0, self.vocab_size)  # (vocab_size,)
        k = k[None, None, :].repeat(k0.size(0), k0.size(1), 1).float()  # (B, D, vocab_size)
        k = k.to(k0.device)

        # ...compute probabilities:

        p_k_to_k1 = self.conditional_probability(t, 1.0, k, k1)
        p_k0_to_k = self.conditional_probability(0.0, t, k0, k)
        p_k0_to_k1 = self.conditional_probability(0.0, 1.0, k0, k1)

        return (p_k_to_k1 * p_k0_to_k) / p_k0_to_k1

    def conditional_probability(self, t_in, t_out, k_in, k_out):
        """
        \begin{equation}
        P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
        \end{equation}

        \begin{equation}
        w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
        \end{equation}

        """

        t_out = right_time_size(t_out, k_out).to(k_in.device)
        t_in = right_time_size(t_in, k_out).to(k_in.device)

        wt = self.thermostat.w_ts(t_in, t_out)

        k_out, k_in = right_shape(k_out), right_shape(k_in)
        kronecker = (k_out == k_in).float()
        prob = 1.0 / self.vocab_size + wt[:, None, None] * ((-1.0 / self.vocab_size) + kronecker)
        return prob


right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
right_time_size = (
    lambda t, x: t
    if isinstance(t, torch.Tensor)
    else torch.full((x.size(0),), t).to(x.device)
)
