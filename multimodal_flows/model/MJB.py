import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.nn import functional as F
from typing import List, Tuple, Dict, Union
from abc import ABC, abstractmethod
from torch.optim.lr_scheduler import CosineAnnealingLR

from tensorclass import TensorMultiModal
from datamodules.datasets import DataCoupling
from model.solvers import DiscreteSolver
from model.bridges import RandomTelegraphBridge
from networks.ParticleTransformers import JetSeqGPT


class MarkovJumpBridge(L.LightningModule):
    """Bridge-Matching model for multi-modal data
    """
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
        self.num_timesteps = config.num_timesteps if hasattr(config, 'num_timesteps') else 100
        self.path_snapshots_idx = False

        thermostat = ConstantThermostat(self.gamma, self.vocab_size)
        self.bridge_discrete = RandomTelegraphBridge(self.gamma, self.vocab_size, thermostat)        
        self.model = JetSeqGPT(config)
        
        self.save_hyperparameters()

    # ...Lightning functions

    def forward(self, state: TensorMultiModal) -> TensorMultiModal:
        return self.model(input_ids=state.discrete.squeeze(-1).long(), 
                          time=state.time.squeeze(-1),
                          )
        
    def training_step(self, batch: DataCoupling, batch_idx):

        state = self.sample_bridges(batch)
        state = state.to(self.device)
        logits = self.model(input_ids=state.discrete.squeeze(-1).long(),
                            time=state.time.squeeze(-1),
                            )
        loss = self.loss(logits, batch)

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
        logits = self.model(input_ids=state.discrete.squeeze(-1).long(),
                            time=state.time.squeeze(-1),
                            )
        loss = self.loss(logits, batch)
        
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

        source_tokens = torch.randint(0, self.vocab_size, (len(batch), self.max_num_particles, 1), device=self.device)
        mask = torch.ones((len(batch), 
                           self.max_num_particles, 1), 
                           device=self.device).long()

        batch.source = TensorMultiModal(None, None, source_tokens, mask)
        noisy_tokens = self.bridge_discrete.sample(time, batch)

        return TensorMultiModal(time, None, noisy_tokens, mask)


    def loss(self, logits, batch: DataCoupling) -> torch.Tensor:
        """cross-entropy loss for discrete state classifier
        """
        targets = batch.target.discrete.squeeze(-1).to(self.device)
        loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='mean') # (B*D,)
        return loss_ce


    def simulate_dynamics(self, state: TensorMultiModal) -> TensorMultiModal:
        """generate target data from source input using trained dynamics
        returns the final state of the bridge at the end of the time interval
        """
        eps = self.time_eps  # min time resolution
        state.time = torch.full((len(state), 1), eps, device=self.device)  # (B,1) t_0=eps
        state.broadcast_time() # (B,1) -> (B,D,1)
        steps = self.num_timesteps
        print('INFO: Simulating {} steps'.format(steps))

        time_steps = torch.linspace(eps, 1.0 - eps, steps, device=self.device)
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)

        solver_discrete = DiscreteSolver(model=self,
                                         vocab_size=self.vocab_size, 
                                         method='tauleap',
                                         )

        paths = [state.clone()]  # append t=0 source

        for i, t in enumerate(time_steps):
            is_last_step = (i == len(time_steps) - 1)

            state.time = torch.full((len(state), 1), t.item(), device=self.device)            
            state = solver_discrete.fwd_step(state, delta_t, is_last_step)
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


class RandomTelegraphBridge:
    """Multivariate Random Telegraph bridge for discrete states
    - t: time
    - k0: discrete source state at t=0
    - k1: discrete  target state at t=1
    - k: discrete state at time t
    """

    def __init__(self, gamma, vocab_size, thermostat_fn):
        self.gamma = gamma
        self.vocab_size = vocab_size
        self.thermostat = thermostat_fn

    def rate(self, state: TensorMultiModal, logits: torch.Tensor):
        """ input:
            - state.time (t): (B, 1) time tensor
            - state.discrete (k) : (B, D, 1) current state tensor
            - logits: (B, D, vocab_size) logits tensor

            output:
            - rates: (B, D, vocab_size) transition rates tensor

        """
        
        t = state.time
        k = state.discrete

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

    def sample(self, t, batch: DataCoupling):

        k0 = batch.source.discrete
        k1 = batch.target.discrete

        transition_probs = self.transition_probability(t, k0, k1)
        drawn_state = Categorical(transition_probs).sample().to(k1.device)

        if drawn_state.dim() == 2:
            drawn_state = drawn_state.unsqueeze(-1)

        return drawn_state

    def transition_probability(self, t, k0, k1):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """
        # ...reshape input tenors:
        t = t.clone().squeeze()  # shape: (B)

        if k0.dim() == 1:
            k0 = k0.unsqueeze(1)  

        if k1.dim() == 1:
            k1 = k1.unsqueeze(1)

        # ...set state configurations:
        k = torch.arange(0, self.vocab_size)  # (V,)
        k = k[None, None, :].repeat(k0.size(0), k0.size(1), 1).float()  # (B, N, V)
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


class Thermostat(ABC):
    def __init__(self, gamma, vocab_size=8):
        self.gamma = gamma
        self.vocab_size = vocab_size

    @abstractmethod
    def _integral(self, t0, t1):
        pass

    def wt_0(self, t):
        wt = self.w_ts(t, 1)
        return wt * self.vocab_size / (1 - wt)

    def wt_1(self, t):
        return self.w_ts(t, 1)

    def w_ts(self, t0, t1):
        return torch.exp(-self.vocab_size * self.gamma * self._integral(t0, t1))

class ConstantThermostat(Thermostat):
    ''' beta(r) = const.
    '''
    def _integral(self, t0, t1):
        return t1 - t0

class InverseThermostat(Thermostat):
    ''' beta(r) = 1/r
    '''
    def _integral(self, t0, t1):
        return torch.log(t1 / t0)

class LinearThermostat(Thermostat):
    ''' beta(r) = r
    '''
    def _integral(self, t0, t1):
        return (t1**2 - t0**2) / 2

class InverseSquareThermostat(Thermostat):
    ''' beta(r) = -1/r^2
    '''
    def _integral(self, t0, t1):
        return (t1 - t0) / (t1 * t0)

class SigmoidThermostat(Thermostat):
    ''' beta(r) = 1/(1+r)
    '''
    def _integral(self, t0, t1):
        return torch.tanh(t1/2) - torch.tanh(t0/2)



right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
right_time_size = (
    lambda t, x: t
    if isinstance(t, torch.Tensor)
    else torch.full((x.size(0),), t).to(x.device)
)
