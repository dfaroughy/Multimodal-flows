import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.nn import functional as F
from typing import List, Tuple, Dict, Union

from torch.optim.lr_scheduler import CosineAnnealingLR

from tensorclass import TensorMultiModal
from datamodules.datasets import DataCoupling
from transformers import GPT2Config

from model.solvers import DiscreteSolver
from model.bridges import RandomTelegraphBridge
from model.thermostats import Thermostat
from networks.JetFlavorGPT import JetSeqGPT

class ConstantThermostat(Thermostat):
    ''' beta(r) = const.
    '''
    def _integral(self, t0, t1):
        return t1 - t0


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

        thermostat = ConstantThermostat(self.gamma , self.vocab_size)

        self.bridge_discrete = RandomTelegraphBridge(gamma=self.gamma,
                                                     vocab_size=self.vocab_size,
                                                     thermostat_fn=thermostat,
                                                     )        
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

        # if state.has_discrete:
        #     max_rate = torch.max(rates, dim=2)[1]
        #     state.discrete = max_rate.unsqueeze(-1)

        paths.append(state)  # append t=1 generated target
        paths = TensorMultiModal.stack(paths, dim=0)
        return paths

    def reshape_time_dim_like(self, t, state: Union[TensorMultiModal, DataCoupling]):
        if isinstance(t, (float, int)):
            return t
        else:
            return t.reshape(-1, *([1] * (state.ndim - 1)))
