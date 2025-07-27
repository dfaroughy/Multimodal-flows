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
from utils.models import MLP, transformer_timestep_embedding
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
        self.loss_combine = MultiTaskLoss(config)

        self.save_hyperparameters(vars(config))
        self.config = config

        # time-dependent loss uncertainty weights
        # if config.weighted_loss:
        #     self.uncertainty_net = MLP(config.n_embd, config.n_embd, n_out=2) 
        #     nn.init.constant_(self.uncertainty_net.c_proj.bias, 0.0) # start balanced L = Lmse + Lce

    # ...Lightning functions

    def forward(self, state: TensorMultiModal) -> (torch.Tensor, torch.Tensor):
        return self.model(state)

    def training_step(self, batch: DataCoupling, batch_idx):
        loss, loss_mse, loss_ce, w_mse, w_ce = self.loss(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
        self.log("train_loss_ce", loss_ce, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=len(batch))
        self.log("train_loss_mse", loss_mse, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=len(batch))
        if self.config.weighted_loss:
            self.log("train_weight_mse", w_mse, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=len(batch))
            self.log("train_weight_ce", w_ce, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=len(batch))
        return {"loss": loss}

    def validation_step(self, batch: DataCoupling, batch_idx):
        loss, loss_mse, loss_ce, w_mse, w_ce = self.loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
        self.log("val_loss_ce", loss_ce, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
        self.log("val_loss_mse", loss_mse, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
        if self.config.weighted_loss:
            self.log("val_weight_mse", w_mse, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=len(batch))
            self.log("val_weight_ce", w_ce, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=len(batch))
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


    def loss(self, batch: DataCoupling) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        total_loss: the uncertainty-weighted sum
        mse_mean:   the (unweighted) MSE averaged over the batch
        ce_mean:    the (unweighted) CE averaged over the batch
        """
        B = len(batch)
        V = self.config.vocab_size
        eps = self.config.time_eps
        time = eps + (1.0 - eps) * torch.rand(B, device=self.device)

        # sample intermediate states
        xt = self.bridge_continuous.sample(time, batch)   # (B,D,dim_cont)
        kt = self.bridge_discrete.sample(time, batch)     # (B,D,1)
        state = TensorMultiModal(continuous=xt, discrete=kt, mask=batch.target.mask, time=time).to(self.device)

        vt, logits = self.model(state)   # (B,D,dim_cont), (B,D,V)

        #  MSE for continuous states
        targets_continuous = self.bridge_continuous.conditional_drift(state, batch)  # (B,D,dim_cont)
        mse = F.mse_loss(vt, targets_continuous, reduction="none")  # (B,D,dim_cont)
        mse = (mse * state.mask).sum(dim=[1,2])                     # (B,)
        loss_mse = mse / state.mask.sum(dim=[1,2]).clamp_min(1.0)        # (B,)

        # CE for discrete states
        targets_discrete = batch.target.discrete.view(-1).to(self.device) 
        ce = F.cross_entropy(logits.view(-1, V), targets_discrete, ignore_index=0, reduction="none") # (B*D,)
        ce = ce.view(B, -1) * state.mask.squeeze(-1)                           # (B,D)
        loss_ce = ce.sum(dim=1) / state.mask.squeeze(-1).sum(dim=1).clamp_min(1.0)  # (B,)
        loss = self.loss_combine(loss_mse, loss_ce, state)
        return loss

        # if self.config.weighted_loss:
        #     # predict time-dependent uncertainty weights for multi-task loss
        #     t_emb = transformer_timestep_embedding(time, self.config.n_embd)  # (B, n_embd)
        #     u_mse, u_ce = self.uncertainty_net(t_emb).unbind(-1)  # each (B,)
        #     w_mse = torch.exp(-u_mse)
        #     w_ce = torch.exp(-u_ce)
        #     loss  = 0.5 * (w_mse * loss_mse + u_mse) + 0.5 * (w_ce * loss_ce + u_ce)       # (B,)
        #     return loss.mean(), loss_mse.mean(), loss_ce.mean(), w_mse.mean(), w_ce.mean() 
        # else:
        #     # unweighted loss:
        #     loss = loss_mse + loss_ce
        #     return loss.mean(), loss_mse.mean(), loss_ce.mean(), None, None


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



class MultiTaskLoss(nn.Module):
    def __init__(self, config):

        if self.config.loss_combine == "time-weighted":
            self.uncertainty_net = MLP(config.n_embd, config.n_embd, n_out=2) 
            nn.init.constant_(self.uncertainty_net.c_proj.bias, 0.0) # start balanced L = Lmse + Lce
            
        elif self.config.loss_combine == "weighted":
            self.loss_weights = nn.Parameter(torch.tensor([0.0, 0.0]))

    def forward(self, loss_1, loss_2, state=None):

        if self.config.loss_combine == "weighted":
            u1, u2 = self.loss_weights 
            w1, w2 = torch.exp(-u1), torch.exp(-u2)
            loss = 0.5 * (w1 * loss_1 + w2 * loss_2)
            return loss.mean(), loss_1.mean(), loss_2.mean(), w1, w2

        if self.config.loss_combine == "time-weighted":
            t_emb = transformer_timestep_embedding(state.time, self.config.n_embd)  # (B, n_embd)
            u1, u2 = self.uncertainty_net(t_emb).unbind(-1)  # each (B,)
            w1, w2 = torch.exp(-u1), torch.exp(-u2)
            loss = 0.5 * (w1 * loss_1 + w2 * loss_2)
            return loss.mean(), loss_1.mean(), loss_2.mean(), w_mse.mean(), w_ce.mean() 

        else:
            loss = loss_1 + loss_2
            return loss.mean(), loss_1.mean(), loss_2.mean(), None, None 
