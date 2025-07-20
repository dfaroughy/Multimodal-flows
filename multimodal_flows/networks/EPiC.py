import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.weight_norm as wn

from utils.tensorclass import TensorMultiModal
from utils.models import transformer_timestep_embedding

class EPiC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_embd = config.n_embd
        self.max_num_particles = config.max_num_particles

        self.epic = nn.ModuleDict({
            'wxe': nn.Linear(config.dim_continuous, config.n_embd),
            'proj':  EPiCProjection(dim_time=config.n_embd,
                                    dim_loc=config.n_embd,
                                    dim_glob=config.n_embd,
                                    dim_hid_loc=config.n_embd,
                                    dim_hid_glob=config.n_embd_glob,
                                    pooling_fn=self._meansum_pool,
                                    dropout=config.dropout,
                                    ),
            'layers': nn.ModuleList([EPiCLayer(dim_time=config.n_embd,
                                               dim_loc=config.n_embd,
                                               dim_glob=config.n_embd_glob,
                                               dim_hid_loc=config.n_embd,
                                               dim_hid_glob=config.n_embd_glob,
                                               pooling_fn=self._meansum_pool,
                                               dropout=config.dropout,
                                               ) for _ in range(config.n_layer)]),
            'head':  nn.Linear(2 * config.n_embd + config.n_embd_glob, config.dim_continuous)
            })


    def forward(self, state: TensorMultiModal) -> torch.Tensor:

        mask = state.mask   # (B, D, 1)

        x_emb = self.epic.wxe(state.continuous)  # feature embeddings of shape (B, D, n_embd)
        time_emb = transformer_timestep_embedding(state.time, self.n_embd)  
        time_glob = time_emb.clone()                                        # (B, n_embd)
        time_emb = self._broadcast_global(time_emb, x_emb)                  # (B, D, n_embd)

        x_local, x_global = self.epic.proj(time_emb, x_emb, time_glob, mask)
        x_local_skip = x_local.clone() 
        x_global_skip = x_global.clone() 

        for layer in self.epic.layers:
            x_local, x_global = layer(time_emb, x_local, x_global, mask)
            x_local += x_local_skip
            x_global += x_global_skip
            
        # ... final layer
        
        out_global = x_global.clone()
        x_global = self._broadcast_global(x_global, local=x_local)
        h = torch.cat([time_emb, x_local, x_global], dim=-1)

        return self.epic.head(h)


    def _meansum_pool(self, mask, x_local, *x_global, scale=0.01):
        """masked pooling local features with mean and sum
        the concat with global features
        """
        x_sum = (x_local * mask).sum(1, keepdim=False)
        x_mean = x_sum / mask.sum(1, keepdim=False)
        x_pool = torch.cat([x_mean, x_sum * scale, *x_global], 1)
        return x_pool

    def _broadcast_global(self, x, local):
        dim = x.size(1)
        D = local.size(1)
        return x.view(-1, 1, dim).repeat(1, D, 1)


class EPiCProjection(nn.Module):
    def __init__(
        self,
        dim_time: int,
        dim_loc: int,
        dim_glob: int,
        dim_hid_loc: int,
        dim_hid_glob: int,
        pooling_fn: callable,
        activation_fn: callable = nn.GELU(),
        dropout: float = 0.0,
    ):
        super(EPiCProjection, self).__init__()

        self.pooling_fn = pooling_fn

        self.mlp_local = nn.Sequential(
            wn(nn.Linear(dim_time + dim_loc, dim_hid_loc)),
            activation_fn,
            wn(nn.Linear(dim_hid_loc, dim_hid_loc)),
            activation_fn,
        )

        self.mlp_global = nn.Sequential(
            wn(nn.Linear(2 * dim_hid_loc + dim_glob, dim_hid_loc)),
            activation_fn,
            wn(nn.Linear(dim_hid_loc, dim_hid_glob)),
            activation_fn,
        )

    def forward(self, time, x_local, x_global, mask):
        """Input shapes:
         - time_local = (B, D, dim_time_emb)
         - x_local = (B, D, dim_local)
         - x_global = (B, dim_global)
        Out shapes:
         - x_local = (B, D, dim_hidden_local)
         - x_global = (B, dim_hidden_global)
        """

        x_local = self.mlp_local(torch.cat([time, x_local], dim=-1))
        x_global = self.pooling_fn(mask, x_local, x_global)
        x_global = self.mlp_global(x_global)

        return x_local, x_global


class EPiCLayer(nn.Module):
    # Temporal layer based on https://github.com/uhh-pd-ml/EPiC-GAN/blob/main/models.py
    def __init__(
        self,
        dim_time: int,
        dim_loc: int,
        dim_glob: int,
        dim_hid_glob: int,
        dim_hid_loc: int,
        pooling_fn: callable,
        activation_fn: callable = F.leaky_relu,
        dropout: float = 0.0,
    ):
        super(EPiCLayer, self).__init__()

        self.pooling_fn = pooling_fn
        self.act_fn = activation_fn

        self.fc_glob1 = wn(nn.Linear(2 * dim_loc + dim_glob, dim_loc))
        self.fc_glob2 = wn(nn.Linear(dim_loc, dim_hid_glob))
        self.fc_loc1 = wn(nn.Linear(dim_time + dim_loc + dim_glob, dim_hid_loc))
        self.fc_loc2 = wn(nn.Linear(dim_hid_loc, dim_hid_loc))

        self.dropout = nn.Dropout(dropout)

    def forward(self, time, x_local, x_global, mask):
        """Input/Output shapes:
        - x_local: (b, num_points, dim_loc)
        - x_global = [b, dim_glob]
        - context = [b, dim_cond]
        """
        # ...global features
        global_hidden = self.pooling_fn(mask, x_local, x_global)
        global_hidden = self.act_fn(self.fc_glob1(global_hidden))
        x_global += self.fc_glob2(global_hidden)  # skip connection
        global_hidden = self.dropout(self.act_fn(x_global))

        # ...broadcast global features to each particle
        global2local = self._broadcast_global(x_global, local=x_local)

        # ...local features
        local_hidden = torch.cat([time, x_local, global2local], 2)
        local_hidden = self.act_fn(self.fc_loc1(local_hidden))
        x_local += self.fc_loc2(local_hidden)  # skip connection
        local_hidden = self.dropout(self.act_fn(x_local))

        return local_hidden, global_hidden

    def _broadcast_global(self, x, local):
        dim = x.size(1)
        D = local.size(1)
        return x.view(-1, 1, dim).repeat(1, D, 1)
