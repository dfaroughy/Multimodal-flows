import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.weight_norm as wn



class EPiC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        dim_input = (
            config.encoder.dim_emb_time
            + config.encoder.dim_emb_continuous
            + config.encoder.dim_emb_discrete * config.data.dim_discrete
        )

        dim_output = (
            config.data.dim_continuous
            + config.data.vocab_size * config.data.dim_discrete
        )

        dim_context = (
            config.encoder.dim_emb_time
            + config.encoder.dim_emb_context_continuous
            + config.encoder.dim_emb_context_discrete * config.data.dim_context_discrete
        )

        self.epic = EPiCEncoder(
            dim_time=config.encoder.dim_emb_time,
            dim_input_loc=dim_input,
            dim_input_glob=dim_context,
            dim_output_loc=dim_output,
            dim_hid_loc=config.encoder.dim_hidden_local,
            dim_hid_glob=config.encoder.dim_hidden_glob,
            num_blocks=config.encoder.num_blocks,
            use_skip_connection=config.encoder.skip_connection,
            dropout=config.encoder.dropout,
        )

    def forward(self, state_local: TensorMultiModal, state_global: TensorMultiModal):
        local_modes = [
            getattr(state_local, mode) for mode in state_local.available_modes()
        ]
        global_modes = [
            getattr(state_global, mode) for mode in state_global.available_modes()
        ]

        local_cat = torch.cat(local_modes, dim=-1)
        global_cat = torch.cat(global_modes, dim=-1)
        mask = state_local.mask

        h_loc, h_glob = self.epic(state_local.time, local_cat, global_cat, mask)

        if self.config.data.modality == "continuous":
            return TensorMultiModal(continuous=h_loc, mask=mask)

        elif self.config.data.modality == "discrete":
            return TensorMultiModal(discrete=h_loc, mask=mask)




class EPiCEncoder(nn.Module):
    def __init__(
        self,
        dim_time: int,
        dim_input_loc: int,
        dim_input_glob: int,
        dim_output_loc: int,
        num_blocks: int = 6,
        dim_hid_loc: int = 128,
        dim_hid_glob: int = 10,
        use_skip_connection: bool = False,
        project_input: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ...model params:
        self.num_blocks = num_blocks
        self.use_skip_connection = use_skip_connection

        # ...components:
        
        self.epic_layers = nn.ModuleList()

        if project_input:
            self.epic_proj = EPiCProjection(
                dim_time=dim_time,
                dim_loc=dim_input_loc,
                dim_glob=dim_input_glob,
                dim_hid_loc=dim_hid_loc,
                dim_hid_glob=dim_hid_glob,
                pooling_fn=self._meansum_pool,
                dropout=dropout,
            )
        else:
            self.epic_layers.append(
            EPiCLayer(
                dim_time=dim_time,
                dim_loc=dim_input_loc,
                dim_glob=dim_input_glob,
                dim_hid_loc=dim_hid_loc,
                dim_hid_glob=dim_hid_glob,
                pooling_fn=self._meansum_pool,
                dropout=dropout,
                )
            )
            self.num_blocks -= 1

        for _ in range(self.num_blocks):
            self.epic_layers.append(
                EPiCLayer(
                    dim_time=dim_time,
                    dim_loc=dim_hid_loc,
                    dim_glob=dim_hid_glob,
                    dim_hid_loc=dim_hid_loc,
                    dim_hid_glob=dim_hid_glob,
                    pooling_fn=self._meansum_pool,
                    dropout=dropout,
                )
            )

        self.output_layer = wn(
            nn.Linear(dim_time + dim_hid_loc + dim_hid_glob, dim_output_loc)
        )

    def forward(self, time_local, x_local, x_global, mask=None):
        """Input shapes:
         - time_local = (B, D, dim_time_emb)
         - x_local = (B, D, dim_local)
         - x_global = (B, dim_global)  
         """

        # ...Projection network:

        if hasattr(self, "epic_proj"):
            x_local, x_global = self.epic_proj(time_local, x_local, x_global, mask)
        
        x_local_skip = x_local.clone() if self.use_skip_connection else 0
        x_global_skip = x_global.clone() if self.use_skip_connection else 0

        # ...EPiC layers:
        for i in range(self.num_blocks):
            x_local, x_global = self.epic_layers[i](time_local, x_local, x_global, mask)
            x_local += x_local_skip
            x_global += x_global_skip

        # ... final layer
        out_global = x_global.clone()
        x_global = self._broadcast_global(x_global, local=x_local)
        h = torch.cat([time_local, x_local, x_global], dim=-1)
        out_local = self.output_layer(h)

        return out_local, out_global

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
