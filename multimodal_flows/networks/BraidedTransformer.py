import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from utils.tensorclass import TensorMultiModal
from utils.models import LayerNorm, SelfAttention, CrossAttention, transformer_timestep_embedding

class MultiModalParticleFormer(nn.Module):

    def __init__(self, config):
        """
        config.vocab_size should include a mask token 
        """
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.transformer = nn.ModuleDict(dict(
            wxe =  nn.Sequential(nn.Linear(config.dim_continuous, config.n_embd),
                                 nn.GELU(),
                                 nn.Linear(config.n_embd, config.n_embd)),
            wte = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd),
                                nn.GELU(),
                                nn.Linear(config.n_embd, config.n_embd)),
            drop = nn.Dropout(config.dropout),
            attn_blocks = nn.ModuleList([ConvexAttentionBlock(config) for _ in range(config.n_layer)]),
            ln_kin  = LayerNorm(config.n_embd),
            ln_flv  = LayerNorm(config.n_embd),
            head_kin = nn.Sequential(nn.Linear(config.n_embd, config.n_inner),
                                     nn.GELU(),
                                     nn.Linear(config.n_inner, config.dim_continuous)),
            head_flv = nn.Sequential(nn.Linear(config.n_embd, config.n_inner), 
                                      nn.GELU(),
                                      nn.Linear(config.n_inner, config.vocab_size)) # classifier head for discrete tokens
        ))

        self.apply(self._init_weights)

    def forward(self, state: TensorMultiModal) -> (torch.Tensor, torch.Tensor):
        """
            state.continuous is the corrupted state (B, D, 1) or (B, D, vocab_size) if onehot is True
            state.time is the time in the corruption process (B,)
        """
        
        attn_mask = state.mask.clone()
        attn_mask = state.mask.bool().squeeze()                   # (B, D)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)           # (B, 1, 1, D)
        attn_mask = attn_mask & attn_mask.transpose(-1, -2)       # (B, 1, D, D)
        attn_mask = attn_mask.expand(-1, self.n_head, -1, -1)     # (B, n_heads, D, D)

        # initial embeddings
        kin_emb = self.transformer.wxe(state.continuous)                       # feature embeddings of shape (B, D, n_embd)
        flv_emb = self.transformer.wte(state.discrete.squeeze(-1))             # token embeddings of shape (B, D, n_embd)
        time_emb = transformer_timestep_embedding(state.time, self.n_embd)     # time embedding of shape (B, n_embd)
        time_emb = time_emb.unsqueeze(1)                                       # (B, 1, n_embd)

        k = self.transformer.drop(kin_emb + time_emb)
        f = self.transformer.drop(flv_emb + time_emb)

        f_skip = f.clone()  # skip connection for final layer norm  
        k_skip = k.clone()  # skip connection for final layer norm

        # mode encoder blocks

        for block in self.transformer.attn_blocks:
            k, f = block(k, f, attn_mask=attn_mask)
            k = k + time_emb 
            f = f + time_emb 

        k = self.transformer.ln_kin(k + k_skip + time_emb)  # final layer norm for continuous features
        f = self.transformer.ln_flv(f + f_skip + time_emb)  # final layer norm for discrete tokens

        head_kin = self.transformer.head_kin(k) # regressor head for continuous feats
        head_flv = self.transformer.head_flv(f) # classifier head for discrete tokens

        return head_kin, head_flv

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.n_inner, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.n_inner, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class ConvexAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        gate_val = math.log(config.init_gate_val / (1 - config.init_gate_val))

        self.ln1_x           = LayerNorm(config.n_embd, bias=config.bias)
        self.ln1_y           = LayerNorm(config.n_embd, bias=config.bias)
        self.self_attn_x     = SelfAttention(config)
        self.self_attn_y     = SelfAttention(config)
        self.cross_attn_xy   = CrossAttention(config)
        self.cross_attn_yx   = CrossAttention(config)
        self.gate_x          = nn.Parameter(torch.tensor(gate_val))
        self.gate_y          = nn.Parameter(torch.tensor(gate_val))
        self.ln2_x           = LayerNorm(config.n_embd, bias=config.bias)
        self.ln2_y           = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp_x           = MLP(config)
        self.mlp_y           = MLP(config)

    def forward(self, x, y, attn_mask=None):
        """
        x: (B, D_x, dim), y: (B, T_y, C)
        returns updated (x', y')
        """

        # first modality
        gx = torch.sigmoid(self.gate_x)
        x_ln = self.ln1_x(x)
        x_sa   = self.self_attn_x(x_ln, attn_mask=attn_mask)      # self-attn
        x_ca   = self.cross_attn_xy(x_ln, y, attn_mask=attn_mask) # cross-attn from y→x
        
        x = x + gx * x_sa + (1.0 - gx) * x_ca                     # convex bimodal attention
        x = x + self.mlp_x(self.ln2_x(x))

        # second modality
        gy = torch.sigmoid(self.gate_y)
        y_ln = self.ln1_y(y)
        y_sa   = self.self_attn_y(y_ln, attn_mask=attn_mask)
        y_ca   = self.cross_attn_yx(y_ln, x, attn_mask=attn_mask)
        
        y = y + gy * y_sa + (1.0 - gy) * y_ca
        y = y + self.mlp_y(self.ln2_y(y))

        return x, y
