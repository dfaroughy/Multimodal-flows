import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from utils.tensorclass import TensorMultiModal
from networks.attention import SelfAttnBlock, TemporalGatedCrossAttnBlock, TemporalConvexCrossAttnBlock
from utils.models import LayerNorm, transformer_timestep_embedding


class GatedParticleFormer(nn.Module):

    def __init__(self, config):
        """
        config.vocab_size should include a mask token 
        """
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.transformer = nn.ModuleDict(dict(
            wxe = nn.Sequential(nn.Linear(config.dim_continuous, config.n_embd),
                                nn.GELU(),
                                nn.Linear(config.n_embd, config.n_embd)),
            wye = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd),
                                nn.GELU(),
                                nn.Linear(config.n_embd, config.n_embd)),
            drop        = nn.Dropout(config.dropout),
            ln_x        = LayerNorm(config.n_embd),
            ln_y        = LayerNorm(config.n_embd),
            blocks      = nn.ModuleList([TemporalGatedCrossAttnBlock(config) for _ in range(config.n_layer)]),
            ln_x_last   = LayerNorm(config.n_embd),
            ln_y_last   = LayerNorm(config.n_embd),
            head_x = nn.Sequential(nn.Linear(config.n_embd, config.n_inner),
                                   nn.GELU(),
                                   nn.Linear(config.n_inner, config.dim_continuous)),
            head_y = nn.Sequential(nn.Linear(config.n_embd, config.n_inner), 
                                   nn.GELU(),
                                   nn.Linear(config.n_inner, config.vocab_size)) # classifier head for discrete tokens
        ))

        self.apply(self._init_weights)

    def forward(self, state: TensorMultiModal) -> (torch.Tensor, torch.Tensor):
        """
            state.continuous: (B, D, dim_continuous) -- corrupted state 
            state.discrete: (B, D, 1) or (B, D, vocab_size) -- corrupted tokens
            state.time: (B,) -- time in corruption process
        """
        
        attn_mask = state.mask.clone()
        attn_mask = state.mask.bool().squeeze()                   # (B, D)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)           # (B, 1, 1, D)
        attn_mask = attn_mask & attn_mask.transpose(-1, -2)       # (B, 1, D, D)
        attn_mask = attn_mask.expand(-1, self.n_head, -1, -1)     # (B, n_heads, D, D)

        # initial embeddings
        x_emb = self.transformer.wxe(state.continuous)                       # feature embeddings of shape (B, D, n_embd)
        y_emb = self.transformer.wye(state.discrete.squeeze(-1))             # token embeddings of shape (B, D, n_embd)
        time_emb = transformer_timestep_embedding(state.time, self.n_embd)     # time embedding of shape (B, n_embd)
        time_emb = time_emb.unsqueeze(1)                                       # (B, 1, n_embd)

        x = self.transformer.drop(x_emb + time_emb)
        y = self.transformer.drop(y_emb + time_emb)
        x_skip = x.clone()  # skip connection for final layer norm  
        y_skip = y.clone()  # skip connection for final layer norm

        # braided blocks

        for block in self.transformer.blocks:
            x, y = block(x, y, time_emb, attn_mask=attn_mask)
            x = x + time_emb
            y = y + time_emb
        
        x = self.transformer.ln_x_last(x + x_skip)  # final layer norm for continuous features
        y = self.transformer.ln_y_last(y + y_skip)  # final layer norm for discrete tokens

        return self.transformer.head_x(x), self.transformer.head_y(y)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class GatedConvexParticleFormer(nn.Module):

    def __init__(self, config):
        """
        config.vocab_size should include a mask token 
        """
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.transformer = nn.ModuleDict(dict(
            wxe = nn.Sequential(nn.Linear(config.dim_continuous, config.n_embd),
                                nn.GELU(),
                                nn.Linear(config.n_embd, config.n_embd)),
            wye = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd),
                                nn.GELU(),
                                nn.Linear(config.n_embd, config.n_embd)),
            drop        = nn.Dropout(config.dropout),
            blocks_x    = nn.ModuleList([SelfAttnBlock(config) for _ in range(config.n_layer)]),
            ln_x        = LayerNorm(config.n_embd),
            blocks_y    = nn.ModuleList([SelfAttnBlock(config) for _ in range(config.n_layer)]),  # half of the layers for discrete tokens
            ln_y        = LayerNorm(config.n_embd),
            blocks_xy   = nn.ModuleList([TemporalConvexCrossAttnBlock(config) for _ in range(config.n_layer_fused)]),
            blocks_yx   = nn.ModuleList([TemporalConvexCrossAttnBlock(config) for _ in range(config.n_layer_fused)]),
            ln_x_last   = LayerNorm(config.n_embd),
            ln_y_last   = LayerNorm(config.n_embd),
            head_x = nn.Sequential(nn.Linear(config.n_embd, config.n_inner),
                                   nn.GELU(),
                                   nn.Linear(config.n_inner, config.dim_continuous)),
            head_y = nn.Sequential(nn.Linear(config.n_embd, config.n_inner), 
                                   nn.GELU(),
                                   nn.Linear(config.n_inner, config.vocab_size)) # classifier head for discrete tokens
        ))


        self.apply(self._init_weights)

    def forward(self, state: TensorMultiModal) -> (torch.Tensor, torch.Tensor):
        """
            state.continuous: (B, D, dim_continuous) -- corrupted state 
            state.discrete: (B, D, 1) or (B, D, vocab_size) -- corrupted tokens
            state.time: (B,) -- time in corruption process
        """
        
        attn_mask = state.mask.clone()
        attn_mask = state.mask.bool().squeeze()                   # (B, D)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)           # (B, 1, 1, D)
        attn_mask = attn_mask & attn_mask.transpose(-1, -2)       # (B, 1, D, D)
        attn_mask = attn_mask.expand(-1, self.n_head, -1, -1)     # (B, n_heads, D, D)

        # initial embeddings
        x_emb = self.transformer.wxe(state.continuous)                       # feature embeddings of shape (B, D, n_embd)
        y_emb = self.transformer.wye(state.discrete.squeeze(-1))             # token embeddings of shape (B, D, n_embd)
        time_emb = transformer_timestep_embedding(state.time, self.n_embd)     # time embedding of shape (B, n_embd)
        time_emb = time_emb.unsqueeze(1)                                       # (B, 1, n_embd)

        x = self.transformer.drop(x_emb + time_emb)
        y = self.transformer.drop(y_emb + time_emb)
        x_skip = x.clone()  # skip connection for final layer norm  
        y_skip = y.clone()  # skip connection for final layer norm

        # mode encoder blocks

        for block in self.transformer.blocks_x:
            x_ = x.clone()  # clone for skip connection
            x = block(x, attn_mask=attn_mask)
            x = x + time_emb 

        x = self.transformer.ln_x(x + x_skip)  # layer norm for continuous features
        x_skip = x.clone()  # skip connection for final layer norm

        for block in self.transformer.blocks_y:
            y = block(y, attn_mask=attn_mask)
            y = y + time_emb 

        y = self.transformer.ln_y(y + y_skip)  # layer norm for discrete tokens
        y_skip = y.clone()  # skip connection for final layer norm

        # braided blocks

        for i, block in enumerate(self.transformer.blocks_xy):
            x_in = x.clone()  # clone for skip connection
            x = block(x, y, time_emb, attn_mask=attn_mask)
            y = self.transformer.blocks_yx[i](y, x_in, time_emb, attn_mask=attn_mask)  # apply the same block to discrete tokens
            x = x + time_emb 
            y = y + time_emb

        
        x = self.transformer.ln_x_last(x + x_skip)  # final layer norm for continuous features
        y = self.transformer.ln_y_last(y + y_skip)  # final layer norm for discrete tokens
        head_x = self.transformer.head_x(x) # regressor head for continuous feats
        head_y = self.transformer.head_y(y) # classifier head for discrete tokens

        return head_x, head_y

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



