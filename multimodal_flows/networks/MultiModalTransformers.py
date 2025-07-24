import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from utils.tensorclass import TensorMultiModal
from utils.models import LayerNorm, SelfAttention, CrossAttention, transformer_timestep_embedding, MLP


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
            blocks_x    = nn.ModuleList([SelfAttnBlock(config) for _ in range(config.n_layer)]),
            ln_x        = LayerNorm(config.n_embd),
            blocks_y    = nn.ModuleList([SelfAttnBlock(config) for _ in range(config.n_layer)]),
            ln_y        = LayerNorm(config.n_embd),
            blocks_xy   = nn.ModuleList([TemporalGatedCrossAttnBlock(config) for _ in range(config.n_layer_fused)]),
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

        for block in self.transformer.blocks_xy:
            x, y = block(x, y, time_emb, attn_mask=attn_mask)
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



class FusedParticleFormer(nn.Module):

    def __init__(self, config):
        """
        config.vocab_size should include a mask token 
        """
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.transformer = nn.ModuleDict(dict(
            wxe = nn.Sequential(nn.Linear(config.dim_continuous, config.n_embd),  # continuous modality
                                 nn.GELU(),
                                 nn.Linear(config.n_embd, config.n_embd)),
            wye = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd),   # discrete modality
                                nn.GELU(),
                                nn.Linear(config.n_embd, config.n_embd)),
            drop = nn.Dropout(config.dropout),
            blocks_x = nn.ModuleList([SelfAttnBlock(config) for _ in range(config.n_layer)]),
            blocks_y = nn.ModuleList([SelfAttnBlock(config) for _ in range(config.n_layer)]),
            blocks_xy = nn.ModuleList([SelfAttnBlock(config, n_embd=2*config.n_embd) for _ in range(config.n_layer_fused)]),
            ln_xy= LayerNorm(2 * config.n_embd),
            ln_x_last  = LayerNorm(config.n_embd),
            ln_y_last  = LayerNorm(config.n_embd),
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
        flv_emb = self.transformer.wye(state.discrete.squeeze(-1))             # token embeddings of shape (B, D, n_embd)
        time_emb = transformer_timestep_embedding(state.time, self.n_embd)     # time embedding of shape (B, n_embd)
        time_emb = time_emb.unsqueeze(1)                                       # (B, 1, n_embd)

        k = self.transformer.drop(kin_emb + time_emb)
        f = self.transformer.drop(flv_emb + time_emb)

        f_skip = f.clone()  # skip connection for final layer norm  
        k_skip = k.clone()  # skip connection for final layer norm

        # mode encoder blocks

        for block in self.transformer.blocks_x:
            k = block(k, attn_mask=attn_mask)
            k = k + time_emb 

        for block in self.transformer.blocks_y:
            f = block(f, attn_mask=attn_mask)
            f = f + time_emb 

        h = torch.cat((k + k_skip, f + f_skip), dim=-1)  # concatenate the continuous and discrete embeddings
        h = self.transformer.ln_fused(h)  

        # fused blocks

        for block in self.transformer.blocks_xy:
            h = block(h, attn_mask=attn_mask)

        k, f = h.split((self.n_embd, self.n_embd), dim=-1)  # split the concatenated embeddings back into continuous and discrete parts

        k = self.transformer.ln_x_last(k + k_skip + time_emb)  # final layer norm for continuous features
        f = self.transformer.ln_y_last(f + f_skip + time_emb)  # final layer norm for discrete tokens

        head_kin = self.transformer.head_x(k) # regressor head for continuous feats
        head_flv = self.transformer.head_y(f) # classifier head for discrete tokens

        return head_kin, head_flv



class BraidedCrossAttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.n_inner is None:
            n_inner = 4 * config.n_embd
        else:
            n_inner = config.n_inner

        self.ln1_x = LayerNorm(config.n_embd, bias=config.bias)
        self.self_attn_x = SelfAttention(config.n_embd, config.n_head, dropout=config.dropout, bias=config.bias, qk_layernorm=config.qk_layernorm)
        self.ln2_x = LayerNorm(config.n_embd, bias=config.bias)
        self.self_ffw_x = MLP(config.n_embd, n_inner, dropout=config.dropout, bias=config.bias)

        self.ln1_y = LayerNorm(config.n_embd, bias=config.bias)
        self.self_attn_y = SelfAttention(config.n_embd, config.n_head, dropout=config.dropout, bias=config.bias, qk_layernorm=config.qk_layernorm)
        self.ln2_y = LayerNorm(config.n_embd, bias=config.bias)
        self.self_ffw_y = MLP(config.n_embd, n_inner, dropout=config.dropout, bias=config.bias)

        self.cross_attn_x = CrossAttention(config.n_embd, config.n_head, dropout=config.dropout, bias=config.bias, qk_layernorm=config.qk_layernorm)
        self.attn_gate_x = nn.Parameter(torch.tensor([0.0]))
        self.cross_ffw_x = MLP(config.n_embd, n_inner, dropout=config.dropout, bias=config.bias)
        self.ffw_gate_x = nn.Parameter(torch.tensor([0.0]))

        self.cross_attn_y = CrossAttention(config.n_embd, config.n_head, dropout=config.dropout, bias=config.bias, qk_layernorm=config.qk_layernorm) 
        self.attn_gate_y = nn.Parameter(torch.tensor([0.0]))
        self.cross_ffw_y = MLP(config.n_embd, n_inner, dropout=config.dropout, bias=config.bias)
        self.ffw_gate_y = nn.Parameter(torch.tensor([0.0]))


    def forward(self, x, y, attn_mask=None):

        # self-modal attention
        x = x + self.self_attn_x(self.ln1_x(x), attn_mask=attn_mask) 
        x = x + self.self_ffw_x(self.ln2_x(x)) 

        y = y + self.self_attn_y(self.ln1_y(y), attn_mask=attn_mask) 
        y = y + self.self_ffw_y(self.ln2_y(y)) 

        # Gatted cross-modal attention
        x_gate = x + self.cross_attn_x(x, y, attn_mask=attn_mask) * self.attn_gate_x.tanh()
        x_gate = x_gate + self.cross_ffw_x(x_gate) * self.ffw_gate_x.tanh()

        y_gate = y + self.cross_attn_y(y, x, attn_mask=attn_mask) * self.attn_gate_y.tanh()
        y_gate = y_gate + self.cross_ffw_y(y_gate) * self.ffw_gate_y.tanh()

        return x_gate, y_gate



class TemporalGatedCrossAttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.n_inner is None:
            n_inner = 4 * config.n_embd
        else:
            n_inner = config.n_inner

        # 1st mode
        self.ln1_x = LayerNorm(config.n_embd, bias=config.bias)
        self.self_attn_x = SelfAttention(config.n_embd, config.n_head, dropout=config.dropout, bias=config.bias, qk_layernorm=config.qk_layernorm)
        self.ln2_x = LayerNorm(config.n_embd, bias=config.bias)
        self.self_ffw_x = MLP(config.n_embd, n_inner, dropout=config.dropout, bias=config.bias)

        # 2nd mode
        self.ln1_y = LayerNorm(config.n_embd, bias=config.bias)
        self.self_attn_y = SelfAttention(config.n_embd, config.n_head, dropout=config.dropout, bias=config.bias, qk_layernorm=config.qk_layernorm)
        self.ln2_y = LayerNorm(config.n_embd, bias=config.bias)
        self.self_ffw_y = MLP(config.n_embd, n_inner, dropout=config.dropout, bias=config.bias)

        # 4 time gates
        self.time_gate = nn.Sequential(nn.Linear(config.n_embd, config.n_embd // 2),
                                       nn.GELU(),
                                       nn.Linear(config.n_embd // 2, 4)
                                       )
        nn.init.constant_(self.time_gate[-1].bias, -3.0) # initialized strongly negative so gates start near zero

        # cross-mode
        self.cross_attn_x = CrossAttention(config.n_embd, config.n_head, dropout=config.dropout, bias=config.bias, qk_layernorm=config.qk_layernorm)
        self.cross_ffw_x = MLP(config.n_embd, n_inner, dropout=config.dropout, bias=config.bias)
        self.cross_attn_y = CrossAttention(config.n_embd, config.n_head, dropout=config.dropout, bias=config.bias, qk_layernorm=config.qk_layernorm) 
        self.cross_ffw_y = MLP(config.n_embd, n_inner, dropout=config.dropout, bias=config.bias)


    def forward(self, x, y, t, attn_mask=None):

        # time gates
        t = t.squeeze(1)          # (B, n_embd)
        g_attn_x, g_attn_y, g_ffw_x, g_ffw_y = torch.sigmoid(self.time_gate(t)).unbind(-1)
        g_attn_x = g_attn_x[:, None, None]   # (B,1,1)
        g_attn_y = g_attn_y[:, None, None]   # (B,1,1)
        g_ffw_x  = g_ffw_x[:, None, None]    # (B,1,1)
        g_ffw_y  = g_ffw_y[:, None, None]    # (B,1,1)

        # self-modal attention
        x = x + self.self_attn_x(self.ln1_x(x), attn_mask=attn_mask) 
        x = x + self.self_ffw_x(self.ln2_x(x)) 

        y = y + self.self_attn_y(self.ln1_y(y), attn_mask=attn_mask) 
        y = y + self.self_ffw_y(self.ln2_y(y)) 

        # Gatted cross-modal attention
        x_out = x + g_attn_x * self.cross_attn_x(x, y, attn_mask=attn_mask)
        x_out = x_out + g_ffw_x * self.cross_ffw_x(x_out) 

        y_out = y + g_attn_y * self.cross_attn_y(y, x, attn_mask=attn_mask) 
        y_out = y_out + g_ffw_y * self.cross_ffw_y(y_out) 

        return x_out, y_out


class SelfAttnBlock(nn.Module):
    def __init__(self, config, n_embd=None):
        super().__init__()

        if n_embd is None: 
            n_embd = config.n_embd

        if config.n_inner is None:
            n_inner = 4 * n_embd
        else:
            n_inner = config.n_inner

        self.ln1 = LayerNorm(n_embd, bias=config.bias)
        self.attn = SelfAttention(n_embd, config.n_head, dropout=config.dropout, bias=config.bias, qk_layernorm=config.qk_layernorm)
        self.ln2 = LayerNorm(n_embd, bias=config.bias)
        self.ffw = MLP(n_embd, n_inner, dropout=config.dropout, bias=config.bias)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask) 
        x = x + self.ffw(self.ln2(x)) 
        return x

