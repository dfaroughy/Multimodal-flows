import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from utils.tensorclass import TensorMultiModal
from utils.models import LayerNorm, MLP


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


class GattedCrossAttnBlock(nn.Module):
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

        # learnable time gates

        self.gate_net = MLP(config.n_embd, config.n_embd // 2, 4)
        nn.init.constant_(self.gate_net.c_proj.bias, -3.0) # initialized strongly negative so gates start near zero

        # cross-mode
        self.cross_attn_x = CrossAttention(config.n_embd, config.n_head, dropout=config.dropout, bias=config.bias, qk_layernorm=config.qk_layernorm)
        self.cross_ffw_x = MLP(config.n_embd, n_inner, dropout=config.dropout, bias=config.bias)
        self.cross_attn_y = CrossAttention(config.n_embd, config.n_head, dropout=config.dropout, bias=config.bias, qk_layernorm=config.qk_layernorm) 
        self.cross_ffw_y = MLP(config.n_embd, n_inner, dropout=config.dropout, bias=config.bias)


    def forward(self, x, y, t, attn_mask=None):

        # time gates
        t = t.squeeze(1)          # (B, n_embd)
        g_attn_x, g_attn_y, g_ffw_x, g_ffw_y = torch.sigmoid(self.gate_net(t)).unbind(-1)
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


# basic attention mechanisms


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0, bias=True, qk_layernorm=True):
        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.qk_layernorm = qk_layernorm
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if qk_layernorm:
            self.q_layernorm = LayerNorm(n_embd // n_head, bias=bias)
            self.k_layernorm = LayerNorm(n_embd // n_head, bias=bias)

    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.qk_layernorm:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        if self.flash: # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=False)
        else:
            raise NotImplementedError
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y


class CrossAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0, bias=True, qk_layernorm=True):
        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.qk_layernorm = qk_layernorm
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.c_query = nn.Linear(n_embd, n_embd, bias=bias)
        self.c_attn = nn.Linear(n_embd, 2 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if qk_layernorm:
            self.q_layernorm = LayerNorm(n_embd // n_head, bias=bias)
            self.k_layernorm = LayerNorm(n_embd // n_head, bias=bias)

    def forward(self, x, z, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q = self.c_query(x)
        k, v = self.c_attn(z).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.qk_layernorm:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=False)
        else:
            raise NotImplementedError

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y
