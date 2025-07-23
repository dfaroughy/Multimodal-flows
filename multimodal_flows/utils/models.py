
import torch
from torch import nn
import math
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, n_embd, n_inner, dropout=0.0, bias=True):
        super().__init__()

        self.c_fc    = nn.Linear(n_embd, n_inner, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(n_inner, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False
    """
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


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


class TimeFourierEmbedding(nn.Module):
    """
    Turn a scalar t∈[0,1] into a D-dim Fourier feature vector:
      [ sin(t * ω₁), …, sin(t * ω_{D/2}), cos(t * ω₁), …, cos(t * ω_{D/2}) ]
    with frequencies ω log-spaced from 1 to max_freq.
    """
    def __init__(self, dim: int, max_freq: float = 10.0):
        super().__init__()
        half = dim // 2
        inv_freq = 1.0 / ( max_freq ** (torch.arange(half).float() / (half - 1)) )
        self.register_buffer("inv_freq", inv_freq)   # (D/2,)

    def forward(self, t: torch.Tensor):
        # t: (B, 1) or (B,) → ensure (B,1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        # x = t * ω_i  →  (B, D/2)
        x = t * self.inv_freq.unsqueeze(0)
        emb = torch.cat([x.sin(), x.cos()], dim=-1)  # (B, D)
        return emb                                   # (B, D)


def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # From https://github.com/yang-song/score_sde_pytorch/ which is from
    #  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    # assumes timesteps is in the range 0 to 1000
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
