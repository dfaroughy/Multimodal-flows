import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from tensorclass import TensorMultiModal

"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False
    """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if config.qk_layernorm:
            self.q_layernorm = LayerNorm(config.n_embd // self.n_head, bias=config.bias)
            self.k_layernorm = LayerNorm(config.n_embd // self.n_head, bias=config.bias)

        self.qk_layernorm = config.qk_layernorm

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

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            raise NotImplementedError
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.n_inner, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.n_inner , config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

# From https://github.com/yang-song/score_sde_pytorch/ which is from
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
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


class FlavorFormer(nn.Module):

    def __init__(self, config):
        """
        config.vocab_size should include a mask token 
        """
        super().__init__()

        self.n_embd = config.n_embd

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(config.vocab_size, config.n_embd) if config.onehot else nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.max_num_particles, config.n_embd),
            drop = nn.Dropout(config.dropout),
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
            head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # output head 
        ))

        self.transformer.wte.weight = self.transformer.head.weight # https://paperswithcode.com/method/weight-tying
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, state: TensorMultiModal) -> torch.Tensor:
        """
            state.continuous is the corrupted tokens (B, D)
            state.time is the time in the corruption process (B,)
        """
        B, D = state.shape
        inputs = state.continuous 
        attn_mask = state.mask.squeeze(-1) 
        time = state.time.squeeze(-1) if state.time.dim() > 1 else state.time 

        pos = torch.arange(0, D, dtype=torch.long, device=state.continuous.device) # shape (D)
        tok_emb = self.transformer.wte(inputs) # token embeddings of shape (B, D, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (D, n_embd)
        time_emb = transformer_timestep_embedding(time.squeeze(-1), self.n_embd)  # time embedding of shape (B, n_embd)

        h = self.transformer.drop(tok_emb.view(B, D, self.n_embd) + pos_emb.view(1, D, self.n_embd) + time_emb.view(B, 1, self.n_embd))

        for block in self.transformer.blocks:
            h = block(h, attn_mask=attn_mask.bool())

        h = self.transformer.ln_f(h)

        return self.transformer.head(h)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    

# class JetSeqGPT(nn.Module):

#     def __init__(self, config):
#         """
#         config.vocab_size should include a mask token 
#         """
#         super().__init__()

#         self.n_embd = config.n_embd

#         self.transformer = nn.ModuleDict(dict(
#             wte = nn.Embedding(config.vocab_size, config.n_embd) ,
#             wpe = nn.Embedding(config.max_num_particles, config.n_embd),
#             drop = nn.Dropout(config.dropout),
#             blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
#             ln_f = LayerNorm(config.n_embd, bias=config.bias),
#             head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # output head 
#         ))

#         # initialization:

#         self.transformer.wte.weight = self.transformer.head.weight # https://paperswithcode.com/method/weight-tying
#         self.apply(self._init_weights)

#         for pn, p in self.named_parameters():
#             if pn.endswith('c_proj.weight'):
#                 torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
  
#     def forward(self, input_ids, time, mask=None, attn_mask=None):
#         """
#             input_ids is the corrupted tokens (B, D)
#             time is the time in the corruption process (B,)
#         """
        
#         B, D = input_ids.size()
#         n_embd = self.n_embd
#         time = time.squeeze(-1) if time.dim() > 1 else time 

#         pos = torch.arange(0, D, dtype=torch.long, device=input_ids.device) # shape (D)
#         tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (B, D, n_embd)
#         pos_emb = self.transformer.wpe(pos) # position embeddings of shape (D, n_embd)
#         time_emb = transformer_timestep_embedding(time, n_embd)

#         h = self.transformer.drop(tok_emb.view(B, D, n_embd) + pos_emb.view(1, D, n_embd) + time_emb.view(B, 1, n_embd))

#         for block in self.transformer.blocks:
#             h = block(h, attn_mask=attn_mask)

#         h = self.transformer.ln_f(h)
#         logits = self.transformer.head(h)

#         return logits

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        