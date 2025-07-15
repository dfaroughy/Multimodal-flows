import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from utils.tensorclass import TensorMultiModal
from utils.models import LayerNorm, SelfAttention, CrossAttention, transformer_timestep_embedding

"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""


class ParticleFormer(nn.Module):

    def __init__(self, config):
        """
        config.vocab_size should include a mask token 
        """
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.transformer = nn.ModuleDict(dict(
            wxe = nn.Linear(config.dim_continuous, config.n_embd),
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            blocks_kin = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            blocks_flv = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            blocks_fused = nn.ModuleList([Block(config, fused=True) for _ in range(config.n_layer_fused)]),
            ln_fused= LayerNorm(2 * config.n_embd, bias=config.bias),
            ln_kin_last  = LayerNorm(config.n_embd, bias=config.bias),
            ln_flv_last  = LayerNorm(config.n_embd, bias=config.bias),
            head_kin = nn.Sequential(nn.Linear(config.n_embd, config.n_inner, bias=config.bias),
                                     nn.GELU(),
                                     nn.Linear(config.n_inner, config.dim_continuous, bias=config.bias)),
            head_flv = nn.Sequential(nn.Linear(config.n_embd, config.n_inner, bias=config.bias), 
                                      nn.GELU(),
                                      nn.Linear(config.n_inner, config.vocab_size, bias=config.bias)) # classifier head for discrete tokens
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
        flv_emb = self.transformer.wte(state.discrete.squeeze(-1))                         # token embeddings of shape (B, D, n_embd)
        time_emb = transformer_timestep_embedding(state.time, self.n_embd)     # time embedding of shape (B, n_embd)
        time_emb = time_emb.unsqueeze(1)                                       # (B, 1, n_embd)

        k = self.transformer.drop(kin_emb + time_emb)
        f = self.transformer.drop(flv_emb + time_emb)

        f_skip = f.clone()  # skip connection for final layer norm  
        k_skip = k.clone()  # skip connection for final layer norm

        # encoder blocks

        for block in self.transformer.blocks_kin:
            k = block(k, attn_mask=attn_mask)
            k += time_emb

        for block in self.transformer.blocks_flv:
            f = block(f, attn_mask=attn_mask)
            f += time_emb

        h = torch.cat((k + k_skip, f + f_skip), dim=-1)  # concatenate the continuous and discrete embeddings
        h = self.transformer.ln_fused(h)  

        for block in self.transformer.blocks_fused:
            h = block(h, attn_mask=attn_mask)

        k, f = h.split((self.n_embd, self.n_embd), dim=-1)  # split the concatenated embeddings back into continuous and discrete parts

        k = self.transformer.ln_kin_last(k + k_skip + time_emb)  # final layer norm for continuous features
        f = self.transformer.ln_flv_last(f + f_skip + time_emb)  # final layer norm for discrete tokens

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

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, state: TensorMultiModal) -> torch.Tensor:
        """
            state.continuous is the corrupted state (B, D, 1) or (B, D, vocab_size) if onehot is True
            state.time is the time in the corruption process (B,)
        """
        
        B, D = state.shape
        inputs = state.discrete if state.has_discrete else state.continuous

        pos = torch.arange(0, D, dtype=torch.long, device=state.time.device)    # shape (D)
        tok_emb = self.transformer.wte(inputs)                                  # token embeddings of shape (B, D, n_embd)
        pos_emb = self.transformer.wpe(pos)                                     # position embeddings of shape (D, n_embd)
        time_emb = transformer_timestep_embedding(state.time, self.n_embd)      # time embedding of shape (B, n_embd)

        h = self.transformer.drop(tok_emb.view(B, D, self.n_embd) + pos_emb.view(1, D, self.n_embd) + time_emb.view(B, 1, self.n_embd))

        for block in self.transformer.blocks:
            h = block(h, attn_mask=None)
            h += pos_emb.view(1, D, self.n_embd) + time_emb.view(B, 1, self.n_embd)

        h = self.transformer.ln_f(h)

        return self.transformer.head(h)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class MLP(nn.Module):
    def __init__(self, config, scale):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd * scale, config.n_inner, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.n_inner , config.n_embd  * scale, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, fused=None):
        super().__init__()

        scale = 2 if fused is not None else 1

        self.ln_1 = LayerNorm(config.n_embd * scale, bias=config.bias)
        self.attn = SelfAttention(config, scale)
        self.ln_2 = LayerNorm(config.n_embd * scale, bias=config.bias)
        self.mlp = MLP(config, scale)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class CrossBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CrossAttention(config)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, z, attn_mask=None):
        x = x + self.attn(self.ln_1(x), self.ln_2(z),  attn_mask=attn_mask)
        x = x + self.mlp(self.ln_3(x))
        return x
