import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from utils.tensorclass import TensorMultiModal
from utils.models import LayerNorm, SelfAttention, transformer_timestep_embedding

"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

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
        inputs = state.continuous if state.has_continuous else state.discrete

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
