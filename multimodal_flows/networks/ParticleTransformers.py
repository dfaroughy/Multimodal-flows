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
            wxe = nn.Linear(config.dim_continuous, config.n_embd),  # continuous emb
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # discrete emb
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
        flv_emb = self.transformer.wte(state.discrete.squeeze(-1))             # token embeddings of shape (B, D, n_embd)
        time_emb = transformer_timestep_embedding(state.time, self.n_embd)     # time embedding of shape (B, n_embd)
        time_emb = time_emb.unsqueeze(1)                                       # (B, 1, n_embd)

        k = self.transformer.drop(kin_emb + time_emb)
        f = self.transformer.drop(flv_emb + time_emb)

        f_skip = f.clone()  # skip connection for final layer norm  
        k_skip = k.clone()  # skip connection for final layer norm

        # mode encoder blocks

        for block in self.transformer.blocks_kin:
            k = block(k, attn_mask=attn_mask)
            k += time_emb

        for block in self.transformer.blocks_flv:
            f = block(f, attn_mask=attn_mask)
            f += time_emb

        h = torch.cat((k + k_skip, f + f_skip), dim=-1)  # concatenate the continuous and discrete embeddings
        h = self.transformer.ln_fused(h)  

        # fused blocks

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
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.vocab_size = config.vocab_size
        self.max_num_particles = config.max_num_particles

        self.transformer = nn.ModuleDict({
            'wte': nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd),
                                 nn.GELU(),
                                 nn.Linear(config.n_embd, config.n_embd)),
            'ln1': LayerNorm(config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'blocks': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln2': LayerNorm(config.n_embd),
            'head': nn.Sequential(nn.Linear(config.n_embd, config.n_inner),
                                  nn.GELU(),
                                  nn.Linear(config.n_inner, config.dim_continuous)),
        })

        if config.use_pos_emb: # Positional embeddings
            self.transformer['wpe'] = nn.Embedding(config.max_num_particles, config.n_embd)

        if config.use_pairwise: # Symmetric token interaction U
            self.transformer['wue'] = nn.Embedding((config.vocab_size * (config.vocab_size + 1)) // 2, config.n_embd)
            self.transformer['wue_proj'] = nn.Linear(config.n_embd, config.n_head)
            self.lambda_u = nn.Parameter(torch.tensor(0.0))

        self.apply(self._init_weights)

    def forward(self, state: TensorMultiModal) -> torch.Tensor:

        attn_mask = state.mask.clone()
        attn_mask = state.mask.bool().squeeze()                   # (B, D) 
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)           # (B, 1, 1, D)
        attn_mask = attn_mask & attn_mask.transpose(-1, -2)       # (B, 1, D, D)
        attn_mask = attn_mask.expand(-1, self.n_head, -1, -1)     # (B, n_heads, D, D)

        tokens = state.discrete.squeeze(-1)

        # Initial embeddings

        tok_emb = self.transformer.wte(tokens)                              # (B, D, n_embd)
        tok_emb = self.transformer.ln1(tok_emb)
        time_emb = transformer_timestep_embedding(state.time, self.n_embd)  # (B, n_embd)
        time_emb = time_emb.unsqueeze(1)                                    # (B, 1, n_embd)
        
        if hasattr(self.transformer, 'wpe'):
            pos = torch.arange(0, self.max_num_particles, dtype=torch.long, device=state.time.device)    # shape (D)
            pos_emb = self.transformer.wpe(pos)  # (D, n_embd)
            tok_emb += pos_emb.view(1, self.max_num_particles , self.n_embd)

        if hasattr(self.transformer, 'wue'):
            U_emb = self.token_interactions_emb(tokens)  
            attn_mask = attn_mask + self.lambda_u * U_emb   
            
        # transformer blocks

        f = self.transformer.drop(tok_emb + time_emb)
        f_skip = f.clone()

        for block in self.transformer.blocks:
            f = block(f, attn_mask=attn_mask)
            f += time_emb

        f = self.transformer.ln2(f + f_skip)

        return self.transformer.head(f)

    def get_attention_mask(self, state):
        attn_mask = state.mask.bool().squeeze(-1)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        attn_mask = attn_mask & attn_mask.transpose(-1, -2)
        return attn_mask.float().expand(-1, self.n_head, -1, -1)

    def token_interactions_emb(self, tokens):
        """ pairwise interactions for the tokens.
        """
        i, j = tokens.unsqueeze(2), tokens.unsqueeze(1)        # (B, D, 1), (B, 1, D)
        min_tok = torch.minimum(i, j)
        max_tok = torch.maximum(i, j)
        U  = (max_tok * (max_tok + 1)) // 2 + min_tok          # triangle-number encoding  
        U_emb = self.transformer.wue(U)                        # (B, D, D, n_embd)
        U_emb = self.transformer.wue_proj(U_emb)               # (B, D, D, n_head)
        return U_emb.permute(0, 3, 1, 2)                       # (B, n_head, D, D)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)      


class KinFormer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.max_num_particles = config.max_num_particles
        self.mu = torch.tensor(config.metadata['mean'])
        self.sig = torch.tensor(config.metadata['std'])

        self.transformer = nn.ModuleDict({
            'wxe': nn.Sequential(nn.Linear(config.dim_continuous, config.n_embd, bias=config.bias),
                                 nn.GELU(),
                                 nn.Linear(config.n_embd, config.n_embd, bias=config.bias)),
            'ln1': LayerNorm(config.n_embd, bias=config.bias),
            'drop': nn.Dropout(config.dropout),
            'blocks': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln2': LayerNorm(config.n_embd, bias=config.bias),
            'head': nn.Sequential(nn.Linear(config.n_embd, config.n_inner, bias=config.bias),
                                  nn.GELU(),
                                  nn.Linear(config.n_inner, config.dim_continuous, bias=config.bias)),
        })

        if config.use_pos_emb: # Positional embeddings 
            self.transformer['wpe'] = nn.Embedding(config.max_num_particles, config.n_embd)

        if config.use_pairwise: # particle interactions with (deltaR, k_t) lund observables
            self.transformer['wue'] = nn.Sequential(nn.Linear(2, config.n_embd),
                                                    nn.GELU(),
                                                    nn.LayerNorm(config.n_embd))

            self.transformer['wue_proj'] = nn.Sequential(nn.Linear(config.n_embd, config.n_embd, bias=config.bias),
                                                         nn.GELU(),
                                                         nn.Linear(config.n_embd, config.n_head, bias=config.bias)
                                                         )
            self.lambda_u = nn.Parameter(torch.tensor(0.0))

        self.apply(self._init_weights)

    def forward(self, state: TensorMultiModal) -> torch.Tensor:

        attn_mask = state.mask.clone()
        attn_mask = state.mask.bool().squeeze()                   # (B, D) 
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)           # (B, 1, 1, D)
        attn_mask = attn_mask & attn_mask.transpose(-1, -2)       # (B, 1, D, D)
        attn_mask = attn_mask.expand(-1, self.n_head, -1, -1)     # (B, n_heads, D, D)

        # Initial embeddings

        x_emb = self.transformer.wxe(state.continuous)                      # feature embeddings of shape (B, D, n_embd)
        x_emb = self.transformer.ln1(x_emb)
        time_emb = transformer_timestep_embedding(state.time, self.n_embd)  # (B, n_embd)
        time_emb = time_emb.unsqueeze(1)                                    # (B, 1, n_embd)

        if hasattr(self.transformer, 'wpe'):
            pos = torch.arange(0, self.max_num_particles, dtype=torch.long, device=state.time.device)    # shape (D)
            pos_emb = self.transformer.wpe(pos)  # (D, n_embd)
            x_emb += pos_emb.view(1, self.max_num_particles, self.n_embd)

        if hasattr(self.transformer, 'wue'):
            U_emb = self.particle_interactions_emb(state)  
            attn_mask = attn_mask + self.lambda_u * U_emb   
            
        # transformer blocks

        x = self.transformer.drop(x_emb + time_emb)
        x_skip = x.clone()

        for block in self.transformer.blocks:
            x = block(x, attn_mask=attn_mask)
            x += time_emb

        x = self.transformer.ln2(x + x_skip)

        return self.transformer.head(x)

    def particle_interactions_emb(self, kin):
        
        U = lund_observables(kin, self.mu, self.sig)   # (B, D, D, 2) 
        U_emb = self.transformer.wue(U)                    # (B, D, D, n_embd)
        U_emb = 0.5 * (U_emb + U_emb.transpose(1, 2))      # symmetrize
        
        U_emb = self.transformer.wue_proj(U_emb)           # (B, D, D, n_head)
        U_emb = 0.5 * (U_emb + U_emb.transpose(1, 2))      # symmetrize

        return U_emb.permute(0, 3, 1, 2)                   # (B, n_head, D, D)

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


def lund_observables(state, mu=1.0, sig=1.0):

    dim = state.continuous.size(-1)
    kin = state.continuous.clone()  # (B, D, 3)
    kin = kin * sig.view(1,1,dim).to(kin.device) + mu.view(1,1,dim).to(kin.device) # destandardize 
    kin *= state.mask
    
    pt_i, pt_j = kin[..., 0].unsqueeze(2), kin[..., 0].unsqueeze(1)         # (B, D, 1), (B, 1, D)
    eta_i, eta_j = kin[..., 1].unsqueeze(2), kin[..., 1].unsqueeze(1)
    phi_i, phi_j = kin[..., 2].unsqueeze(2), kin[..., 2].unsqueeze(1)

    # pairwise observables (B, D, D)

    deta = eta_i - eta_j
    dphi = torch.remainder(phi_i - phi_j + torch.pi, 2 * torch.pi) - torch.pi  
    dR = torch.sqrt(deta**2 + dphi**2)  # deltaR
    log_dR = torch.log(dR) 
    log_kt = torch.log(torch.minimum(pt_i, pt_j) * dR**2 / (pt_i * pt_j)  + 1e-8)
    lund = [log_kt, log_dR]
    # log_z = torch.log(z + 1e-8)  
    # log_dR = torch.log(torch.sqrt(deta ** 2 + dphi ** 2 + 1e-8)) 
    # log_psi = torch.log(torch.abs(torch.arctan2(delta_eta, delta_phi) ) + 1e-8)  
    # log_kt = torch.log(z * (deta ** 2 + dphi ** 2) + 1e-8)
    # log_m2 = torch.log(2 * pt_i * pt_j * (torch.cosh(deta) - torch.cos(dphi)) + 1e-8) 

    # pairwise interaction tensor (B, D, D, 2)
    U = torch.stack([log_kt, log_dR], dim=-1) 
    U = (U - U.mean(dim=-1, keepdim=True)) / (U.std(dim=-1, keepdim=True) + 1e-8)
    return U