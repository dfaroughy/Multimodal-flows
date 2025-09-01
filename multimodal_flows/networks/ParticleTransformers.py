import torch
import torch.nn as nn

from utils.tensorclass import TensorMultiModal
from utils.models import LayerNorm, transformer_timestep_embedding
from networks.attention import SelfAttnBlock

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
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.max_num_particles = config.max_num_particles
        self.mu = torch.tensor(config.metadata['mean'])
        self.sig = torch.tensor(config.metadata['std'])

        self.transformer = nn.ModuleDict(dict(
            wxe = nn.Sequential(nn.Linear(config.dim_continuous, config.n_embd), 
                                 nn.GELU(),
                                 nn.Linear(config.n_embd, config.n_embd // 2)),
            wye = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), 
                                nn.GELU(),
                                nn.Linear(config.n_embd, config.n_embd // 2)),                    
            ln1_x = LayerNorm(config.n_embd // 2),
            ln1_y = LayerNorm(config.n_embd // 2),
            drop_x = nn.Dropout(config.dropout),
            drop_y = nn.Dropout(config.dropout),
            blocks_x = nn.ModuleList([SelfAttnBlock(config, config.n_embd // 2) for _ in range(config.n_layer)]),
            blocks_y = nn.ModuleList([SelfAttnBlock(config, config.n_embd // 2) for _ in range(config.n_layer)]),
            ln2_x = LayerNorm(config.n_embd // 2),
            ln2_y = LayerNorm(config.n_embd // 2),
            drop = nn.Dropout(config.dropout),
            blocks_fuse = nn.ModuleList([SelfAttnBlock(config) for _ in range(config.n_layer_fused)]),
            time_expand = nn.Linear(config.n_embd // 2, config.n_embd),  # time embedding expansion
            ln3_x = LayerNorm(config.n_embd // 2),
            ln3_y = LayerNorm(config.n_embd // 2),
            head_x = nn.Sequential(nn.Linear(config.n_embd//2, config.n_inner),
                                    nn.GELU(),
                                    nn.Linear(config.n_inner, config.dim_continuous)),
            head_y = nn.Sequential(nn.Linear(config.n_embd//2, config.n_inner),
                                    nn.GELU(),
                                    nn.Linear(config.n_inner, config.vocab_size)),
        ))

        if config.use_coocurrence: # Symmetric token co-ocurrence matrix U
            self.transformer['wue'] = nn.Embedding((config.vocab_size * (config.vocab_size + 1)) // 2, config.n_embd)
            self.transformer['wue_proj'] = nn.Linear(config.n_embd, config.n_head)

        self.apply(self._init_weights)

    def forward(self, state: TensorMultiModal) -> torch.Tensor:

        attn_mask = state.mask.clone()
        attn_mask = state.mask.bool().squeeze()                   # (B, D) 
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)           # (B, 1, 1, D)
        attn_mask = attn_mask & attn_mask.transpose(-1, -2)       # (B, 1, D, D)
        attn_mask = attn_mask.expand(-1, self.n_head, -1, -1)     # (B, n_heads, D, D)

        if hasattr(self.transformer, 'wue'):
            U_emb = self.token_co_ocurrence_emb(state.discrete.squeeze(-1))  
            attn_mask = attn_mask + U_emb   

        # Initial embeddings


        time_emb = transformer_timestep_embedding(state.time, self.n_embd // 2)  # (B, n_embd)
        time_emb = time_emb.unsqueeze(1)                                    # (B, 1, n_embd)

        # continuous stream:

        x = self.transformer.wxe(state.continuous)                      # continuous (B, D, n_embd/2)
        x = self.transformer.ln1_x(x)                               # layer norm (B, D, n_embd/2)
        x = self.transformer.drop_x(x + time_emb)
        x_skip = x.clone()  # skip connection

        for block in self.transformer.blocks_x:
            x = block(x, attn_mask=attn_mask)
            x = x + time_emb 

        x = self.transformer.ln2_x(x + x_skip)

        # discrete stream:

        y = self.transformer.wye(state.discrete.squeeze(-1))            # discrete (B, D, n_embd/2)
        y = self.transformer.ln1_y(y)                               # layer norm (B, D, n_embd/2)
        y = self.transformer.drop_y(y + time_emb)
        y_skip = y.clone()  # skip connection  

        for block in self.transformer.blocks_y:
            y = block(y, attn_mask=attn_mask)
            y = y + time_emb 

        y = self.transformer.ln2_y(y + y_skip)

        # fused stream:

        z = torch.cat((x, y), dim=-1)                           # joint emb (B, D, n_embd)
        time_emb_2 = self.transformer.time_expand(time_emb)   
        z = self.transformer.drop(z + time_emb_2)

        for block in self.transformer.blocks_fuse:
            z = block(z, attn_mask=attn_mask)
            z = z + time_emb_2 
            
        # Final layer norm and heads

        x, y = z.split((self.n_embd // 2, self.n_embd // 2), dim=-1)  
        x = self.transformer.ln3_x(x + x_skip)
        y = self.transformer.ln3_y(y + y_skip)

        return self.transformer.head_x(x), self.transformer.head_y(y)  

    def token_co_ocurrence_emb(self, tokens):
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


class FusedParticleFormer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.max_num_particles = config.max_num_particles
        self.mu = torch.tensor(config.metadata['mean'])
        self.sig = torch.tensor(config.metadata['std'])

        self.transformer = nn.ModuleDict(dict(
            wxe = nn.Sequential(nn.Linear(config.dim_continuous, config.n_embd), 
                                 nn.GELU(),
                                 nn.Linear(config.n_embd, config.n_embd // 2)),
            wye = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), 
                                nn.GELU(),
                                nn.Linear(config.n_embd, config.n_embd // 2)),                    
            ln1_x = LayerNorm(config.n_embd // 2),
            ln1_y = LayerNorm(config.n_embd // 2),
            drop = nn.Dropout(config.dropout),
            blocks = nn.ModuleList([SelfAttnBlock(config) for _ in range(config.n_layer)]),
            ln2= LayerNorm(config.n_embd),
            head_x = nn.Sequential(nn.Linear(config.n_embd//2, config.n_inner),
                                    nn.GELU(),
                                    nn.Linear(config.n_inner, config.dim_continuous)),
            head_y = nn.Sequential(nn.Linear(config.n_embd//2, config.n_inner),
                                    nn.GELU(),
                                    nn.Linear(config.n_inner, config.vocab_size)),
        ))

        self.apply(self._init_weights)

    def forward(self, state: TensorMultiModal) -> torch.Tensor:

        attn_mask = state.mask.clone()
        attn_mask = state.mask.bool().squeeze()                   # (B, D) 
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)           # (B, 1, 1, D)
        attn_mask = attn_mask & attn_mask.transpose(-1, -2)       # (B, 1, D, D)
        attn_mask = attn_mask.expand(-1, self.n_head, -1, -1)     # (B, n_heads, D, D)

        # Initial embeddings

        x = self.transformer.wxe(state.continuous)                      # continuous (B, D, n_embd/2)
        x = self.transformer.ln1_x(x)                            # layer norm (B, D, n_embd/2)

        y = self.transformer.wye(state.discrete.squeeze(-1))            # discrete (B, D, n_embd/2)
        y = self.transformer.ln1_y(y)                            # layer norm (B, D, n_embd/2)

        z = torch.cat((x, y), dim=-1)                           # joint emb (B, D, n_embd)

        time_emb = transformer_timestep_embedding(state.time, self.n_embd)  # (B, n_embd)
        time_emb = time_emb.unsqueeze(1)                                    # (B, 1, n_embd)

        # transformer blocks

        z = self.transformer.drop(z + time_emb)
        z_skip = z.clone()  # skip connection for final layer norm  

        for block in self.transformer.blocks:
            z = block(z, attn_mask=attn_mask)
            z = z + time_emb 
            
        z = self.transformer.ln2(z + z_skip) 
        x, y = z.split((self.n_embd // 2, self.n_embd // 2), dim=-1)  

        return self.transformer.head_x(x), self.transformer.head_y(y)  

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

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd),
                                nn.GELU(),
                                nn.Linear(config.n_embd, config.n_embd)),
            ln1 = LayerNorm(config.n_embd),
            drop = nn.Dropout(config.dropout),
            blocks = nn.ModuleList([SelfAttnBlock(config) for _ in range(config.n_layer)]),
            ln2 = LayerNorm(config.n_embd),
            head = nn.Sequential(nn.Linear(config.n_embd, config.n_inner),
                                 nn.GELU(),
                                 nn.Linear(config.n_inner, config.vocab_size)),
        ))

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

        # Initial embeddings

        tok_emb = self.transformer.wte(state.discrete.squeeze(-1))                              # (B, D, n_embd)
        tok_emb = self.transformer.ln1(tok_emb)
        time_emb = transformer_timestep_embedding(state.time, self.n_embd)  # (B, n_embd)
        time_emb = time_emb.unsqueeze(1)                                    # (B, 1, n_embd)
        
        if hasattr(self.transformer, 'wpe'):
            pos = torch.arange(0, self.max_num_particles, dtype=torch.long, device=state.time.device)    # shape (D)
            pos_emb = self.transformer.wpe(pos)  # (D, n_embd)
            tok_emb += pos_emb.view(1, self.max_num_particles , self.n_embd)

        if hasattr(self.transformer, 'wue'):
            U_emb = self.token_interactions_emb(state.discrete.squeeze(-1))  
            attn_mask = attn_mask + self.lambda_u * U_emb   
            
        # transformer blocks

        f = self.transformer.drop(tok_emb + time_emb)
        f_skip = tok_emb.clone()

        for block in self.transformer.blocks:
            f = block(f, attn_mask=attn_mask)
            f = f + time_emb 

        f = self.transformer.ln2(f + f_skip)

        return self.transformer.head(f)


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
            'blocks': nn.ModuleList([SelfAttnBlock(config) for _ in range(config.n_layer)]),
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
            x_emb + x_emb + pos_emb.view(1, self.max_num_particles, self.n_embd)

        if hasattr(self.transformer, 'wue'):
            U_emb = self.particle_interactions_emb(state)  
            attn_mask = attn_mask + self.lambda_u * U_emb   
            
        # transformer blocks

        x = self.transformer.drop(x_emb + time_emb)
        x_skip = x.clone()

        for block in self.transformer.blocks:
            x = block(x, attn_mask=attn_mask)
            x = x + time_emb 
            
        x = self.transformer.ln2(x + x_skip)

        return self.transformer.head(x)

    def particle_interactions_emb(self, kin): # TODO fix
        
        U = lund_observables(kin, self.mu, self.sig)       # (B, D, D, 2) 
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
    U = torch.stack([log_kt, log_dR], dim=-1) 
    U = (U - U.mean(dim=-1, keepdim=True)) / (U.std(dim=-1, keepdim=True) + 1e-8)
    return U