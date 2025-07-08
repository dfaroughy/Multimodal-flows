import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

# ───────────────────────────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────────────────────────
class ParticleTransformer(GPT2LMHeadModel):
    """
    GPT2LMHeadModel but
      1) no causal mask (fully permutation-invariant)
      2) adds a continuous-time embedding at the input
    """
    def __init__(self, config: GPT2Config, max_time_freq: float = 10.0):
        super().__init__(config)

        # remove the causal mask from HF Transformer:

        for block in self.transformer.h:
            block.attn.bias.data.fill_(1)

        self.time_embedding = TimeFourierEmbedding(config.n_embd, max_freq=max_time_freq)
        self.target_embedding = nn.Embedding(config.vocab_size, config.n_embd)  # k_1

    def forward(
        self,
        input_ids: torch.LongTensor = None,   # noisy tokens (k_t)
        labels: torch.LongTensor = None,      # clean target tokens (k_1)
        time: torch.Tensor = None,
        attention_mask: torch.LongTensor = None,
        **kwargs
    ):
        """
        Accepts:
          - input_ids             (B, D)
          - time                  (B,) or (B,1) with values in [0,1]
          - attention_mask        (B, D)
          - target labels         (B, D)
        """

        inputs_emb = self.transformer.wte(input_ids)  # (B, D, hdim)
        target_emb = self.target_embedding(labels) if labels else torch.zeros_like(inputs_emb)  # (B, D, hdim)
        time_emb = self.time_embedding(time)                            # (B, hdim)
        time_emb = time_emb.unsqueeze(1).expand(*inputs_emb.size())     # (B, D, hdim)
        
        return super().forward(
            inputs_embeds=inputs_emb + time_emb + target_emb, 
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
