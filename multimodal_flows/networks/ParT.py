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

        # remove the causal mask from Transformer
        for block in self.transformer.h:
            block.attn.bias.data.fill_(1)

        self.time_emb = TimeFourierEmbedding(config.n_embd, max_freq=max_time_freq)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        time: torch.Tensor        = None,
        attention_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        **kwargs
    ):
        """
        Accepts:
          - input_ids      (B, seq_len)
          - time           (B,) or (B,1) with values in [0,1]
          - attention_mask (B, seq_len)
          - labels         (B, seq_len), if doing LM training
        """
        inputs_embeds = self.transformer.wte(input_ids)  # (B, seq, D)
        t_emb = self.time_emb(time)                      # (B, D)
        B, seq, D = inputs_embeds.size()
        t_emb = t_emb.unsqueeze(1).expand(B, seq, D)     # (B, seq, D)
        inputs_embeds = inputs_embeds + t_emb
        

        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
