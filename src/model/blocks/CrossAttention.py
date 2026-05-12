"""
Cross-attention for the ChordHarmonizer decoder.

Queries come from the decoder hidden states.
Keys and values come from the encoder output (past + future chord context).
No causal mask, every decoder position attends to every encoder position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, T_dec, d_model) — decoder hidden states (queries)
        encoder_out: (B, T_enc, d_model) — encoder output (keys + values)
        Returns:     (B, T_dec, d_model)
        """

        B, T_dec, C = x.size()
        T_enc = encoder_out.size(1)

        q = self.q_proj(x)
        k = self.k_proj(encoder_out)
        v = self.v_proj(encoder_out)

        q = q.reshape(B, T_dec, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T_enc, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T_enc, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)

        # skip the causal mask

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v

        concat = out.transpose(1, 2).reshape(B, T_dec, C)
        out = self.resid_dropout(self.out_proj(concat))
        
        return out