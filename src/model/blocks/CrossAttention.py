"""
Cross-attention for the ChordHarmonizer decoder.

Queries come from the decoder hidden states.
Keys and values come from the encoder output (past + future chord context).
No causal mask — every decoder position attends to every encoder position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, T_dec, d_model) — decoder hidden states (queries)
        encoder_out: (B, T_enc, d_model) — encoder output (keys + values)
        Returns:     (B, T_dec, d_model)
        """
        raise NotImplementedError
