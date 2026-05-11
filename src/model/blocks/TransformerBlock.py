import torch
import torch.nn as nn
from src.model.blocks.CausalSelfAttention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, cross_attn: bool = False):
        """
        Project into a wider token space,
        apply activation function,
        second layer gets back to original token size,
        regularize.

        cross_attn: if True, adds a CrossAttention layer between self-attention and MLP.
        When encoder_out is None in forward(), the cross-attention is skipped.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)

        if cross_attn:
            from src.model.blocks.CrossAttention import CrossAttention
            self.ln_cross = nn.LayerNorm(d_model)
            self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        else:
            self.ln_cross = None
            self.cross_attn = None

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if self.cross_attn is not None and encoder_out is not None:
            x = x + self.cross_attn(self.ln_cross(x), encoder_out)
        x = x + self.mlp(self.ln2(x))
        return x