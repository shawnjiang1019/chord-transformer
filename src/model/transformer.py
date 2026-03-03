"""
Phase 2 — GPT-2 style decoder-only chord language model.

Suggested starting config (much smaller than NLP GPT-2):
  vocab_size : ~800  (749 chords + special tokens)
  n_layers   : 4-6
  n_heads    : 4-8
  d_model    : 256-512
  max_seq_len: 512   (median song is ~71 chords)

Conditioning tokens (genre, structure, decade) are prepended to the
sequence before the BOS token and handled by the same embedding table.
"""

import torch
import torch.nn as nn


class ChordTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        x = self.transformer(x, x, tgt_mask=causal_mask)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """Autoregressive generation with temperature + top-K sampling."""
        raise NotImplementedError
