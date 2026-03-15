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
from src.model.TransformerBlock import TransformerBlock


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
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.size()
        pos = torch.arange(T, device=input_ids.device)

        # now each token has its embedding info, position info
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        # after each block it gets info about previous tokens and processed understanding
        x = self.ln_f(x)
        # compute the dot product of each ID against each token
        logits = self.head(x) 
        return logits
        

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=64, temperature=1.0, top_k=50, eos_id=2):
        ids = prompt_ids

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            ids_cond = ids[:, -self.pos_emb.num_embeddings:]

            # Forward pass, get logits for last position only
            logits = self(ids_cond)[:, -1, :]

            # Apply temperature (higher = more random)
            logits = logits / temperature

            # Top-k: zero out everything except top k scores
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float('-inf')

            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            # Append and continue
            ids = torch.cat([ids, next_id], dim=1)

            # Stop if EOS
            if next_id.item() == eos_id:
                break

        return ids
