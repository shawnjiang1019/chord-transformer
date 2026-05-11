"""
Chord Harmonizer — melody + chord context -> chord infilling.

Given:
    - past chord progression  (before the melody section)
    - future chord progression (after the melody section)
    - melody chroma vectors    (beat-level) for the gap

Predicts chord IDs that fit the melody and bridge past -> future.

Architecture:
    Encoder: bidirectional transformer over (past chords | SEP | future chords)
             Each chord beat = embed(root) + embed(quality) + embed(voicing)
    Decoder: ChordTransformer with cross-attention + melody chroma injection

Input shapes (batch B, past length P, future length F, gap width W):
    past_chord_ids   : (B, P, 3)   long
    future_chord_ids : (B, F, 3)   long
    melody_chroma    : (B, W, 12)  float
    target_chord_ids : (B, W, 3)   long  — used during training only

Output:
    logits           : (B, W*3, vocab_size)  — one logit vector per token position
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from src.model.transformer import ChordTransformer


@dataclass
class HarmonizerConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 512
    dropout: float = 0.1
    sep_id: int = 2   # [EOS] token used as separator between past and future


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Bidirectional transformer over past and future chord sequences.

    Each chord beat is embedded as a single vector:
        chord_emb(root) + chord_emb(quality) + chord_emb(voicing)

    Sequence layout passed to the transformer:
        [past_0 ... past_P | SEP | future_0 ... future_F]

    Returns encoder_out: (B, P + 1 + F, d_model)
    """

    def __init__(self, config: HarmonizerConfig):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        past_chord_ids: torch.Tensor,    # (B, P, 3)
        future_chord_ids: torch.Tensor,  # (B, F, 3)
        past_len: torch.Tensor = None,   # (B,) real lengths before padding
        future_len: torch.Tensor = None, # (B,) real lengths before padding
    ) -> torch.Tensor:
        """Returns (B, P + 1 + F, d_model)."""
        raise NotImplementedError


# ── Full Harmonizer ────────────────────────────────────────────────────────────

class ChordHarmonizer(nn.Module):
    """
    Full harmonizer: bidirectional encoder + melody-conditioned autoregressive decoder.

    Training (teacher forcing):
        forward(past, future, melody_chroma, target) -> logits (B, W*3, vocab_size)

    Inference:
        generate(past, future, melody_chroma) -> chord_ids (B, W, 3)
    """

    def __init__(self, config: HarmonizerConfig):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)

        # Decoder: existing ChordTransformer with cross-attention enabled
        self.decoder = ChordTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            cross_attn=True,
        )

        # Projects 12-dim melody chroma into d_model, added to decoder token embeddings
        self.chroma_proj = nn.Linear(12, config.d_model)

        self.bos_id = 1  # [BOS]

    def forward(
        self,
        past_chord_ids: torch.Tensor,    # (B, P, 3)
        future_chord_ids: torch.Tensor,  # (B, F, 3)
        melody_chroma: torch.Tensor,     # (B, W, 12)
        target_chord_ids: torch.Tensor,  # (B, W, 3)
        past_len: torch.Tensor = None,
        future_len: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass for training.
        Decoder input is target_chord_ids shifted right by one (BOS prepended).
        Returns logits of shape (B, W*3, vocab_size).
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        past_chord_ids: torch.Tensor,    # (B, P, 3)
        future_chord_ids: torch.Tensor,  # (B, F, 3)
        melody_chroma: torch.Tensor,     # (B, W, 12)
        temperature: float = 1.0,
        top_k: int = 50,
        past_len: torch.Tensor = None,
        future_len: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate chord IDs for the melody section.
        Encodes past + future chord context once, then generates W*3 tokens.
        Returns chord_ids of shape (B, W, 3).
        """
        raise NotImplementedError

    def load_pretrained_decoder(self, checkpoint_path: str, device: str = "cpu"):
        """
        Load weights from a saved ChordTransformer checkpoint into the decoder.
        Cross-attention layers are not in the checkpoint and stay randomly initialized.
        Logs how many weights were transferred.
        """
        raise NotImplementedError
