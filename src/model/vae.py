"""
Melody-conditioned Chord VAE.

Architecture:
    Encoder:  (melody_chroma, chord_ids) -> (mu, log_var)
    Sample:   z = mu + exp(0.5 * log_var) * epsilon
    Decoder:  (z, melody_chroma) -> chord_logits

The melody chroma acts as conditioning context for both encoder and decoder.
The latent variable z captures harmonic style — variation not determined by
the melody alone (e.g. dense vs sparse voicings, jazzy vs simple).

Input shapes (all with batch dimension B, window W):
    melody_chroma : (B, W, 12)   float — beat-level pitch class weights
    chord_ids     : (B, W, 3)    long  — [root_id, quality_id, voicing_id]

Output:
    chord_logits  : (B, W, 3, vocab_size)  — one distribution per token slot
    mu            : (B, latent_dim)
    log_var       : (B, latent_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class VAEConfig:
    vocab_size: int         # from ChordTokenizer.vocab_size (~88)
    d_model: int = 128      # internal hidden size
    latent_dim: int = 64    # size of latent vector z
    n_heads: int = 4        # attention heads in encoder/decoder
    n_layers: int = 2       # transformer layers in encoder/decoder
    window_size: int = 32   # beats per training window
    dropout: float = 0.1


# ── Encoder ───────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Encodes (melody_chroma, chord_ids) into (mu, log_var).

    Melody chroma is projected to d_model.
    Chord IDs are embedded and summed across root/quality/voicing slots.
    Both are concatenated and passed through transformer layers.
    A pooled representation is projected to mu and log_var.
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        melody_chroma: torch.Tensor,  # (B, W, 12)
        chord_ids: torch.Tensor,      # (B, W, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_var), each (B, latent_dim)."""
        raise NotImplementedError


# ── Reparameterization ────────────────────────────────────────────

def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Sample z = mu + std * epsilon using the reparameterization trick.
    epsilon ~ N(0, I), std = exp(0.5 * log_var).

    At inference with no gradient, pass mu directly (set log_var to zeros).
    Returns z of shape (B, latent_dim).
    """
    raise NotImplementedError


# ── Decoder ───────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Decodes (z, melody_chroma) into chord token logits.

    z is projected and broadcast to (B, W, d_model).
    Melody chroma is projected to (B, W, d_model).
    Both are summed and passed through transformer layers.
    Output is projected to three separate heads:
        root_logits    (B, W, vocab_size)
        quality_logits (B, W, vocab_size)
        voicing_logits (B, W, vocab_size)
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        z: torch.Tensor,              # (B, latent_dim)
        melody_chroma: torch.Tensor,  # (B, W, 12)
    ) -> torch.Tensor:
        """
        Returns chord_logits of shape (B, W, 3, vocab_size).
        Dim 2 indexes [root, quality, voicing].
        """
        raise NotImplementedError


# ── Full VAE ──────────────────────────────────────────────────────

class ChordVAE(nn.Module):
    """
    Full melody-conditioned chord VAE.

    During training:   encode -> reparameterize -> decode
    During inference:  sample z ~ N(0, I) -> decode with melody context
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(
        self,
        melody_chroma: torch.Tensor,  # (B, W, 12)
        chord_ids: torch.Tensor,      # (B, W, 3)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Returns:
            chord_logits : (B, W, 3, vocab_size)
            mu           : (B, latent_dim)
            log_var      : (B, latent_dim)
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        melody_chroma: torch.Tensor,  # (B, W, 12)
        temperature: float = 1.0,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate chord token IDs from melody context.

        If z is None, samples from N(0, I).
        If z is provided, uses it directly (for interpolation / style control).

        Returns chord_ids of shape (B, W, 3).
        """
        raise NotImplementedError


# ── Loss ──────────────────────────────────────────────────────────

def vae_loss(
    chord_logits: torch.Tensor,   # (B, W, 3, vocab_size)
    chord_ids: torch.Tensor,      # (B, W, 3)
    mu: torch.Tensor,             # (B, latent_dim)
    log_var: torch.Tensor,        # (B, latent_dim)
    pad_id: int,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss = reconstruction loss + kl_weight * KL divergence.

    Reconstruction loss: cross-entropy over [root, quality, voicing] slots,
    ignoring pad positions.

    KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    Averaged over batch. kl_weight is annealed from 0 -> 1 during training.

    Returns:
        total_loss  : scalar
        recon_loss  : scalar (for logging)
        kl_loss     : scalar (for logging)
    """
    raise NotImplementedError
