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
        self.sep_id = config.sep_id

        self.chord_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

    def forward(
        self,
        past_chord_ids: torch.Tensor,
        future_chord_ids: torch.Tensor,
        past_len: torch.Tensor = None,
        future_len: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns (B, P + 1 + F, d_model)."""
        B, P, _ = past_chord_ids.shape
        F_len   = future_chord_ids.size(1)
        device  = past_chord_ids.device

        # Step 1 embed each chord beat as sum of root + quality + voicing embeddings
        past_emb   = self.chord_emb(past_chord_ids).sum(dim=2)
        future_emb = self.chord_emb(future_chord_ids).sum(dim=2)

        # Step 2 SEP token
        sep_ids = torch.full((B, 1), self.sep_id, device=device, dtype=torch.long)
        sep_emb = self.chord_emb(sep_ids)                          # (B, 1, d_model)

        # Step 3 concatenate [past | SEP | future]
        x = torch.cat([past_emb, sep_emb, future_emb], dim=1)     # (B, P+1+F, d_model)

        # Step 4 positional embeddings
        T = x.size(1)
        pos = torch.arange(T, device=device)
        x = x + self.pos_emb(pos)

        # Step 5 padding mask (True = ignore)
        padding_mask = None
        if past_len is not None and future_len is not None:
            padding_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            for b in range(B):
                if past_len[b] < P:
                    padding_mask[b, past_len[b]:P] = True
                if future_len[b] < F_len:
                    padding_mask[b, P + 1 + future_len[b]:] = True

        # Step 6 bidirectional transformer
        return self.transformer(x, src_key_padding_mask=padding_mask)


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

        self.bos_id = 1

    def forward(
        self,
        past_chord_ids: torch.Tensor,
        future_chord_ids: torch.Tensor,
        melody_chroma: torch.Tensor,
        target_chord_ids: torch.Tensor,
        past_len: torch.Tensor = None,
        future_len: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass for training.
        Decoder input is target_chord_ids shifted right by one (BOS prepended).
        Returns logits of shape (B, W*3, vocab_size).
        """
        device = target_chord_ids.device
        encoder_out = self.encoder(past_chord_ids, future_chord_ids, past_len, future_len)
        chroma_exp = melody_chroma.repeat_interleave(3, dim = 1)
        chroma_emb = self.chroma_proj(chroma_exp)

        B, W, _ = target_chord_ids.shape
        flat = target_chord_ids.reshape(B, W * 3)
        bos  = torch.full((B, 1), self.bos_id, device=device, dtype=torch.long)
        decoder_input = torch.cat([bos, flat[:, :-1]], dim=1)

        logits = self.decoder(decoder_input, encoder_out=encoder_out, chroma_emb=chroma_emb)

        return logits
    


    @torch.no_grad()
    def generate(
        self,
        past_chord_ids: torch.Tensor,
        future_chord_ids: torch.Tensor,
        melody_chroma: torch.Tensor,
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
        B, W, _ = melody_chroma.shape
        device = melody_chroma.device

        # Encode once — context is fixed for all generation steps
        encoder_out = self.encoder(past_chord_ids, future_chord_ids, past_len, future_len)

        # Pre-compute full chroma embeddings for all W*3 token positions
        chroma_exp = melody_chroma.repeat_interleave(3, dim=1)  # (B, W*3, 12)
        chroma_emb = self.chroma_proj(chroma_exp)               # (B, W*3, d_model)

        # Start with BOS
        ids = torch.full((B, 1), self.bos_id, device=device, dtype=torch.long)

        for i in range(W * 3):
            # Slice chroma to match current sequence length
            logits = self.decoder(ids, encoder_out=encoder_out, chroma_emb=chroma_emb[:, :ids.size(1), :])
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_vals[:, -1:]] = float('-inf')

            probs   = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # (B, 1)
            ids     = torch.cat([ids, next_id], dim=1)

        # Strip BOS, reshape to (B, W, 3)
        return ids[:, 1:].reshape(B, W, 3)

    def load_pretrained_decoder(self, checkpoint_path: str, device: str = "cpu"):
        """
        Load weights from a saved ChordTransformer checkpoint into the decoder.
        Cross-attention layers are not in the checkpoint and stay randomly initialized.
        Logs how many weights were transferred.
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)

        decoder_state = self.decoder.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if k in decoder_state and v.shape == decoder_state[k].shape
        }
        decoder_state.update(compatible)
        self.decoder.load_state_dict(decoder_state)
        print(f"Loaded {len(compatible)}/{len(decoder_state)} decoder weights from {checkpoint_path}")
