"""
Phase 5 — Sequence decoding: Viterbi / beam search.

Combines two scores at each timestep:
  local_score      : template similarity (melodic fit) from candidates.py
  transition_score : language model probability (harmonic coherence) from the transformer

Final score = alpha * local_score + (1 - alpha) * transition_score

`alpha` is the main hyperparameter to tune:
  alpha = 1.0  → pure template matching (ignores harmonic flow)
  alpha = 0.0  → pure language model (ignores melody)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DecoderConfig:
    alpha: float = 0.5       # balance between melodic fit and harmonic coherence
    beam_width: int = 5      # number of beams for beam search
    top_n_candidates: int = 8  # candidate chords per segment (from Stage 2)


def viterbi_decode(
    candidate_scores: list[list[tuple[str, float]]],  # per segment: [(chord, local_score)]
    lm_log_probs_fn,                                   # callable: (history) -> {chord: log_prob}
    config: DecoderConfig = DecoderConfig(),
) -> list[str]:
    """
    Viterbi decoding over the segment sequence.

    Args:
        candidate_scores : list of (chord, local_score) lists, one per segment
        lm_log_probs_fn  : function that takes a chord history and returns
                           a dict of {chord: log_prob} for next-chord distribution
        config           : decoder hyperparameters

    Returns:
        Best chord sequence as a list of chord strings.
    """
    raise NotImplementedError


def beam_search_decode(
    candidate_scores: list[list[tuple[str, float]]],
    lm_log_probs_fn,
    config: DecoderConfig = DecoderConfig(),
) -> list[list[str]]:
    """
    Beam search decoding. Returns `config.beam_width` candidate sequences,
    best-first. Caller picks the top-1 or presents alternatives.
    """
    raise NotImplementedError
