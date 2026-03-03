"""
Phase 4 — Candidate chord generation via template matching.

For each melody segment's chroma vector, compare against binary 12-semitone
templates for all 749 Chordonomicon chord types using cosine similarity.
Return top-N candidates per segment.

Scoring notes (from project spec):
  - Penalize missing chord tones more than extra notes
  - Weight root and third more heavily than fifth
  - Allow tolerance for incomplete voicings
"""

import numpy as np


class ChordTemplate:
    """Binary 12-semitone vector for a chord type."""
    def __init__(self, name: str, semitones: np.ndarray):
        self.name = name
        self.semitones = semitones  # shape (12,), dtype bool or float


def load_templates(chordonomicon_script_output: str) -> list[ChordTemplate]:
    """
    Load chord templates from the Chordonomicon chord-to-semitone conversion output.
    Returns one ChordTemplate per unique chord type (749 total).
    """
    raise NotImplementedError


def score_chord(chroma: np.ndarray, template: ChordTemplate) -> float:
    """Cosine similarity between a chroma vector and a chord template."""
    denom = np.linalg.norm(chroma) * np.linalg.norm(template.semitones)
    if denom == 0:
        return 0.0
    return float(np.dot(chroma, template.semitones) / denom)


def get_candidates(
    chroma: np.ndarray,
    templates: list[ChordTemplate],
    top_n: int = 8,
) -> list[tuple[ChordTemplate, float]]:
    """
    Return top-N (chord_template, score) pairs for a single chroma vector,
    sorted by descending similarity score.
    """
    scored = [(t, score_chord(chroma, t)) for t in templates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]
