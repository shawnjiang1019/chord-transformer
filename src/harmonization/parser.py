"""
Phase 4 — Melody parsing and beat-level segmentation.

Tools: music21 (MusicXML), mido / pretty_midi (MIDI)

Output per segment: 12-dimensional chroma vector weighted by:
  - Note duration  (longer notes are more likely chord tones)
  - Metrical position (beat 1 & 3 in 4/4 are harmonically stronger)
  - Repetition (frequently occurring pitch classes)
"""

import numpy as np
from pathlib import Path


class MelodySegment:
    """One beat-level segment of a melody."""
    def __init__(self, beat: int, chroma: np.ndarray):
        self.beat = beat
        self.chroma = chroma  # shape (12,), float weights per pitch class


def parse_musicxml(filepath: str | Path) -> list[MelodySegment]:
    """Parse a MusicXML file into beat-level MelodySegments using music21."""
    raise NotImplementedError


def parse_midi(filepath: str | Path) -> list[MelodySegment]:
    """Parse a MIDI file into beat-level MelodySegments using mido/pretty_midi."""
    raise NotImplementedError


def compute_chroma(
    notes: list,         # list of (pitch_class, duration, beat_position)
    beats_per_measure: int = 4,
) -> np.ndarray:
    """
    Compute a weighted 12-dim chroma vector from the notes in a segment.
    Weighting: duration * metrical_weight, summed per pitch class, then L2-normalized.
    """
    chroma = np.zeros(12, dtype=float)
    # TODO: fill in weighting logic
    norm = np.linalg.norm(chroma)
    return chroma / norm if norm > 0 else chroma
