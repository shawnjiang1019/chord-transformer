"""
Data augmentation via chromatic transposition.

The Chordonomicon repo provides a transposition script; this wraps it
to produce all 12 transpositions of each progression, effectively
multiplying training data by 12x and making the model key-agnostic.
"""

import re

from src.data.vocab.roots import ROOT_TOKENS, ENHARMONIC

SEMITONE_STEPS = list(range(12))  # 0 = no change, 1 = up one semitone, etc.

_ROOT_RE = re.compile(r'^([A-G](?:s(?!us)|b)?)(.*)$')


def transpose_chord(chord: str, semitones: int) -> str:
    """Transpose a single chord string by N semitones.

    Section markers and other non-chord tokens are returned unchanged.
    The root is shifted chromatically; the suffix (quality/voicing/bass) is preserved.

    Examples:
        transpose_chord("C", 2)      -> "D"
        transpose_chord("Amin", 3)   -> "Cmin"
        transpose_chord("Fs7", 1)    -> "G7"
        transpose_chord("Bbadd9", 5) -> "Dsadd9"
    """
    # Pass through section markers unchanged
    if chord.startswith("<") and chord.endswith(">"):
        return chord

    m = _ROOT_RE.match(chord)
    if not m:
        return chord

    raw_root, suffix = m.group(1), m.group(2)

    # Normalize enharmonic spelling to canonical sharp form
    root = ENHARMONIC.get(raw_root, raw_root)

    # Find index in chromatic scale, transpose, wrap around
    idx = ROOT_TOKENS.index(root)
    new_idx = (idx + semitones) % 12
    new_root = ROOT_TOKENS[new_idx]

    return new_root + suffix


def augment_sequence(sequence: list[str]) -> list[list[str]]:
    """Return all 12 transpositions of a chord sequence.

    Takes a list of chord strings (e.g. from splitting a raw chord string)
    and returns 12 lists, one per semitone shift (0 = original).
    """
    return [
        [transpose_chord(c, n) for c in sequence]
        for n in SEMITONE_STEPS
    ]
