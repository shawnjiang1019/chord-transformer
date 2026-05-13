"""
Translates between frontend chord format and the ChordTokenizer format.

Frontend uses:  C#, D#, F#, G#, A#   and quality labels like "7", "6", "sus4"
Tokenizer uses: Cs, Ds, Fs, Gs, As   and QUALITY_MAP / QUALITY_TO_SUFFIX

Encode path:
  ChordIn(root="C#", q="7")
    -> chord string "Cs7"
    -> tok.encode("<verse_1> Cs7 ...") -> [BOS, <verse>, Cs, dom7, none, ...]

Decode path:
  model output ids
    -> tok.decode(ids) -> ["C", "Amin", "Cs7", ...]
    -> parse_chord each -> (root_tok, quality, voicing)
    -> ChordOut(root="C#", q="7")
"""

from src.data.tokenizer import ChordTokenizer, parse_chord
from src.data.vocab.qualities import QUALITY_TO_SUFFIX
from app.backend.schemas.chord import ChordIn, ChordOut

# Bidirectional root maps between frontend (#) and tokenizer (s) notation
_ROOT_TO_TOKENIZER: dict[str, str] = {
    "C#": "Cs", "D#": "Ds", "F#": "Fs", "G#": "Gs", "A#": "As",
}
_ROOT_TO_FRONTEND: dict[str, str] = {v: k for k, v in _ROOT_TO_TOKENIZER.items()}


def _chord_to_string(root: str, q: str) -> str:
    """Build a tokenizer-compatible chord string from a frontend ChordIn."""
    tok_root = _ROOT_TO_TOKENIZER.get(root, root)
    return tok_root + q  # e.g. "Cs7", "Amin", "Fmaj7"


def encode_prompt(tok: ChordTokenizer, prompt: list[ChordIn], section: str) -> list[int]:
    """Encode a frontend prompt into token IDs ready for model.generate().

    Produces: [BOS] <section> root qual voicing ...
    (no trailing EOS — we want the model to continue generating)
    """
    section_tag = f"<{section}_1>"
    chord_strings = [_chord_to_string(c.root, c.q) for c in prompt]
    full_string = " ".join([section_tag] + chord_strings)

    ids = tok.encode(full_string)

    # Strip trailing [EOS] added by tok.encode — prompt is open-ended
    eos_id = tok.token2id["[EOS]"]
    if ids and ids[-1] == eos_id:
        ids = ids[:-1]

    return ids


def decode_output(tok: ChordTokenizer, ids: list[int]) -> list[ChordOut]:
    """Decode model output token IDs into frontend ChordOut objects.

    Skips special tokens (<section>, [UNK], etc.) and returns only chords.
    """
    strings = tok.decode(ids)
    chords: list[ChordOut] = []

    for s in strings:
        # Skip structure/special tokens
        if s.startswith("<") or s.startswith("["):
            continue

        root_tok, quality, _ = parse_chord(s)
        if root_tok == "[UNK]":
            continue

        root_fe = _ROOT_TO_FRONTEND.get(root_tok, root_tok)
        suffix = QUALITY_TO_SUFFIX.get(quality, quality)
        q_fe = suffix if suffix else "maj"  # empty suffix = major

        chords.append(ChordOut(root=root_fe, q=q_fe))

    return chords
