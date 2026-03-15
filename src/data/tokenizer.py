"""
Chord vocabulary and tokenizer — decomposed token scheme.

Each chord is represented as THREE consecutive tokens: [root, quality_family, voicing]
  "Amin"  -> [A_root, min_quality,  none_voicing]
  "Fs7"   -> [Fs_root, dom7_quality, none_voicing]
  "Cadd9" -> [C_root,  maj_quality,  add9_voicing]

Single tokens for: special markers, structure sections, decades, genres.

Sequence format:
  <genre> <section> [BOS] root qual voicing root qual voicing ... [EOS]
"""

import re

from src.data.vocab.special import SPECIAL_TOKENS, STRUCTURE_TOKENS, DECADE_TOKENS, GENRE_TOKENS
from src.data.vocab.roots import ROOT_TOKENS, ENHARMONIC
from src.data.vocab.qualities import QUALITY_TOKENS, QUALITY_MAP, QUALITY_TO_SUFFIX
from src.data.vocab.voicings import VOICING_TOKENS

_ROOT_RE = re.compile(r'^([A-G](?:s(?!us)|b)?)(.*)$')

# Maps raw suffix -> (quality_family, voicing)
# Built from QUALITY_MAP (voicing="none") + voicing-specific overrides
CHORD_MAP: dict[str, tuple[str, str]] = {
    suffix: (quality, "none") for suffix, quality in QUALITY_MAP.items()
}
CHORD_MAP.update({
    "no3d":     ("maj",  "no3d"),
    "add9":     ("maj",  "add9"),
    "add13":    ("maj",  "add13"),
    "add11":    ("maj",  "add11"),
    "7sus4":    ("dom7", "7sus4"),
    "7sus2":    ("dom7", "7sus2"),
    "minadd13": ("min",  "minadd13"),
    "minadd9":  ("min",  "minadd9"),
    "minmaj7":  ("min",  "minmaj7"),
    "7b9":      ("dom7", "7b9"),
    "maj7sus2": ("maj7", "maj7sus2"),
    "majs9":    ("maj",  "majs9"),
    "augmaj7":  ("aug",  "augmaj7"),
    "maj13":    ("maj7", "maj13"),
    "minadd11": ("min",  "minadd11"),
    "maj911s":  ("maj7", "maj911s"),
    "maj7sus4": ("maj7", "maj7sus4"),
    "maj11":    ("maj7", "maj11"),
    "13b":      ("dom7", "13b"),
    "augmaj9":  ("aug",  "augmaj9"),
})


def parse_chord(chord: str) -> tuple[str, str, str]:
    """Parse a raw chord string into (root, quality, voicing).

    Examples:
        "C"       -> ("C",  "maj",  "none")
        "Amin"    -> ("A",  "min",  "none")
        "Fs7"     -> ("Fs", "dom7", "none")
        "Cadd9"   -> ("C",  "maj",  "add9")
        "Bbmin7"  -> ("As", "min7", "none")
    """
    m = _ROOT_RE.match(chord)
    if not m:
        return ("[UNK]", "other", "other_voicing")

    raw_root, suffix = m.group(1), m.group(2)
    suffix = suffix.split("/")[0]  # strip bass note: min/G -> min

    # Normalize enharmonic spelling: Bb -> As, Db -> Cs
    root = ENHARMONIC.get(raw_root, raw_root)

    # Look up (quality, voicing) from unified map
    quality, voicing = CHORD_MAP.get(suffix, ("other", "other_voicing"))

    return (root, quality, voicing)


class ChordTokenizer:
    """Maps chord strings to integer token IDs and back.

    Chords are decomposed into three tokens each: [root, quality_family, voicing].
    All other tokens (special, structure, decade, genre) are single tokens.
    """

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._root_ids: set[int] = set()
        self._quality_ids: set[int] = set()
        self._voicing_ids: set[int] = set()
        self.build_vocab()

    def build_vocab(self):
        """Assign integer IDs to every token from the fixed constant lists."""
        idx = 0

        # 1. Special tokens ([PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3)
        for tok in SPECIAL_TOKENS:
            self.token2id[tok] = idx
            idx += 1

        # 2. Structure tokens (<intro>, <verse>, ...)
        for tok in STRUCTURE_TOKENS:
            self.token2id[tok] = idx
            idx += 1

        # 3. Decade tokens (<1890s>, <1900s>, ...)
        for tok in DECADE_TOKENS:
            self.token2id[tok] = idx
            idx += 1

        # 4. Genre tokens (<pop>, <rock>, ...)
        for tok in GENRE_TOKENS:
            self.token2id[tok] = idx
            idx += 1

        # 5. Root tokens (C, Cs, D, ...)
        for tok in ROOT_TOKENS:
            self.token2id[tok] = idx
            self._root_ids.add(idx)
            idx += 1

        # 6. Quality tokens (maj, min, dom7, ...)
        for tok in QUALITY_TOKENS:
            self.token2id[tok] = idx
            self._quality_ids.add(idx)
            idx += 1

        # 7. Voicing tokens (none, no3d, add9, ...)
        for tok in VOICING_TOKENS:
            self.token2id[tok] = idx
            self._voicing_ids.add(idx)
            idx += 1

        # Build reverse mapping
        self.id2token = {v: k for k, v in self.token2id.items()}

    def encode(self, chord_string: str) -> list[int]:
        """Convert a raw chord string from the dataset into token IDs.

        Input:  "<intro_1> C <verse_1> F C E7 Amin"
        Output: [<intro>, [BOS], C, maj, none, F, maj, none, C, dom7, none, A, min, none, [EOS]]
                (as integer IDs)
        """
        ids = []
        tokens = chord_string.split()

        for t in tokens:
            # Section marker: <verse_1> -> <verse>
            if t.startswith("<") and t.endswith(">"):
                section = re.sub(r'_\d+>', '>', t)
                if section in self.token2id:
                    ids.append(self.token2id[section])
            else:
                # Chord: decompose into 3 tokens
                root, quality, voicing = parse_chord(t)
                ids.append(self.token2id.get(root, self.token2id["[UNK]"]))
                ids.append(self.token2id.get(quality, self.token2id["[UNK]"]))
                ids.append(self.token2id.get(voicing, self.token2id["[UNK]"]))

        # Wrap with [BOS] and [EOS]
        ids = [self.token2id["[BOS]"]] + ids + [self.token2id["[EOS]"]]
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        """Convert token IDs back to human-readable strings.

        Groups consecutive (root, quality, voicing) triplets back into
        chord strings like "Amin7" or "Cadd9". Special/structure/genre/decade
        tokens pass through as-is. [BOS], [EOS], [PAD] are skipped.
        """
        result = []
        i = 0
        skip = {"[BOS]", "[EOS]", "[PAD]"}

        while i < len(ids):
            tok = self.id2token.get(ids[i], "[UNK]")

            if tok in skip:
                i += 1
                continue

            # Root token -> start of a chord triplet
            if ids[i] in self._root_ids and i + 2 < len(ids):
                root = tok
                quality = self.id2token.get(ids[i + 1], "other")
                voicing = self.id2token.get(ids[i + 2], "none")

                # Build chord string: root + quality suffix + voicing suffix
                suffix = QUALITY_TO_SUFFIX.get(quality, "?")
                if voicing != "none":
                    suffix = voicing  # voicing replaces the basic suffix
                result.append(root + suffix)
                i += 3
            else:
                # Single token (section, genre, decade, [UNK])
                result.append(tok)
                i += 1

        return result

    def save(self, path: str):
        """Save the vocabulary to a JSON file."""
        import json
        data = {
            "token2id": self.token2id,
            "root_ids": sorted(self._root_ids),
            "quality_ids": sorted(self._quality_ids),
            "voicing_ids": sorted(self._voicing_ids),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load the vocabulary from a JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.id2token = {int(v): k for k, v in self.token2id.items()}
        self._root_ids = set(data["root_ids"])
        self._quality_ids = set(data["quality_ids"])
        self._voicing_ids = set(data["voicing_ids"])

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)
