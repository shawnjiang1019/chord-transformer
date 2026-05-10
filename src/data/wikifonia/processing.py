"""
Wikifonia MusicXML (.mxl) preprocessing pipeline.

Extracts paired (melody_chroma, chord_label) data from Wikifonia lead sheets
for VAE training. Chord symbols are explicit in the file — no inference needed.

Pipeline:
  1. parse_file()          - Load .mxl with music21, extract score
  2. extract_melody()      - Get Note objects from first part
  3. extract_chord_map()   - Build beat_offset -> chord_string mapping
  4. compute_beat_chroma() - 12-dim chroma per beat from melody notes
  5. align_pairs()         - Pair each beat's chroma with its chord label
  6. tokenize_chords()     - Convert chord strings to [root, quality, voicing] IDs
  7. process_file()        - Full pipeline for one file
  8. process_dataset()     - Batch processing with manifest output
"""

import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from music21 import converter, note, harmony, stream

from src.data.tokenizer import ChordTokenizer, parse_chord

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────
MIN_BEATS = 8                   # reject files shorter than this
MAX_AMBIGUOUS_RATIO = 0.5       # reject if >50% beats have no chord
METRICAL_WEIGHTS = {0: 1.0, 1: 0.5, 2: 1.0, 3: 0.5}  # strong/weak beats


# ── Data Classes ───────────────────────────────────────────────────

@dataclass
class WikifoniaResult:
    """Result of processing a single .mxl file."""
    filepath: str
    title: str
    melody_chroma: np.ndarray
    chord_labels: list[str]
    chord_token_ids: np.ndarray
    n_beats: int
    time_signature: str


# ── Stage 1: File Loading ─────────────────────────────────────────

def parse_file(filepath: str | Path):
    """
    Load a .mxl file with music21 and return the Score object.
    Returns None if parsing fails.

    music21 handles .mxl (compressed) and .xml (uncompressed) transparently.
    """
    try:
        score = converter.parse(str(filepath))
        return score
    except Exception:
        return None


# ── Stage 2: Melody Extraction ────────────────────────────────────

def extract_melody(score) -> list:
    """
    Extract Note objects from the first part (melody line).

    Returns a list of music21 Note objects, each with:
        note.pitch.pitchClass   -> int 0-11 (C=0, Cs=1, ..., B=11)
        note.offset             -> float beat position in the score
        note.quarterLength      -> float duration in quarter notes

    ChordSymbols are mixed into part.flatten().notes alongside real notes,
    so we filter to only note.Note instances (skips rests and chord symbols).
    """
    return [
        n for n in score.parts[0].flatten().notes
        if isinstance(n, note.Note)
    ]


# ── Stage 3: Chord Symbol Extraction ─────────────────────────────

def extract_chord_map(score) -> dict[float, str]:
    """
    Build a mapping of beat_offset -> chord_string from ChordSymbol objects.

    music21 ChordSymbol has:
        cs.offset   -> float beat position
        cs.figure   -> str chord name e.g. "Cm7", "F7", "Bbmaj7"

    The map is used to find the active chord at any beat:
        active chord at beat t = last chord with offset <= t

    Returns dict {offset: chord_string} sorted by offset.
    """
    chord_map = {}
    for n in score.parts[0].flatten().notes:
        if isinstance(n, harmony.ChordSymbol):
            chord_map[float(n.offset)] = n.figure
    return dict(sorted(chord_map.items()))


def get_chord_at_beat(chord_map: dict[float, str], beat_offset: float) -> str | None:
    """
    Return the active chord at beat_offset — the last chord symbol
    whose offset is <= beat_offset. Returns None if none precede it.
    """
    active = None
    for offset, chord_str in chord_map.items():
        if offset <= beat_offset:
            active = chord_str
        else:
            break
    return active


# ── Stage 4: Chroma Computation ───────────────────────────────────

def get_beat_grid(score) -> list[float]:
    """
    Return a list of beat offsets (in quarter notes) for the score.

    Uses the time signature to step through the score measure by measure.
    e.g. 4/4 at 120 BPM -> [0.0, 1.0, 2.0, 3.0, 4.0, ...]

    Returns list of float offsets, one per beat.
    """
    ts = score.flatten().getElementsByClass('TimeSignature')[0]
    beat_dur = float(ts.beatDuration.quarterLength)

    measures = list(score.parts[0].getElementsByClass('Measure'))
    if not measures:
        return []

    last = measures[-1]
    total_length = float(last.offset) + float(last.quarterLength)

    beats = []
    offset = 0.0
    while offset < total_length - 1e-6:
        beats.append(round(offset, 6))
        offset += beat_dur

    return beats


def compute_beat_chroma(notes: list, beat_offsets: list[float]) -> np.ndarray:
    """
    Compute a 12-dim chroma vector for each beat window.

    For each window [beat_offsets[i], beat_offsets[i+1]):
        - Collect notes whose offset falls within the window
        - Weight each note by: overlap_duration * metrical_weight
        - Accumulate into 12-dim chroma (indexed by pitchClass)
        - L2-normalize

    Returns np.ndarray of shape (n_beats, 12).
    """
    n_beats = len(beat_offsets)
    chroma = np.zeros((n_beats, 12), dtype=np.float32)

    for i in range(n_beats):
        t_start = beat_offsets[i]
        t_end = beat_offsets[i + 1] if i + 1 < n_beats else t_start + 1.0
        metrical_weight = METRICAL_WEIGHTS.get(i % 4, 0.5)

        for n in notes:
            n_start = float(n.offset)
            n_end = n_start + float(n.quarterLength)
            if n_start >= t_end or n_end <= t_start:
                continue
            overlap = min(n_end, t_end) - max(n_start, t_start)
            chroma[i, n.pitch.pitchClass] += overlap * metrical_weight

        norm = np.linalg.norm(chroma[i])
        if norm > 0:
            chroma[i] /= norm

    return chroma


# ── Stage 5: Alignment ────────────────────────────────────────────

def align_pairs(
    melody_chroma: np.ndarray,
    beat_offsets: list[float],
    chord_map: dict[float, str],
) -> tuple[np.ndarray, list[str]] | None:
    """
    Pair each beat's chroma vector with its active chord label.

    Drops beats where no chord is active (None from get_chord_at_beat).
    Returns None if fewer than MIN_BEATS valid pairs remain or
    more than MAX_AMBIGUOUS_RATIO beats have no chord.

    Returns:
        melody_out:  (n_valid, 12)
        labels_out:  list of chord strings, length n_valid
    """
    labels = [get_chord_at_beat(chord_map, b) for b in beat_offsets]

    n_total = len(labels)
    n_ambiguous = sum(1 for l in labels if l is None)

    if n_ambiguous / max(n_total, 1) > MAX_AMBIGUOUS_RATIO:
        return None

    valid = [(melody_chroma[i], labels[i]) for i in range(n_total) if labels[i] is not None]

    if len(valid) < MIN_BEATS:
        return None

    melody_out = np.stack([v[0] for v in valid])
    labels_out = [v[1] for v in valid]

    return melody_out, labels_out


# ── Stage 6: Tokenization ─────────────────────────────────────────

def tokenize_chords(chord_labels: list[str],
                    tokenizer: ChordTokenizer) -> np.ndarray:
    """
    Convert chord strings to [root_id, quality_id, voicing_id] token IDs.

    music21 figures use flat notation (Bb, Eb) which parse_chord() normalizes
    via the ENHARMONIC map (Bb -> As, Eb -> Ds).

    Returns (n_beats, 3) int32 array.
    """
    chord_ids = np.zeros((len(chord_labels), 3), dtype=np.int32)
    for i, label in enumerate(chord_labels):
        root, quality, voicing = parse_chord(label)
        chord_ids[i, 0] = tokenizer.token2id.get(root, tokenizer.token2id["[UNK]"])
        chord_ids[i, 1] = tokenizer.token2id.get(quality, tokenizer.token2id["[UNK]"])
        chord_ids[i, 2] = tokenizer.token2id.get(voicing, tokenizer.token2id["[UNK]"])
    return chord_ids


# ── Stage 7: Full Pipeline ────────────────────────────────────────

def process_file(filepath: str | Path,
                 tokenizer: ChordTokenizer) -> WikifoniaResult | None:
    """Full pipeline for one .mxl file. Returns None on failure."""
    filepath = Path(filepath)

    score = parse_file(filepath)
    if score is None or not score.parts:
        return None

    melody_notes = extract_melody(score)
    if not melody_notes:
        return None

    chord_map = extract_chord_map(score)
    if not chord_map:
        return None

    beat_offsets = get_beat_grid(score)
    if len(beat_offsets) < MIN_BEATS:
        return None

    melody_chroma = compute_beat_chroma(melody_notes, beat_offsets)

    aligned = align_pairs(melody_chroma, beat_offsets, chord_map)
    if aligned is None:
        return None
    melody_out, labels_out = aligned

    chord_ids = tokenize_chords(labels_out, tokenizer)

    try:
        ts = score.flatten().getElementsByClass('TimeSignature')[0]
        time_sig = ts.ratioString
    except Exception:
        time_sig = "4/4"

    return WikifoniaResult(
        filepath=str(filepath),
        title=filepath.stem,
        melody_chroma=melody_out.astype(np.float32),
        chord_labels=labels_out,
        chord_token_ids=chord_ids,
        n_beats=len(labels_out),
        time_signature=time_sig,
    )


# ── Stage 8: Batch Processing ─────────────────────────────────────

def process_dataset(input_dir: str | Path,
                    output_dir: str | Path,
                    tokenizer: ChordTokenizer) -> dict:
    """
    Process all .mxl files in input_dir, save results to output_dir.

    Each result saved as {stem}.npz with melody_chroma and chord_ids.
    Writes manifest.json with per-file metadata.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.mxl"))
    total = len(files)
    print(f"Found {total} .mxl files in {input_dir}")

    manifest = []
    counts = {"total": total, "saved": 0, "rejected_parse": 0,
              "rejected_short": 0, "rejected_no_chords": 0}

    for i, filepath in enumerate(files):
        result = process_file(filepath, tokenizer)

        if result is None:
            counts["rejected_parse"] += 1
            continue

        npz_name = filepath.stem + ".npz"
        np.savez_compressed(
            output_dir / npz_name,
            melody_chroma=result.melody_chroma,
            chord_ids=result.chord_token_ids,
        )

        manifest.append({
            "filename": npz_name,
            "title": result.title,
            "n_beats": result.n_beats,
            "time_signature": result.time_signature,
        })
        counts["saved"] += 1

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total} processed (saved={counts['saved']})")

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Saved {counts['saved']}/{total} files to {output_dir}")
    return counts
