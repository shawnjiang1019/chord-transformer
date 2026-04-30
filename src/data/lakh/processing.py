"""
Lakh MIDI Dataset preprocessing pipeline.

Extracts paired (melody_chroma, chord_label) data from raw MIDI files
for VAE training. Uses pretty_midi for MIDI parsing.

Pipeline stages:
  1. load_midi()         - Parse MIDI, reject malformed files
  2. select_melody()     - Heuristic melody track selection
  3. extract_melody_chroma() / extract_accompaniment_chroma()
  4. recognize_chords()  - Template-match accompaniment to chord labels
  5. align_pairs()       - Align melody chroma with chord labels
  6. process_file()      - Full pipeline for one MIDI file
  7. process_dataset()   - Batch processing with quality filtering
"""

import json
import logging
import numpy as np
import pretty_midi
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from pathlib import Path

from src.data.tokenizer import ChordTokenizer, parse_chord
from src.data.vocab.roots import ROOT_TOKENS
from src.data.vocab.qualities import QUALITY_TOKENS
from src.harmonization.candidates import ChordTemplate, score_chord, get_candidates

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────
PERCUSSION_CHANNEL = 9
MIN_BEATS = 8
MIN_MELODY_NOTES = 16
CHORD_SIMILARITY_THRESHOLD = 0.65
MAX_AMBIGUOUS_RATIO = 0.5
MELODY_PITCH_CENTER = 72  # C5
MELODY_PITCH_SIGMA = 12   # 1 octave std dev
FILE_TIMEOUT_SECONDS = 10

# Intervals for each quality family (semitones above root)
QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "dim7": [0, 3, 6, 9],
    "aug":  [0, 4, 8],
    "sus":  [0, 5, 7],
    "dom9": [0, 4, 7, 10, 2],   # 14 % 12 = 2
    "maj9": [0, 4, 7, 11, 2],
    "min9": [0, 3, 7, 10, 2],
    "maj6": [0, 4, 7, 9],
    "min6": [0, 3, 7, 9],
}

# Metrical weights: beats 0 and 2 (strong) = 1.0, beats 1 and 3 (weak) = 0.5
METRICAL_WEIGHTS = {0: 1.0, 1: 0.5, 2: 1.0, 3: 0.5}


# ── Data Classes ───────────────────────────────────────────────────

@dataclass
class TrackScore:
    """Score components for melody track selection."""
    track_idx: int
    monophonicity: float
    pitch_centrality: float
    density_score: float
    range_score: float

    @property
    def composite(self) -> float:
        return (0.40 * self.monophonicity +
                0.25 * self.pitch_centrality +
                0.20 * self.density_score +
                0.15 * self.range_score)


@dataclass
class MidiExtractionResult:
    """Result of processing a single MIDI file."""
    filepath: str
    melody_chroma: np.ndarray       # (n_beats, 12)
    chord_labels: list[str]         # length n_beats
    chord_token_ids: np.ndarray     # (n_beats, 3)
    beats: np.ndarray               # (n_beats,)
    tempo: float
    melody_track_idx: int
    melody_track_score: float


# ── Stage 1: MIDI Loading ─────────────────────────────────────────

def load_midi(filepath: str | Path) -> pretty_midi.PrettyMIDI | None:
    """Safely load a MIDI file. Returns None if parsing fails or file is invalid."""
    try:
        midi = pretty_midi.PrettyMIDI(str(filepath))
    except Exception:
        return None

    # Must have at least 2 instruments (melody + accompaniment)
    non_drum = [inst for inst in midi.instruments if not inst.is_drum]
    if len(non_drum) < 2:
        return None

    # Must have beats and be long enough
    try:
        beats = midi.get_beats()
    except Exception:
        return None

    if len(beats) < MIN_BEATS:
        return None

    return midi


# ── Stage 2: Melody Track Selection ───────────────────────────────

def _compute_monophonicity(instrument: pretty_midi.Instrument,
                           beats: np.ndarray) -> float:
    """Fraction of beat windows where only 1 note is sounding."""
    n_beats = len(beats) - 1
    if n_beats <= 0:
        return 0.0

    mono_count = 0
    for i in range(n_beats):
        t_start, t_end = beats[i], beats[i + 1]
        # Count notes overlapping this beat window
        active = sum(
            1 for n in instrument.notes
            if n.start < t_end and n.end > t_start
        )
        if active == 1:
            mono_count += 1

    return mono_count / n_beats


def _compute_pitch_centrality(instrument: pretty_midi.Instrument) -> float:
    """Gaussian-weighted score: how close mean pitch is to expected melody range."""
    if not instrument.notes:
        return 0.0
    mean_pitch = np.mean([n.pitch for n in instrument.notes])
    return float(np.exp(-0.5 * ((mean_pitch - MELODY_PITCH_CENTER) / MELODY_PITCH_SIGMA) ** 2))


def _compute_density_score(instrument: pretty_midi.Instrument,
                           duration: float) -> float:
    """Score note density. Target: 1-4 notes per beat at ~120 BPM (~2 notes/sec)."""
    if duration <= 0:
        return 0.0
    notes_per_sec = len(instrument.notes) / duration
    # Optimal range: 1-4 notes/sec; Gaussian centered at 2.5
    return float(np.exp(-0.5 * ((notes_per_sec - 2.5) / 1.5) ** 2))


def _compute_range_score(instrument: pretty_midi.Instrument) -> float:
    """Score pitch range. Target: 12-24 semitones (1-2 octaves)."""
    if not instrument.notes:
        return 0.0
    pitches = [n.pitch for n in instrument.notes]
    pitch_range = max(pitches) - min(pitches)
    if pitch_range < 5:
        return 0.1  # drone
    if pitch_range > 36:
        return 0.1  # multi-part
    # Optimal: 12-24; peak at 18
    return float(np.exp(-0.5 * ((pitch_range - 18) / 8) ** 2))


def score_track(instrument: pretty_midi.Instrument,
                beats: np.ndarray,
                duration: float,
                track_idx: int) -> TrackScore | None:
    """Compute composite melody score for one track."""
    # Skip percussion
    if instrument.is_drum:
        return None
    # Skip sound effects (GM programs 112-127)
    if instrument.program >= 112:
        return None
    # Need enough notes
    if len(instrument.notes) < MIN_MELODY_NOTES:
        return None

    return TrackScore(
        track_idx=track_idx,
        monophonicity=_compute_monophonicity(instrument, beats),
        pitch_centrality=_compute_pitch_centrality(instrument),
        density_score=_compute_density_score(instrument, duration),
        range_score=_compute_range_score(instrument),
    )


def select_melody(midi: pretty_midi.PrettyMIDI) -> tuple[int, float] | None:
    """Select the most likely melody track. Returns (track_index, score) or None."""
    beats = midi.get_beats()
    duration = midi.get_end_time()

    scores = []
    for i, inst in enumerate(midi.instruments):
        ts = score_track(inst, beats, duration, i)
        if ts is not None:
            scores.append(ts)

    if not scores:
        return None

    best = max(scores, key=lambda s: s.composite)
    if best.composite < 0.3:
        return None

    return (best.track_idx, best.composite)


# ── Stage 3: Chroma Extraction ────────────────────────────────────

def _extract_chroma(instrument_or_instruments, beats: np.ndarray,
                    is_list: bool = False) -> np.ndarray:
    """
    Compute beat-level 12-dim chroma vectors from instrument note(s).

    For each beat window [beats[i], beats[i+1]):
      - Collect overlapping notes
      - Weight by duration_in_window * metrical_weight
      - Accumulate into 12-dim chroma, L2-normalize
    """
    n_beats = len(beats) - 1
    chroma = np.zeros((n_beats, 12), dtype=np.float32)

    # Gather all notes from one or multiple instruments
    if is_list:
        all_notes = []
        for inst in instrument_or_instruments:
            if not inst.is_drum:
                all_notes.extend(inst.notes)
    else:
        all_notes = instrument_or_instruments.notes

    for i in range(n_beats):
        t_start, t_end = beats[i], beats[i + 1]
        beat_in_measure = i % 4
        metrical_weight = METRICAL_WEIGHTS.get(beat_in_measure, 0.5)

        for note in all_notes:
            if note.start >= t_end or note.end <= t_start:
                continue
            # Duration of note within this beat window
            overlap = min(note.end, t_end) - max(note.start, t_start)
            pitch_class = note.pitch % 12
            chroma[i, pitch_class] += overlap * metrical_weight

        # L2 normalize
        norm = np.linalg.norm(chroma[i])
        if norm > 0:
            chroma[i] /= norm

    return chroma


def extract_melody_chroma(instrument: pretty_midi.Instrument,
                          beats: np.ndarray) -> np.ndarray:
    """Compute beat-level chroma from a melody track. Returns (n_beats, 12)."""
    return _extract_chroma(instrument, beats, is_list=False)


def extract_accompaniment_chroma(midi: pretty_midi.PrettyMIDI,
                                 melody_track_idx: int,
                                 beats: np.ndarray) -> np.ndarray:
    """Compute beat-level chroma from all non-melody, non-drum tracks. Returns (n_beats, 12)."""
    accomp = [inst for i, inst in enumerate(midi.instruments)
              if i != melody_track_idx and not inst.is_drum]
    return _extract_chroma(accomp, beats, is_list=True)


# ── Stage 4: Chord Recognition ────────────────────────────────────

def build_chord_templates() -> list[ChordTemplate]:
    """
    Generate ChordTemplate objects for all (root, quality) combinations.
    12 roots x 14 qualities = 168 templates.
    """
    templates = []
    for root_idx, root_name in enumerate(ROOT_TOKENS):
        for quality_name, intervals in QUALITY_INTERVALS.items():
            semitones = np.zeros(12, dtype=np.float32)
            for interval in intervals:
                semitones[(root_idx + interval) % 12] = 1.0
            # Build chord name matching tokenizer convention
            from src.data.vocab.qualities import QUALITY_TO_SUFFIX
            suffix = QUALITY_TO_SUFFIX.get(quality_name, quality_name)
            chord_name = root_name + suffix
            templates.append(ChordTemplate(name=chord_name, semitones=semitones))
    return templates


def recognize_chords(accompaniment_chroma: np.ndarray,
                     templates: list[ChordTemplate],
                     threshold: float = CHORD_SIMILARITY_THRESHOLD) -> list[str]:
    """
    For each beat, match accompaniment chroma to best chord template.
    Returns chord name string or "N" for ambiguous beats.
    """
    labels = []
    for i in range(len(accompaniment_chroma)):
        chroma = accompaniment_chroma[i]
        if np.linalg.norm(chroma) == 0:
            labels.append("N")
            continue
        candidates = get_candidates(chroma, templates, top_n=1)
        if candidates and candidates[0][1] >= threshold:
            labels.append(candidates[0][0].name)
        else:
            labels.append("N")
    return labels


# ── Stage 5: Alignment & Tokenization ────────────────────────────

def align_pairs(melody_chroma: np.ndarray,
                chord_labels: list[str],
                tokenizer: ChordTokenizer
                ) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """
    Align melody chroma with chord labels, dropping "N" beats.

    Returns:
        melody_out:  (n_valid, 12) melody chroma for valid beats
        chords_out:  (n_valid, 3) chord token IDs [root_id, quality_id, voicing_id]
        labels_out:  list of chord strings
    Or None if too few valid beats or too many ambiguous.
    """
    n_total = len(chord_labels)
    n_ambiguous = sum(1 for c in chord_labels if c == "N")

    if n_ambiguous / max(n_total, 1) > MAX_AMBIGUOUS_RATIO:
        return None

    valid_indices = [i for i, c in enumerate(chord_labels) if c != "N"]
    if len(valid_indices) < MIN_BEATS:
        return None

    melody_out = melody_chroma[valid_indices]
    labels_out = [chord_labels[i] for i in valid_indices]

    # Tokenize chord labels
    chords_out = np.zeros((len(labels_out), 3), dtype=np.int32)
    for i, label in enumerate(labels_out):
        root, quality, voicing = parse_chord(label)
        chords_out[i, 0] = tokenizer.token2id.get(root, tokenizer.token2id["[UNK]"])
        chords_out[i, 1] = tokenizer.token2id.get(quality, tokenizer.token2id["[UNK]"])
        chords_out[i, 2] = tokenizer.token2id.get(voicing, tokenizer.token2id["[UNK]"])

    return melody_out, chords_out, labels_out


# ── Stage 6: Full Pipeline ────────────────────────────────────────

def process_file(filepath: str | Path,
                 tokenizer: ChordTokenizer,
                 templates: list[ChordTemplate]) -> MidiExtractionResult | None:
    """Full pipeline for one MIDI file. Returns result or None on failure."""
    filepath = Path(filepath)

    # Stage 1: Load
    midi = load_midi(filepath)
    if midi is None:
        return None

    # Stage 2: Select melody
    result = select_melody(midi)
    if result is None:
        return None
    track_idx, track_score = result

    # Get beat times
    beats = midi.get_beats()
    if len(beats) < 2:
        return None

    # Stage 3: Extract chroma
    melody_chroma = extract_melody_chroma(midi.instruments[track_idx], beats)
    accomp_chroma = extract_accompaniment_chroma(midi, track_idx, beats)

    # Stage 4: Recognize chords
    chord_labels = recognize_chords(accomp_chroma, templates)

    # Stage 5: Align pairs
    aligned = align_pairs(melody_chroma, chord_labels, tokenizer)
    if aligned is None:
        return None
    melody_out, chord_ids, labels_out = aligned

    tempo = midi.estimate_tempo()

    return MidiExtractionResult(
        filepath=str(filepath),
        melody_chroma=melody_out,
        chord_labels=labels_out,
        chord_token_ids=chord_ids,
        beats=beats[:len(melody_out)],
        tempo=tempo,
        melody_track_idx=track_idx,
        melody_track_score=track_score,
    )


def _process_file_worker(args: tuple) -> dict | None:
    """Worker function for parallel processing. Returns serializable dict or None."""
    filepath, = args
    # Each worker creates its own tokenizer and templates (lightweight)
    tokenizer = ChordTokenizer()
    templates = build_chord_templates()

    try:
        result = process_file(filepath, tokenizer, templates)
    except Exception as e:
        logger.warning(f"Error processing {filepath}: {e}")
        return None

    if result is None:
        return None

    return {
        "filepath": result.filepath,
        "melody_chroma": result.melody_chroma,
        "chord_ids": result.chord_token_ids,
        "chord_labels": result.chord_labels,
        "tempo": result.tempo,
        "n_beats": len(result.chord_labels),
        "melody_track_idx": result.melody_track_idx,
        "melody_track_score": result.melody_track_score,
    }


def process_dataset(input_dir: str | Path,
                    output_dir: str | Path,
                    tokenizer: ChordTokenizer,
                    n_workers: int = 4,
                    max_files: int | None = None) -> dict:
    """
    Process all MIDI files in input_dir, save results to output_dir.

    Saves each result as {song_id}.npz and writes a manifest.json.
    Returns summary dict with processing counts.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all .mid files
    midi_files = sorted(input_dir.rglob("*.mid"))
    midi_files.extend(sorted(input_dir.rglob("*.midi")))
    if max_files is not None:
        midi_files = midi_files[:max_files]

    total = len(midi_files)
    print(f"Found {total} MIDI files in {input_dir}")

    manifest = []
    counts = {
        "total": total,
        "processed": 0,
        "saved": 0,
        "rejected": 0,
        "errors": 0,
    }

    # Process in parallel
    args_list = [(str(f),) for f in midi_files]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_file_worker, args): args[0]
                   for args in args_list}

        for i, future in enumerate(futures):
            filepath = futures[future]
            counts["processed"] += 1

            try:
                result = future.result(timeout=FILE_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.warning(f"Timeout processing {filepath}")
                counts["errors"] += 1
                continue
            except Exception as e:
                logger.warning(f"Error processing {filepath}: {e}")
                counts["errors"] += 1
                continue

            if result is None:
                counts["rejected"] += 1
                continue

            # Save as .npz
            song_id = Path(filepath).stem
            # Avoid collisions by including parent dir name
            parent_name = Path(filepath).parent.name
            npz_name = f"{parent_name}_{song_id}.npz"
            npz_path = output_dir / npz_name

            np.savez_compressed(
                npz_path,
                melody_chroma=result["melody_chroma"].astype(np.float32),
                chord_ids=result["chord_ids"].astype(np.int32),
            )

            manifest.append({
                "filename": npz_name,
                "source": result["filepath"],
                "n_beats": result["n_beats"],
                "tempo": round(result["tempo"], 1),
            })
            counts["saved"] += 1

            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1}/{total} "
                      f"(saved={counts['saved']}, rejected={counts['rejected']})")

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Saved {counts['saved']} files to {output_dir}")
    print(f"  Total: {counts['total']}")
    print(f"  Saved: {counts['saved']}")
    print(f"  Rejected: {counts['rejected']}")
    print(f"  Errors: {counts['errors']}")

    return counts
