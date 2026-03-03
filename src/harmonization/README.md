# Melody Harmonization Pipeline

Given a symbolic melody (MusicXML or MIDI), this pipeline generates a chord progression
to go underneath it. It combines **bottom-up evidence** (what notes are in the melody) with
**top-down harmonic knowledge** (what progressions actually appear in real music) learned
from the Chordonomicon dataset.

## Pipeline overview

```
Symbolic Melody (MusicXML / MIDI)
        │
        ▼
┌─────────────────────────────┐
│ Stage 1: parser.py          │  ← Extract weighted chroma vectors per beat
└──────────┬──────────────────┘
           │  list[MelodySegment]  (beat_index, chroma[12])
           ▼
┌─────────────────────────────┐
│ Stage 2: candidates.py      │  ← Match each chroma against 749 chord templates
└──────────┬──────────────────┘
           │  list[list[(ChordTemplate, score)]]  (top-N per beat)
           ▼
┌─────────────────────────────┐
│ Stage 3: decoder.py         │  ← Viterbi / beam search combining stages 1 + 2
└──────────┬──────────────────┘
           │  list[str]
           ▼
    Chord Progression Output
```

---

## Stage 1 — `parser.py`

**What it does:** Reads a melody file and splits it into beat-level segments. Each segment
is converted into a 12-dimensional **chroma vector** — a pitch-class fingerprint that
describes which notes are present and how important they are.

**Note weighting scheme:**

| Factor | Why it matters |
|--------|----------------|
| Duration | A half note is more likely a chord tone than a grace note |
| Metrical position | Beat 1 and beat 3 in 4/4 are harmonically stronger |
| Repetition | A note that appears in every bar is probably structural |

**API:**

| Name | Status | Description |
|------|--------|-------------|
| `MelodySegment` | Implemented | Holds `beat` index and `chroma` array (shape `(12,)`) |
| `parse_musicxml(filepath)` | Stub | MusicXML → `list[MelodySegment]` via `music21` |
| `parse_midi(filepath)` | Stub | MIDI → `list[MelodySegment]` via `mido` / `pretty_midi` |
| `compute_chroma(notes)` | Partial | Note list → weighted, L2-normalized chroma vector |

---

## Stage 2 — `candidates.py`

**What it does:** For each beat's chroma vector, scores all 749 Chordonomicon chord types
by cosine similarity and returns the top-N candidates. This is the "bottom-up" evidence —
it answers *"given the notes in this beat, which chords are most compatible?"*

**Scoring rules (from the project spec):**
- Penalize **missing chord tones** more than extra notes (extra notes may be passing tones)
- Weight **root and third** more heavily than the fifth
- Tolerate incomplete voicings (real music often omits the fifth)

**API:**

| Name | Status | Description |
|------|--------|-------------|
| `ChordTemplate` | Implemented | Stores `name` (Harte syntax) and `semitones` array (shape `(12,)`) |
| `load_templates(output)` | Stub | Load all 749 templates from Chordonomicon chord-to-semitone script |
| `score_chord(chroma, template)` | Implemented | Cosine similarity between a chroma vector and a template |
| `get_candidates(chroma, templates, top_n)` | Implemented | Return top-N `(ChordTemplate, score)` pairs |

**Example output for one beat:**
```
Beat 4 candidates:
  Gmaj  0.94
  G7    0.88
  Gmaj7 0.85
  Em    0.71
  Bm    0.68
  ...
```

---

## Stage 3 — `decoder.py`

**What it does:** Takes the per-beat candidate lists from Stage 2 and finds the globally
best chord sequence across all beats using Viterbi or beam search.

At each timestep, it combines two signals:

```
final_score = alpha * local_score + (1 - alpha) * lm_score
```

| Signal | Source | Meaning |
|--------|--------|---------|
| `local_score` | Stage 2 template similarity | How well this chord fits the melody notes |
| `lm_score` | Transformer language model | How likely this chord follows the preceding chords |

**The `alpha` parameter** is the main hyperparameter to tune:

```
alpha = 1.0  →  pure template matching — ignores harmonic flow entirely
alpha = 0.5  →  balanced (default, start here)
alpha = 0.0  →  pure language model — ignores the melody entirely
```

In practice: use higher alpha in beats with clear, strong melody notes; lower alpha in
ambiguous or rest-heavy passages where the language model's harmonic intuition is more reliable.

**API:**

| Name | Status | Description |
|------|--------|-------------|
| `DecoderConfig` | Implemented | Dataclass: `alpha`, `beam_width`, `top_n_candidates` |
| `viterbi_decode(candidates, lm_fn, config)` | Stub | Returns single best sequence as `list[str]` |
| `beam_search_decode(candidates, lm_fn, config)` | Stub | Returns `beam_width` sequences, best-first |

---

## Full data flow (with types)

```python
# Stage 1
segments: list[MelodySegment] = parse_musicxml("song.xml")

# Stage 2
templates: list[ChordTemplate] = load_templates(...)
candidates: list[list[tuple[ChordTemplate, float]]] = [
    get_candidates(seg.chroma, templates, top_n=8)
    for seg in segments
]

# Stage 3
chord_sequence: list[str] = viterbi_decode(
    candidate_scores=[(t.name, s) for t, s in beat] for beat in candidates],
    lm_log_probs_fn=model.log_probs,
    config=DecoderConfig(alpha=0.5),
)
# -> ["Am", "Am", "F", "F", "C", "C", "G", "G"]
```

---

## Implementation status

| File | Implemented | Stubs remaining |
|------|-------------|-----------------|
| `parser.py` | `MelodySegment` | `parse_musicxml`, `parse_midi`, `compute_chroma` body |
| `candidates.py` | `score_chord`, `get_candidates` | `load_templates` |
| `decoder.py` | `DecoderConfig` | `viterbi_decode`, `beam_search_decode` |

Implement in order: `parser.py` → `candidates.py` (`load_templates`) → `decoder.py`.
Both decode functions depend on a trained language model from `src/model/`.
