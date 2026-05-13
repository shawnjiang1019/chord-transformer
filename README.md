# Chord Transformer

A melody-conditioned chord harmonizer built on a GPT-2 style decoder-encoder transformer. Given a melody and surrounding chord context, the model generates chord progressions that fit the melody and bridge naturally between the known chords.

## What it does

- **Chord language model** — trained on 679K songs from the Chordonomicon dataset, learns chord progressions and harmonic grammar
- **Chord harmonizer** — given past chords, a melody, and future chords, fills in the missing chord progression for the melody section

### Harmonizer task

```
[C  F  G  C]  [melody beats...]  [F  C  G  C]
  past chords    generate here     future chords
```

The model uses the surrounding chord context and beat-level melody chroma vectors to generate chords that are musically coherent with both sides.

## Architecture

```
Past + Future chords → Encoder (bidirectional transformer) → context vectors
                                                                    │
Melody chroma ──────────────────────────────────────────► Decoder (cross-attention)
                                                                    │
                                                             chord logits
```

- **Encoder** — bidirectional transformer over `[past chords | SEP | future chords]`. Each chord beat is embedded as `embed(root) + embed(quality) + embed(voicing)`.
- **Decoder** — ChordTransformer (causal) extended with cross-attention to the encoder and melody chroma injection at each token position.
- **Chords** — decomposed into 3 tokens each: `[root, quality, voicing]`

## Project structure

```
src/
    model/
        blocks/
            CausalSelfAttention.py
            CrossAttention.py
            TransformerBlock.py
        transformer.py      # ChordTransformer (decoder-only chord LM)
        harmonizer.py       # ChordHarmonizer (encoder + decoder)
        train.py            # training utilities for chord LM
        vae.py              # VAE skeleton (future)
    data/
        tokenizer.py        # chord tokenizer (3-token decomposition)
        melody_chord_dataset.py
        harmonizer_dataset.py
        wikifonia/
            processing.py   # MusicXML → (melody chroma, chord) pairs
        lakh/
            processing.py   # MIDI processing pipeline
    harmonization/
        candidates.py
        parser.py
    graph/
        chord_graph.py
app/                        # inference application (future)
scripts/
    download_data.py        # download Chordonomicon from HuggingFace
    train_model.py          # train the chord language model
    train_harmonizer.py     # train the chord harmonizer
    process_wikifonia.py    # process Wikifonia dataset
    test_generate.py        # test chord generation
    build_graph.py          # build chord transition graph
data/
    processed/
        wikifonia/          # processed .npz files + manifest.json
checkpoints/                # saved model weights
```

## Getting started

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourname/chord-transformer
cd chord-transformer
pip install -r requirements.txt
```

### 2. Download the Chordonomicon dataset

```bash
PYTHONPATH=. python scripts/download_data.py
```

Downloads ~679K chord progressions from HuggingFace into `data/`.

### 3. Train the chord language model

```bash
PYTHONPATH=. python scripts/train_model.py
```

Trains a GPT-2 style chord language model. Checkpoints saved to `checkpoints/`.

### 4. Process the Wikifonia dataset (for harmonizer training)

Download the Wikifonia dataset (`.mxl` files) and place them in `data/Wikifonia/`. Then:

```bash
PYTHONPATH=. python scripts/process_wikifonia.py \
    --input_dir data/Wikifonia \
    --output_dir data/processed/wikifonia
```

Processes ~6K lead sheets into beat-level `(melody chroma, chord)` pairs. Produces `.npz` files and a `manifest.json`.

### 5. Train the harmonizer

```bash
PYTHONPATH=. python scripts/train_harmonizer.py \
    --data_dir data/processed/wikifonia \
    --output_dir checkpoints/harmonizer \
    --pretrained_decoder checkpoints/best_model.pt
```

`--pretrained_decoder` initializes the decoder with chord LM weights, transferring learned harmonic knowledge into the harmonizer.

### 6. Generate chords

```bash
PYTHONPATH=. python scripts/test_generate.py
```

## Inference

```python
import torch
from src.model.harmonizer import HarmonizerConfig, ChordHarmonizer
from src.data.tokenizer import ChordTokenizer

tokenizer = ChordTokenizer()
config    = HarmonizerConfig(vocab_size=tokenizer.vocab_size)
model     = ChordHarmonizer(config)

ckpt = torch.load("checkpoints/harmonizer/best_model.pt", weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# past_chord_ids:   (1, P, 3) long tensor
# future_chord_ids: (1, F, 3) long tensor
# melody_chroma:    (1, W, 12) float tensor — beat-level pitch class weights

with torch.no_grad():
    chord_ids = model.generate(
        past_chord_ids=past,
        future_chord_ids=future,
        melody_chroma=chroma,
        temperature=1.0,
        top_k=50,
    )
# chord_ids: (1, W, 3) — one [root, quality, voicing] triplet per beat
```

## Data format

Processed Wikifonia files are `.npz` archives with two arrays:

| Field | Shape | Description |
|---|---|---|
| `melody_chroma` | `(n_beats, 12)` | Beat-level pitch class weights, L2-normalized |
| `chord_ids` | `(n_beats, 3)` | `[root_id, quality_id, voicing_id]` per beat |

## Requirements

- Python 3.10+
- PyTorch 2.2+
- music21 (MusicXML parsing)
- numpy, scipy
- HuggingFace datasets (Chordonomicon download)
