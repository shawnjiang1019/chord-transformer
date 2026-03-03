# Chord Recommendation, Generation & Melody Harmonization

## Project Overview

Build a suite of tools powered by the Chordonomicon dataset that help musicians compose and explore chord progressions. The project has two interconnected systems:

1. **Chord Recommendation & Generation**: A standalone system that predicts, recommends, and generates chord progressions — conditioned on genre, song structure, and era. This includes top-K next chord suggestion, full progression generation, structure-guided composition, and an interactive songwriter's tool.

2. **Melody Harmonization**: A pipeline that takes a symbolic melody (MusicXML or MIDI) as input and generates chord progressions underneath it, combining bottom-up evidence (what notes are present in the melody) with top-down harmonic knowledge (what chord progressions actually occur in real music) learned from the Chordonomicon dataset.

The chord recommendation system is both a useful product on its own and the foundation that the melody harmonization system builds on top of.

## Core Dataset: Chordonomicon

- **Paper**: "CHORDONOMICON: A Dataset of 666,000 Songs and their Chord Progressions" (arXiv:2410.22046v3)
- **Source**: https://huggingface.co/datasets/ailsntua/Chordonomicon
- **GitHub**: https://github.com/spyroskantarelis/chordonomicon
- **Size**: 679,807 unique tracks, ~52 million total chords, 749 unique chord types
- **Metadata**: Genre (12 main categories), structural parts (Intro, Verse, Chorus, Bridge, etc.), release decade, Spotify IDs
- **Representations**: Harte syntax chord sequences, weighted directed graphs, binary 12-semitone vectors
- **Included scripts**: Chord transposition, chord-to-notes conversion, chord-to-binary-semitone conversion

## Melody Harmonization Pipeline

The following architecture describes the melody harmonization system, which uses the chord language model from the Recommendation & Generation system (above) as its core component.

### Pipeline Overview

```
Symbolic Melody (MusicXML/MIDI)
        │
        ▼
┌─────────────────────────┐
│ Stage 1: Parse & Segment│  ← Extract pitch classes per beat
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Stage 2: Candidate Chords│ ← Template match against 749 chord types
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Stage 3: Language Model  │ ← Trained on 666K Chordonomicon progressions
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Stage 4: Sequence Decode │ ← Viterbi/beam search combining stages 2+3
└──────────┬──────────────┘
           │
           ▼
    Chord Progression Output
```

### Stage 1: Melody Parsing & Segmentation

**Input**: Symbolic melody (MusicXML or MIDI)

**Process**:
- Parse into beat-level segments
- For each segment, collect all pitch classes present
- Weight notes by:
  - **Duration**: Longer notes are more likely chord tones
  - **Metrical position**: Notes on strong beats (1 and 3 in 4/4) are more harmonically significant
  - **Repetition**: Frequently occurring notes in a segment are more likely structural
- Output a 12-dimensional pitch-class profile (chroma vector) per segment

**Tools**: music21 (Python) for MusicXML parsing; mido or pretty_midi for MIDI

**Key considerations**:
- Handle tied notes that sustain across beats
- Ignore grace notes and ornaments
- Account for rests (empty segments)

### Stage 2: Candidate Chord Generation

**Process**:
- Use Chordonomicon's chord-to-semitone script to get binary 12-semitone templates for all 749 chord types
- Compare each segment's chroma vector against all templates using cosine similarity or weighted binary matching
- For each segment, retain top N candidates (e.g., top 5-10)
- Generate inversion candidates if the bass note doesn't match the chord root

**Scoring considerations**:
- Penalize missing chord tones more than extra notes (extra notes may be passing tones)
- Allow tolerance for incomplete voicings (real music often omits the fifth, etc.)
- Weight root and third more heavily than fifth in matching

### Stage 3: Contextual Disambiguation (Language Model)

**Training data**: All 666K+ chord progressions from Chordonomicon

**Model options**:
- GPT-2 style transformer (paper achieved 60% next-chord accuracy)
- LSTM (simpler, may be sufficient given the smaller vocabulary vs. natural language)
- Smaller transformer if latency matters for real-time use

**Conditioning** (optional but valuable):
- **Genre**: Jazz suggestions differ from pop suggestions for the same melody
- **Structural position**: Verse chords vs. chorus chords vs. bridge chords
- **Decade**: Reproduce harmonic style of specific eras

### Stage 4: Sequence Decoding

**Approach**: Viterbi or beam search that combines:
- **Local score**: Template similarity from Stage 2 (how well does this chord match the melody notes?)
- **Transition score**: Language model probability from Stage 3 (how likely is this chord given the preceding chords?)

**Key parameter**: Balance weight between melodic fit and harmonic coherence — this is the main hyperparameter to tune. In passages with clear note clusters, lean on template matching. In ambiguous passages, lean on the language model.

**Output**: Time-aligned sequence of chord labels in Harte syntax (convertible to standard chord symbols)

### Stage 5 (Optional): Structural Awareness

- If the input has section markers, condition the language model on section type
- Even without markers, use repetition structure to infer sections, then refine chord choices in a second pass
- Apply music-theoretic constraints: key consistency, functional harmony patterns (V→I resolutions, etc.)

## Chord Recommendation & Generation System

This is a standalone system that also serves as the foundation for the melody harmonization pipeline. It operates purely on chord progressions — no melody input required.

### Top-K Next Chord Recommendation

**Core idea**: Given a partial chord progression, predict the most likely next chords with probabilities.

**Input**: A sequence of chords, e.g., `G → Em → C`
**Output**: Ranked list of next chord candidates with probabilities, e.g., `D (34%), Am (18%), F (12%), G7 (9%), Bm (7%)`

**How it works**:
- Train a causal language model (GPT-2 style) on all 666K Chordonomicon progressions
- At inference time, feed the input sequence and sample from the output distribution
- Return the top K tokens (chords) by probability
- The paper already validated this approach: 60% next-chord accuracy, 75% note-level accuracy

**Conditioning options** (each adds a special token or embedding to the input):
- **Genre**: Prepend a genre token (e.g., `<jazz> Dm7 G7 ...`) so the model draws from genre-specific harmonic patterns. The dataset has 12 main genres and 179 rock sub-genres.
- **Structural position**: Prepend a section token (e.g., `<chorus> G Em C ...`). The dataset has 2.67 million structural labels across 397K tracks covering Intro, Verse, Chorus, Bridge, Interlude, Solo, Instrumental, and Outro.
- **Decade**: Prepend a decade token to reproduce era-specific harmonic style. The dataset covers 15 decades from 1890 to 2020s.
- **Combined**: These can be stacked — `<rock> <verse> <2000s> G Em C ...`

**Tokenization considerations**:
- Each unique chord is a token (749 base tokens + special tokens for genre/structure/decade)
- The paper notes that exploring good tokenization schemes is an open question — sub-chord tokenization (root + quality separately) could help with rare chord types
- Data augmentation via transposition: the dataset includes a script to transpose all progressions into all 12 keys, multiplying effective training data by 12

### Full Chord Progression Generation

**Core idea**: Generate complete chord progressions from scratch or from a prompt, following realistic song structures.

**Modes of generation**:

1. **Unconditioned**: Generate a progression from scratch — model picks a starting chord and continues
2. **Prompted**: User provides first few chords, model completes the rest
3. **Structure-guided**: User specifies a song structure (e.g., Intro → Verse → Chorus → Verse → Chorus → Bridge → Chorus → Outro), model generates appropriate chords for each section
4. **Style-conditioned**: User specifies genre, decade, or both, model generates in that style
5. **Infilling**: User provides chords at specific positions (e.g., "starts on Am, chorus starts on F, ends on C") and the model fills in the gaps

**Structure-guided generation in detail**:
- This is uniquely enabled by Chordonomicon's structural annotations, which no other large-scale dataset has
- Train the model on progressions with section tokens: `<intro> G ... <verse> Em C G D ... <chorus> C G Am F ...`
- At generation time, provide the section sequence as a template
- The model learns that intros tend to be short and atmospheric, verses are more repetitive, choruses are more resolved and tonic-heavy, bridges introduce harmonic surprise
- Could generate multiple candidates per section and let the user pick

**Sampling strategies**:
- **Temperature control**: Lower temperature for more conventional/predictable progressions, higher for more creative/unusual ones
- **Top-K sampling**: Restrict to top K most likely chords at each step
- **Top-P (nucleus) sampling**: Restrict to the smallest set of chords whose cumulative probability exceeds P
- **Repetition penalty**: Prevent the model from looping on the same 2-3 chord cycle (a known issue with music generation)
- **Key consistency**: Optionally constrain generated chords to stay within a key or set of related keys

**Evaluation**:
- **Perplexity**: Standard language model metric — lower is better
- **Note-level accuracy**: Convert predicted and actual chords to notes, measure overlap (paper reports 75.45%)
- **Musical plausibility**: Human evaluation — do generated progressions sound good? Do they follow genre conventions?
- **Diversity**: Are generated progressions varied or do they collapse to the same few patterns?
- **Structural coherence**: When generating with section tokens, do different sections actually sound different?

### Chord Progression Completion (Songwriter's Tool)

**Core idea**: An interactive tool where a songwriter builds a progression incrementally with AI assistance.

**Workflow**:
1. User enters a chord (e.g., Am)
2. System suggests top 5 next chords, ranked by probability
3. User picks one or enters their own choice
4. System updates suggestions based on the growing sequence
5. User can branch/undo at any point to explore alternatives

**Additional features**:
- **"Surprise me"**: Sample from the tail of the distribution to suggest unexpected but musically valid chords
- **Genre steering**: Change the genre mid-progression to shift the harmonic vocabulary
- **Section transitions**: User marks "now start the chorus" and the model adjusts its suggestions
- **Progression analysis**: Given a completed progression, identify similar songs in the dataset, detect the key, label functional harmony (I, IV, V, etc.)
- **Variation generation**: Given a verse progression, generate variations suitable for a second verse or chorus

### Graph-Based Chord Recommendation

**Core idea**: Use the graph representation of songs in Chordonomicon for recommendation instead of (or alongside) the sequence model.

**How it works**:
- Each song is a weighted directed graph where nodes are chords and edges are transitions with frequency weights
- Build a global transition graph aggregating all songs (or genre-specific subgraphs)
- For a given partial progression, traverse the graph to find the most weighted outgoing edges from the current chord
- This provides a fast, interpretable baseline that doesn't require neural network inference

**Advantages over the language model**:
- Extremely fast (graph lookup vs. model inference)
- Fully interpretable (you can see exactly why a chord was suggested)
- Can be filtered by genre trivially by using genre-specific subgraphs

**Limitations**:
- Only considers the current chord (or small window), not the full progression history
- Can't capture long-range dependencies the way a transformer can
- Less flexible for conditioned generation

**Hybrid approach**: Use graph-based recommendations for fast, real-time suggestions in an interactive tool, and fall back to the language model for more sophisticated tasks like structure-guided generation or long-sequence completion.

### Training Architecture Details

**Data preparation**:
1. Load Chordonomicon chord progressions in Harte syntax
2. Build vocabulary from all unique chords + special tokens (genre, structure, decade, BOS, EOS, PAD)
3. Optionally augment by transposing each progression into all 12 keys (using the provided transposition script)
4. Format training sequences with conditioning tokens prepended
5. Split: use the dataset's existing splits if available, otherwise 80/10/10 train/val/test

**Model architecture**:
- GPT-2 style decoder-only transformer (paper's approach)
- Suggested starting point: 4-6 layers, 4-8 attention heads, embedding dim 256-512
- This is much smaller than NLP GPT-2 because the vocabulary is tiny (749 chords vs. 50K+ words) and sequences are shorter (median 71 chords)
- Positional encoding: standard sinusoidal or learned

**Training details**:
- Causal language modeling objective (predict next chord)
- Cross-entropy loss
- Adam optimizer, learning rate ~1e-4 with warmup
- The paper pre-trained on unlabeled progressions then fine-tuned for classification — same pre-training step applies here
- Monitor validation perplexity and next-chord accuracy

**Inference optimizations**:
- KV-cache for autoregressive generation
- For the interactive songwriter tool, keep the model loaded and update incrementally as the user adds chords
- Batch multiple sampling strategies (different temperatures, genres) in parallel to offer diverse suggestions

## Supplementary Datasets

| Dataset | Size | What it provides | Link |
|---------|------|------------------|------|
| POP909 | 909 songs | Melody + chord + arrangement in MIDI | https://github.com/music-x-lab/POP909-Dataset |
| Lakh MIDI | ~176K files | Large-scale MIDI (messy, not consistently annotated) | https://colinraffel.com/projects/lmd/ |
| MusicNet | 330 recordings | Note-level annotations, classical music | https://zenodo.org/record/5120004 |
| MAESTRO | ~200 hrs | Piano performances in MIDI | https://magenta.tensorflow.org/datasets/maestro |

**Recommended strategy**: Pre-train chord language model on full Chordonomicon, then fine-tune on POP909 (or similar) for melody-chord pairing.

## Implementation Stack

- **music21**: MusicXML/MIDI parsing, pitch extraction
- **NumPy/SciPy**: Template matching, chroma vector computation
- **PyTorch or HuggingFace Transformers**: Language model training
- **Chordonomicon scripts**: Chord-to-note and chord-to-semitone conversions
- **Viterbi/beam search**: Custom implementation for sequence decoding

## Research Questions

### Chord Recommendation & Generation
1. **Tokenization**: Is it better to treat each chord as a single token, or decompose into root + quality (e.g., `C` + `maj7`)? Sub-chord tokenization could help generalize to rare chord types.
2. **Conditioning effectiveness**: How much does genre/structure/decade conditioning improve generation quality? Which conditioning signal matters most?
3. **Graph vs. transformer**: For next-chord prediction, how does the simple graph-lookup baseline compare to the transformer? At what sequence length does the transformer's advantage become clear?
4. **Transposition augmentation**: Does training on all 12 transpositions actually help, or does it just make the model key-agnostic at the cost of losing key-specific patterns?
5. **Structural coherence**: When generating with section tokens, do the generated sections actually exhibit different harmonic characteristics? Can listeners tell the difference?
6. **Vocabulary scope**: Should we use all 749 chord types or reduce to a smaller set for practical use? What's the accuracy/expressiveness tradeoff?
7. **Repetition vs. creativity**: How to balance generating progressions that are musically conventional (and thus useful) vs. novel and surprising?

### Melody Harmonization
8. **Weighting balance**: What's the optimal balance between melodic fit (template matching) and harmonic coherence (language model) in the decoding step?
9. **Transfer learning**: How well does pre-training on Chordonomicon transfer to melody-conditioned chord generation when fine-tuned on smaller paired datasets?
10. **Segment granularity**: Is beat-level segmentation optimal, or do measure-level or half-beat segments produce better harmonizations?
11. **Note weighting**: What's the best scheme for weighting melody notes (duration, metrical position, repetition) when generating chord candidates?

## Development Phases

### Phase 1: Data Exploration & Graph Baseline
- Download and explore the Chordonomicon dataset
- Build the global chord transition graph (aggregated across all songs)
- Build genre-specific subgraphs
- Implement basic graph-lookup chord recommendation as baseline
- Analyze chord distributions, common progressions, genre differences

### Phase 2: Language Model Training
- Build vocabulary and tokenizer (749 chords + special tokens)
- Implement data augmentation via transposition (12x multiplier)
- Train GPT-2 style transformer on all progressions
- Add conditioning tokens for genre, structure, and decade
- Evaluate: perplexity, next-chord accuracy, note-level accuracy
- Target: match or exceed paper's 60% chord / 75% note accuracy

### Phase 3: Chord Recommendation & Generation Tools
- Build top-K next chord recommendation with conditioning
- Implement full progression generation with multiple sampling strategies
- Implement structure-guided generation using section tokens
- Build interactive songwriter's tool (incremental composition with AI suggestions)
- Evaluate generation quality: diversity, musical plausibility, structural coherence

### Phase 4: Melody Parsing & Candidate Generation
- Implement MusicXML/MIDI parser with beat-level segmentation
- Build template matching against Chordonomicon chord vocabulary
- Test candidate generation quality on known melody-chord pairs (POP909)

### Phase 5: Full Harmonization Pipeline
- Integrate language model with candidate generation via Viterbi/beam search
- Tune the balance parameter between melodic fit and harmonic coherence
- Fine-tune on paired melody-chord data (POP909)
- Evaluate against human judgments of harmonization quality

### Phase 6: Refinements & Polish
- Add structural awareness and key detection
- Experiment with music-theoretic constraints via FHO/Chord Ontology
- Build user-facing interface for musicians
- Explore hybrid graph + transformer approaches
