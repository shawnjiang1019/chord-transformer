# Graph-Based Chord Recommendation

## What it does

Builds a weighted directed graph from all 666K+ Chordonomicon progressions.
Each node is a unique chord string, each edge is a transition between two chords,
and each edge weight is a count of how many times that transition appeared in the dataset.

```
G ──12,400──▶ Em
G ──9,800───▶ D
G ──8,700───▶ C
G ──3,100───▶ Am
...
```

When given a chord, the recommender looks up its outgoing edges, normalizes the counts
to probabilities, and returns the top-K most likely next chords.

## Data structure

```python
transitions: {
    "G":   { "Em": 12400, "D": 9800, "C": 8700, "Am": 3100 },
    "Em":  { "C": 9200,  "Am": 6700, "G": 4400  },
    "Dm7": { "G7": 8800, "Cmaj7": 5100           },
    ...
}
```

## Files

### `chord_graph.py`

| Name | Type | Description |
|------|------|-------------|
| `ChordGraph` | class | The graph container |
| `ChordGraph.add_sequence(seq)` | method | Ingest one chord progression, incrementing transition counts |
| `ChordGraph.recommend(chord, top_k)` | method | Return top-K `(chord, probability)` tuples for the next chord |
| `ChordGraph.save(path)` | method | Persist graph to JSON |
| `ChordGraph.load(path)` | method | Load graph from JSON |
| `build_graph(sequences)` | function | Build a `ChordGraph` from a list of chord sequences |

## Usage

```python
from src.graph.chord_graph import ChordGraph, build_graph

# Build from sequences
graph = build_graph(all_sequences)
graph.save("data/processed/graph_global.json")

# Load and query
graph = ChordGraph()
graph.load("data/processed/graph_global.json")

graph.recommend("G", top_k=5)
# -> [("Em", 0.29), ("D", 0.23), ("C", 0.21), ("Am", 0.07), ("Bm", 0.06)]
```

## Advantages over the language model

| Property | Graph | Transformer |
|----------|-------|-------------|
| Speed | Instant (dict lookup) | Slower (neural inference) |
| Interpretability | Full — you can inspect exact counts | Black box |
| Training required | No | Yes |
| Memory context | Current chord only | Full progression history |
| Conditioning | Separate subgraph per genre | Single model, conditioned via tokens |

## Limitations

- Only considers the **current chord**, not the full progression history
- Cannot capture long-range harmonic patterns (e.g. "a ii–V–I that started 4 bars ago")
- Genre filtering requires building and storing separate subgraphs

## Where it fits in the project

- **Phase 1** — built as the first working baseline; used to analyze the dataset
- **Phase 3+** — used as the fast real-time fallback in the interactive songwriter tool
- **Hybrid** — the recommended long-term approach is graph for instant suggestions,
  transformer for structure-guided generation and long-sequence completion
