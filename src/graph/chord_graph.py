"""
Phase 1 — Graph-based chord recommendation baseline.

Builds a weighted directed transition graph from Chordonomicon progressions.
Fast, interpretable alternative to the language model; also useful as a
hybrid fallback for real-time suggestions.
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional
import json


class ChordGraph:
    """
    Weighted directed graph of chord transitions.

    nodes  : unique chord strings
    edges  : (src, dst) -> count
    """

    def __init__(self):
        # { chord: { next_chord: count } }
        self.transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add_sequence(self, sequence: list[str]):
        """Ingest one chord progression into the graph."""
        for src, dst in zip(sequence, sequence[1:]):
            self.transitions[src][dst] += 1

    def recommend(self, current_chord: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Return top-K next chords by transition frequency.
        Returns list of (chord, probability) tuples.
        """
        counts = self.transitions.get(current_chord, {})
        if not counts:
            return []
        total = sum(counts.values())
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [(chord, count / total) for chord, count in ranked[:top_k]]

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump({k: dict(v) for k, v in self.transitions.items()}, f)

    def load(self, path: str | Path):
        with open(path) as f:
            data = json.load(f)
        self.transitions = defaultdict(lambda: defaultdict(int))
        for src, dsts in data.items():
            for dst, count in dsts.items():
                self.transitions[src][dst] = count


def build_graph(sequences: list[list[str]], genre_filter: Optional[str] = None) -> ChordGraph:
    """Build a ChordGraph from a list of chord sequences."""
    graph = ChordGraph()
    for seq in sequences:
        graph.add_sequence(seq)
    return graph
