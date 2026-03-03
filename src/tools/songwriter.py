"""
Phase 3 — Interactive songwriter's tool.

Stateful session where a user builds a progression incrementally.
At each step the model surfaces top-K suggestions; the user picks one
or enters their own, and the session grows.

Features:
  - undo / branch to explore alternative paths
  - "surprise me" mode (sample from the tail of the distribution)
  - genre steering mid-progression
  - section transition prompting
"""

from src.tools.recommend import ChordRecommender


class SongwriterSession:

    def __init__(self, recommender: ChordRecommender):
        self.recommender = recommender
        self.history: list[str] = []
        self._snapshots: list[list[str]] = []  # for undo/branch

    def add(self, chord: str):
        """Accept a chord (user-chosen or suggested) and append to history."""
        self._snapshots.append(list(self.history))
        self.history.append(chord)

    def suggest(self, top_k: int = 5, surprise: bool = False, **kwargs) -> list[tuple[str, float]]:
        """
        Return next-chord suggestions based on current history.
        If surprise=True, sample from the lower-probability tail.
        """
        results = self.recommender.recommend(self.history, top_k=top_k if not surprise else 20, **kwargs)
        if surprise:
            # Return the bottom half of candidates to encourage unexpected choices
            return results[len(results) // 2:]
        return results

    def undo(self):
        """Revert to the previous state."""
        if self._snapshots:
            self.history = self._snapshots.pop()

    def reset(self):
        self.history = []
        self._snapshots = []
