"""
Phase 3 — Top-K next chord recommendation.

Given a partial progression (optionally conditioned on genre/structure/decade),
returns ranked next chord candidates with probabilities.

Example:
    recommender = ChordRecommender(model, tokenizer)
    results = recommender.recommend(["G", "Em", "C"], genre="rock", top_k=5)
    # -> [("D", 0.34), ("Am", 0.18), ("F", 0.12), ...]
"""

from src.model.transformer import ChordTransformer
from src.data.tokenizer import ChordTokenizer
from src.graph.chord_graph import ChordGraph


class ChordRecommender:
    """
    Wraps both the language model and graph baseline for next-chord recommendation.
    Switch `use_graph=True` for fast/interpretable results, False for the full LM.
    """

    def __init__(
        self,
        model: ChordTransformer,
        tokenizer: ChordTokenizer,
        graph: ChordGraph | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.graph = graph

    def recommend(
        self,
        progression: list[str],
        genre: str | None = None,
        structure: str | None = None,
        decade: str | None = None,
        top_k: int = 5,
        use_graph: bool = False,
    ) -> list[tuple[str, float]]:
        """
        Return top-K (chord, probability) pairs for the next chord.

        Conditioning tokens are prepended in order: genre, structure, decade.
        If use_graph=True and a graph is loaded, uses graph lookup instead of the LM.
        """
        if use_graph and self.graph:
            return self.graph.recommend(progression[-1], top_k=top_k)
        raise NotImplementedError
