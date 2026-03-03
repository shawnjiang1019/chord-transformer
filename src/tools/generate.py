"""
Phase 3 — Full chord progression generation.

Supports five modes:
  1. unconditioned   : generate from scratch
  2. prompted        : user provides first N chords, model completes
  3. structure_guided: user provides section sequence, model fills each section
  4. style_conditioned: genre + decade conditioning
  5. infilling       : anchors at specific positions, model fills gaps
"""

from src.model.transformer import ChordTransformer
from src.data.tokenizer import ChordTokenizer


class ProgressionGenerator:

    def __init__(self, model: ChordTransformer, tokenizer: ChordTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: list[str] | None = None,
        genre: str | None = None,
        structure: list[str] | None = None,  # e.g. ["intro","verse","chorus"]
        decade: str | None = None,
        max_length: int = 32,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_sequences: int = 1,
    ) -> list[list[str]]:
        """
        Generate `num_sequences` chord progressions.

        Returns a list of chord-string lists.
        """
        raise NotImplementedError
