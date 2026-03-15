"""
Chordonomicon dataset — PyTorch Dataset that streams from HuggingFace,
encodes each song with ChordTokenizer, and pads/truncates to max_seq_len.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from src.data.tokenizer import ChordTokenizer


class ChordDataset(Dataset):
    """Loads Chordonomicon songs, encodes them, and serves padded tensors."""

    def __init__(self, tokenizer: ChordTokenizer, max_seq_len: int = 512,
                 split: str = "train", max_songs: int | None = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.token2id["[PAD]"]
        self.sequences: list[list[int]] = []

        self._load(split, max_songs)

    def _load(self, split: str, max_songs: int | None):
        """Stream from HuggingFace and encode each song."""
        ds = load_dataset("ailsntua/Chordonomicon", split="train", streaming=True)

        for i, example in enumerate(ds):
            if max_songs is not None and i >= max_songs:
                break
            ids = self.tokenizer.encode(example["chords"])
            # Truncate if too long
            if len(ids) > self.max_seq_len:
                ids = ids[:self.max_seq_len - 1] + [self.tokenizer.token2id["[EOS]"]]
            self.sequences.append(ids)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        ids = self.sequences[idx]

        # Pad to max_seq_len
        padded = ids + [self.pad_id] * (self.max_seq_len - len(ids))

        # Input: all tokens except last, Target: all tokens except first
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)

        return {"input_ids": x, "labels": y}
