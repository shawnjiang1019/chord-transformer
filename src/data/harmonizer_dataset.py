"""
Dataset for ChordHarmonizer training.

Produces infilling examples from paired (melody_chroma, chord_ids) .npz files.
Each window is randomly split into three sections:

    [  past chords  |  melody gap  |  future chords  ]

The model must predict target_chord_ids (the gap's chords) given:
    - past_chord_ids   : chord context before the gap
    - future_chord_ids : chord context after the gap
    - melody_chroma    : beat-level melody in the gap

Because lengths vary per sample, a custom collate_fn pads each field to the
batch maximum and returns length tensors so the model ignores padding.
"""

import json
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from src.data.tokenizer import ChordTokenizer


class HarmonizerDataset(Dataset):
    """
    Each __getitem__ returns a dict with:
        past_chord_ids   : (past_len, 3)   long
        future_chord_ids : (future_len, 3) long
        melody_chroma    : (gap_len, 12)   float
        target_chord_ids : (gap_len, 3)    long
        past_len         : int
        future_len       : int
        melody_len       : int
    """

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: ChordTokenizer,
        window_size: int = 32,
        stride: int = 16,
        min_section: int = 4,
        augment: bool = True,
        n_splits: int = 3,
    ):
        self.window_size = window_size
        self.min_section = min_section
        self.pad_id = tokenizer.token2id["[PAD]"]
        self.root_offset = tokenizer.token2id["C"]
        self.augment = augment
        self.n_splits = n_splits

        self.windows: list[tuple[Path, int, int]] = []
        self._build_index(Path(data_dir), stride)

    def _build_index(self, data_dir: Path, stride: int):
        """Build sliding window index from manifest.json."""
        manifest_path = data_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        transpositions = range(12) if self.augment else range(1)

        for entry in manifest:
            npz_path = data_dir / entry["filename"]
            n_beats = entry["n_beats"]
            if n_beats < 3 * self.min_section:
                continue
            for semitones in transpositions:
                max_start = max(1, n_beats - self.window_size + 1)
                for start in range(0, max_start, stride):
                    self.windows.append((npz_path, start, semitones))

    def __len__(self) -> int:
        return len(self.windows) * self.n_splits

    def __getitem__(self, idx) -> dict:
        window_idx = idx // self.n_splits
        npz_path, start, semitones = self.windows[window_idx]

        data = np.load(npz_path)
        melody = data["melody_chroma"]   # (n_beats, 12)
        chords = data["chord_ids"]       # (n_beats, 3)

        end = min(start + self.window_size, len(melody))
        melody_win = melody[start:end].copy()
        chord_win = chords[start:end].copy()
        n = len(melody_win)

        if semitones > 0:
            melody_win = np.roll(melody_win, semitones, axis=1)
            chord_win = self._transpose_chords(chord_win, semitones)

        past_end, gap_end = self._random_split(n)

        past_chord_ids   = chord_win[:past_end]           # (past_len, 3)
        melody_chroma    = melody_win[past_end:gap_end]   # (gap_len, 12)
        target_chord_ids = chord_win[past_end:gap_end]    # (gap_len, 3)
        future_chord_ids = chord_win[gap_end:]            # (future_len, 3)

        return {
            "past_chord_ids":   torch.tensor(past_chord_ids,   dtype=torch.long),
            "future_chord_ids": torch.tensor(future_chord_ids, dtype=torch.long),
            "melody_chroma":    torch.tensor(melody_chroma,    dtype=torch.float32),
            "target_chord_ids": torch.tensor(target_chord_ids, dtype=torch.long),
            "past_len":         torch.tensor(len(past_chord_ids),   dtype=torch.long),
            "future_len":       torch.tensor(len(future_chord_ids), dtype=torch.long),
            "melody_len":       torch.tensor(len(melody_chroma),    dtype=torch.long),
        }

    def _random_split(self, n: int) -> tuple[int, int]:
        """
        Return (past_end, gap_end) such that each section has at least min_section beats.
        """
        m = self.min_section
        past_end = random.randint(m, n - 2 * m)
        gap_end  = random.randint(past_end + m, n - m)
        return past_end, gap_end

    def _transpose_chords(self, chord_ids: np.ndarray, semitones: int) -> np.ndarray:
        result = chord_ids.copy()
        roots = result[:, 0]
        mask = roots != self.pad_id
        if mask.any():
            roots[mask] = self.root_offset + (roots[mask] - self.root_offset + semitones) % 12
            result[:, 0] = roots
        return result


def harmonizer_collate_fn(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """
    Pads variable-length fields to the maximum length in the batch.

    past_chord_ids and future_chord_ids are padded along the beat dimension.
    melody_chroma and target_chord_ids are padded to the max gap length.
    """
    def pad_2d(tensors, pad_val, dtype):
        max_len = max(t.size(0) for t in tensors)
        D = tensors[0].size(1)
        out = torch.full((len(tensors), max_len, D), pad_val, dtype=dtype)
        for i, t in enumerate(tensors):
            out[i, :t.size(0)] = t
        return out

    return {
        "past_chord_ids":   pad_2d([b["past_chord_ids"]   for b in batch], pad_id, torch.long),
        "future_chord_ids": pad_2d([b["future_chord_ids"] for b in batch], pad_id, torch.long),
        "melody_chroma":    pad_2d([b["melody_chroma"]    for b in batch], 0.0,    torch.float32),
        "target_chord_ids": pad_2d([b["target_chord_ids"] for b in batch], pad_id, torch.long),
        "past_len":   torch.stack([b["past_len"]   for b in batch]),
        "future_len": torch.stack([b["future_len"] for b in batch]),
        "melody_len": torch.stack([b["melody_len"] for b in batch]),
    }
