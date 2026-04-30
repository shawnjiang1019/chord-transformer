"""
Melody-Chord paired dataset for VAE training.

Loads preprocessed .npz files from the Lakh pipeline output.
Each sample is a fixed-length window of (melody_chroma, chord_ids) pairs.
"""

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from src.data.tokenizer import ChordTokenizer


class MelodyChordDataset(Dataset):
    """
    Dataset of paired (melody_chroma, chord_token_ids) sequences
    for training a melody-conditioned chord VAE.

    Each __getitem__ returns:
        melody_chroma: (window_size, 12) float tensor
        chord_ids:     (window_size, 3) long tensor  [root, quality, voicing]
        length:        int, actual length before padding
    """

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: ChordTokenizer,
        window_size: int = 32,
        stride: int = 16,
        augment: bool = True,
    ):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.pad_id = tokenizer.token2id["[PAD]"]

        # Root token offset: the token ID of "C" (first root token)
        self.root_offset = tokenizer.token2id["C"]

        self.windows: list[tuple[Path, int, int]] = []
        self._build_index(Path(data_dir), stride, augment)

    def _build_index(self, data_dir: Path, stride: int, augment: bool):
        """Build sliding windows over each song in the manifest."""
        manifest_path = data_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        transpositions = range(12) if augment else range(1)

        for entry in manifest:
            npz_path = data_dir / entry["filename"]
            n_beats = entry["n_beats"]

            for semitones in transpositions:
                # Sliding windows with given stride
                max_start = max(1, n_beats - self.window_size + 1)
                for start in range(0, max_start, stride):
                    self.windows.append((npz_path, start, semitones))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        npz_path, start, semitones = self.windows[idx]
        data = np.load(npz_path)

        melody = data["melody_chroma"]  # (n_beats, 12)
        chords = data["chord_ids"]      # (n_beats, 3)

        # Extract window
        end = min(start + self.window_size, len(melody))
        melody_window = melody[start:end].copy()
        chord_window = chords[start:end].copy()
        actual_len = len(melody_window)

        # Apply transposition
        if semitones > 0:
            melody_window = self._transpose_chroma(melody_window, semitones)
            chord_window = self._transpose_chords(chord_window, semitones)

        # Pad to window_size
        if actual_len < self.window_size:
            pad_len = self.window_size - actual_len
            melody_window = np.pad(
                melody_window, ((0, pad_len), (0, 0)), mode='constant'
            )
            chord_pad = np.full((pad_len, 3), self.pad_id, dtype=np.int32)
            chord_window = np.concatenate([chord_window, chord_pad], axis=0)

        return {
            "melody_chroma": torch.tensor(melody_window, dtype=torch.float32),
            "chord_ids": torch.tensor(chord_window, dtype=torch.long),
            "length": torch.tensor(actual_len, dtype=torch.long),
        }

    @staticmethod
    def _transpose_chroma(chroma: np.ndarray, semitones: int) -> np.ndarray:
        """Roll chroma vectors by semitones (circular shift along pitch axis)."""
        return np.roll(chroma, semitones, axis=1)

    def _transpose_chords(self, chord_ids: np.ndarray, semitones: int) -> np.ndarray:
        """
        Transpose chord token IDs by shifting the root index.
        Root tokens are contiguous in the vocab: C=offset, Cs=offset+1, ..., B=offset+11.
        Quality and voicing tokens are invariant under transposition.
        """
        result = chord_ids.copy()
        root_ids = result[:, 0]

        # Only transpose non-pad entries
        mask = root_ids != self.pad_id
        if mask.any():
            root_ids[mask] = (
                self.root_offset
                + (root_ids[mask] - self.root_offset + semitones) % 12
            )
            result[:, 0] = root_ids

        return result
