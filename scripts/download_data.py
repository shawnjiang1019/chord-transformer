"""
Download the Chordonomicon dataset from HuggingFace.

Usage:
    python scripts/download_data.py --output_dir data/raw
"""

import argparse
from pathlib import Path


def download(output_dir: Path):
    """Pull Chordonomicon via the HuggingFace datasets library."""
    from datasets import load_dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("ailsntua/Chordonomicon")
    ds.save_to_disk(str(output_dir / "chordonomicon"))
    print(f"Saved to {output_dir / 'chordonomicon'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/raw")
    args = parser.parse_args()
    download(Path(args.output_dir))
