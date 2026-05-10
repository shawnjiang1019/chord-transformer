"""
Process Wikifonia dataset into paired melody-chord .npz files.

Usage:
    python scripts/process_wikifonia.py
    python scripts/process_wikifonia.py --input_dir data/Wikifonia --output_dir data/processed/wikifonia
"""

import argparse
from pathlib import Path

from src.data.tokenizer import ChordTokenizer
from src.data.wikifonia.processing import process_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Wikifonia MXL files")
    parser.add_argument("--input_dir", default="data/Wikifonia")
    parser.add_argument("--output_dir", default="data/processed/wikifonia")
    args = parser.parse_args()

    tokenizer = ChordTokenizer()
    counts = process_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        tokenizer=tokenizer,
    )

    print("\nSummary:")
    for key, value in counts.items():
        print(f"  {key}: {value}")
