"""
Phase 1 — Build and persist chord transition graphs.

Usage:
    python scripts/build_graph.py --data_dir data/raw --output_dir data/processed
"""

import argparse
from pathlib import Path

from src.graph.chord_graph import build_graph


def main(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: load sequences from data_dir
    sequences: list[list[str]] = []

    global_graph = build_graph(sequences)
    global_graph.save(output_dir / "graph_global.json")
    print("Saved global graph.")

    # Build per-genre subgraphs
    # TODO: group sequences by genre metadata, then call build_graph per genre


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.output_dir))
