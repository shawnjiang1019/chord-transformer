"""
Phase 2 — Training entry point.

Usage:
    python scripts/train_model.py --config configs/model_config.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.data.dataset import ChordDataset
from src.data.tokenizer import ChordTokenizer
from src.model.transformer import ChordTransformer
from src.model.train import train


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tokenizer = ChordTokenizer()
    tokenizer.load("data/processed/tokenizer.json")

    model = ChordTransformer(
        vocab_size=tokenizer.vocab_size,
        **cfg["model"],
    )

    train_ds = ChordDataset(cfg["data"]["data_dir"], split="train")
    val_ds = ChordDataset(cfg["data"]["data_dir"], split="val")
    train_ds.load()
    val_ds.load()

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        warmup_steps=cfg["training"]["warmup_steps"],
        device=cfg["training"]["device"],
    )

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/chord_transformer.pt")
    print("Model saved to checkpoints/chord_transformer.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args.config)
