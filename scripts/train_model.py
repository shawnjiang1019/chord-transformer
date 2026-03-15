"""
Phase 2 — Training entry point.

Usage:
    python scripts/train_model.py --config configs/model_config.yaml
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataset import ChordDataset
from src.data.tokenizer import ChordTokenizer
from src.model.transformer import ChordTransformer
from src.model.train import train


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Tokenizer
    tokenizer = ChordTokenizer()

    # Load full dataset
    max_seq_len = cfg["model"]["max_seq_len"]
    print("Loading dataset from HuggingFace (streaming)...")
    full_ds = ChordDataset(tokenizer, max_seq_len=max_seq_len)
    print(f"Loaded {len(full_ds)} songs")

    # Split into train/val
    train_ratio = cfg["data"]["train_split"]
    val_ratio = cfg["data"]["val_split"]
    n_total = len(full_ds)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, _ = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")

    # DataLoaders
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = ChordTransformer(
        vocab_size=tokenizer.vocab_size,
        **cfg["model"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        warmup_steps=cfg["training"]["warmup_steps"],
        max_grad_norm=cfg["training"]["grad_clip"],
        device=cfg["training"]["device"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args.config)
