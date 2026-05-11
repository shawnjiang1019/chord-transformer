"""
Training script for ChordHarmonizer.

Usage:
    python scripts/train_harmonizer.py \
        --data_dir data/processed/wikifonia \
        --output_dir checkpoints/harmonizer \
        [--pretrained_decoder checkpoints/chord_transformer/best_model.pt]
"""

import argparse
import math
import os
from functools import partial

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split

from src.data.tokenizer import ChordTokenizer
from src.data.harmonizer_dataset import HarmonizerDataset, harmonizer_collate_fn
from src.model.harmonizer import HarmonizerConfig, ChordHarmonizer


def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup from 0 to base_lr over warmup_steps."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def compute_loss(
    logits: torch.Tensor,           # (B, W*3, vocab_size)
    target_chord_ids: torch.Tensor, # (B, W, 3)
    melody_len: torch.Tensor,       # (B,) actual gap lengths
    pad_id: int,
    vocab_size: int,
) -> torch.Tensor:
    """
    Cross-entropy over token positions, ignoring positions beyond melody_len (padding).
    Target is flattened to (B, W*3) before computing loss.
    """
    B, W, _ = target_chord_ids.shape
    target_flat = target_chord_ids.reshape(B, W * 3)  # (B, W*3)

    # Mask out padded beats: True = real, False = padding
    beat_mask = torch.arange(W, device=melody_len.device).unsqueeze(0) < melody_len.unsqueeze(1)
    token_mask = beat_mask.repeat_interleave(3, dim=1)  # (B, W*3)
    target_flat = target_flat.masked_fill(~token_mask, pad_id)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    return criterion(logits.reshape(-1, vocab_size), target_flat.reshape(-1))


def forward_batch(model: ChordHarmonizer, batch: dict, device: str) -> torch.Tensor:
    return model(
        past_chord_ids=batch["past_chord_ids"].to(device),
        future_chord_ids=batch["future_chord_ids"].to(device),
        melody_chroma=batch["melody_chroma"].to(device),
        target_chord_ids=batch["target_chord_ids"].to(device),
        past_len=batch["past_len"].to(device),
        future_len=batch["future_len"].to(device),
    )


def evaluate(
    model: ChordHarmonizer,
    loader: DataLoader,
    pad_id: int,
    vocab_size: int,
    device: str,
) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            logits = forward_batch(model, batch, device)
            loss = compute_loss(
                logits,
                batch["target_chord_ids"].to(device),
                batch["melody_len"].to(device),
                pad_id,
                vocab_size,
            )
            total += loss.item()
            n += 1
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",           default="data/processed/wikifonia")
    parser.add_argument("--output_dir",         default="checkpoints/harmonizer")
    parser.add_argument("--pretrained_decoder", default=None,
                        help="Path to a ChordTransformer checkpoint to initialize the decoder")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int,   default=500)
    parser.add_argument("--d_model",      type=int,   default=256)
    parser.add_argument("--n_heads",      type=int,   default=4)
    parser.add_argument("--n_layers",     type=int,   default=4)
    parser.add_argument("--window_size",  type=int,   default=32)
    parser.add_argument("--val_split",    type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = ChordTokenizer()
    pad_id = tokenizer.token2id["[PAD]"]

    collate = partial(harmonizer_collate_fn, pad_id=pad_id)

    dataset = HarmonizerDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        window_size=args.window_size,
        augment=True,
    )

    val_size   = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=2, pin_memory=True,
    )

    config = HarmonizerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.window_size * 3 + 4,
        dropout=0.1,
    )

    model = ChordHarmonizer(config).to(args.device)

    if args.pretrained_decoder:
        model.load_pretrained_decoder(args.pretrained_decoder, device=args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = args.device == "cuda"
    scaler  = GradScaler("cuda") if use_amp else None

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0

        for batch in train_loader:
            current_lr = get_lr(global_step, args.warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            with autocast("cuda", enabled=use_amp):
                logits = forward_batch(model, batch, args.device)
                loss = compute_loss(
                    logits,
                    batch["target_chord_ids"].to(args.device),
                    batch["melody_len"].to(args.device),
                    pad_id,
                    config.vocab_size,
                )

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n += 1
            global_step += 1

        avg_train = total_loss / n
        val_loss  = evaluate(model, val_loader, pad_id, config.vocab_size, args.device)
        ppl       = math.exp(val_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs}  "
            f"train={avg_train:.4f}  val={val_loss:.4f}  "
            f"ppl={ppl:.2f}  lr={current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> Saved best model (val={val_loss:.4f})")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": config,
    }, os.path.join(args.output_dir, "final_model.pt"))


if __name__ == "__main__":
    main()
