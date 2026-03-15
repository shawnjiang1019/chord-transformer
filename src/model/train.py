"""
Phase 2 — Training loop for the chord language model.

Objective : causal language modeling (predict next chord)
Optimizer : AdamW, lr ~1e-4 with linear warmup
Metrics   : validation perplexity, next-chord accuracy, note-level accuracy
"""

import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from src.model.transformer import ChordTransformer


def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup: ramp from 0 to base_lr over warmup_steps."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def train(
    model: ChordTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-4,
    warmup_steps: int = 1000,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = GradScaler('cuda')
    use_amp = device == "cuda"

    os.makedirs(checkpoint_dir, exist_ok=True)
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            # Learning rate warmup
            current_lr = get_lr(global_step, warmup_steps, lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with autocast('cuda', enabled=use_amp):
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            n_train += 1
            global_step += 1

        avg_train_loss = total_train_loss / n_train
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"ppl={val_loss.exp():.2f}  "
            f"lr={current_lr:.6f}"
        )

        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss.item(),
                "global_step": global_step,
            }, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    # Save final model
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss.item(),
        "global_step": global_step,
    }, os.path.join(checkpoint_dir, "final_model.pt"))

    return model


def evaluate(model, loader, criterion, device) -> torch.Tensor:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            n += 1
    return torch.tensor(total_loss / n)
