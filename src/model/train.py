"""
Phase 2 — Training loop for the chord language model.

Objective : causal language modeling (predict next chord)
Optimizer : Adam, lr ~1e-4 with linear warmup
Metrics   : validation perplexity, next-chord accuracy, note-level accuracy
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.transformer import ChordTransformer


def train(
    model: ChordTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-4,
    warmup_steps: int = 1000,
    device: str = "cuda",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = PAD

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids[:, :-1])          # (B, T-1, V)
            targets = input_ids[:, 1:].contiguous()    # (B, T-1)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}  val_loss={val_loss:.4f}  ppl={val_loss.exp():.2f}")


def evaluate(model, loader, criterion, device) -> torch.Tensor:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids[:, :-1])
            targets = input_ids[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            n += 1
    return torch.tensor(total_loss / n)
