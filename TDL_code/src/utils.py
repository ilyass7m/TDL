"""
Utility helpers: seeding, metrics, training loop.
"""
from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def accuracy_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return float((preds == y_true).float().mean().item())


def train_full_batch(
    model: torch.nn.Module,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xte: torch.Tensor,
    yte: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    grad_clip: float | None = None,
    log_every: int = 50,
) -> list[dict]:
    """
    Full-batch training with Adam and BCEWithLogitsLoss.
    Returns a history list of dicts containing train/test metrics at logging epochs.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    hist: list[dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if ep == 1 or ep % log_every == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                tr_logits = model(Xtr)
                te_logits = model(Xte)
                tr_loss = float(loss_fn(tr_logits, ytr).item())
                te_loss = float(loss_fn(te_logits, yte).item())
                tr_acc = accuracy_from_logits(tr_logits, ytr)
                te_acc = accuracy_from_logits(te_logits, yte)
            hist.append(
                dict(epoch=ep, train_loss=tr_loss, test_loss=te_loss, train_acc=tr_acc, test_acc=te_acc)
            )
    return hist
