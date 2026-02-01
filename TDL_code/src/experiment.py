"""
Implements two experiments:
A) Train a shallow/wide depth-2 ReLU MLP; convert it to a narrow/deep network using the paper's construction;
   verify functional equivalence and compare generalization (overfitting).
B) Train the same narrow/deep architecture from scratch and compare.

Outputs:
- results/config.json
- results/history_shallow.csv
- results/history_deep_scratch.csv
- results/summary.csv
- figures/train_acc.pdf, figures/test_acc.pdf, figures/train_loss.pdf, figures/test_loss.pdf
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import ShallowNet, PaperDeepNetScratch, convert_shallow_to_paperdeep, nonzero_params
from utils import set_seed, train_full_batch, accuracy_from_logits


@dataclass
class Config:
    seed: int = 7
    n_samples: int = 600
    noise: float = 0.25
    train_size: int = 32
    flip_p: float = 0.4
    hidden_width: int = 32
    shallow_epochs: int = 8000
    shallow_lr: float = 0.01
    deep_epochs: int = 2000
    deep_lr: float = 0.002
    grad_clip: float = 5.0


def plot_curves(df_sh: pd.DataFrame, df_deep: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(ycol: str, filename: str, ylabel: str) -> None:
        plt.figure()
        plt.plot(df_sh["epoch"], df_sh[ycol], label="Shallow (trained)")
        plt.plot(df_deep["epoch"], df_deep[ycol], label="Deep (trained from scratch)")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.close()

    save("train_acc", "train_acc.pdf", "Training accuracy")
    save("test_acc", "test_acc.pdf", "Test accuracy")
    save("train_loss", "train_loss.pdf", "Training loss")
    save("test_loss", "test_loss.pdf", "Test loss")


def eval_metrics(model: torch.nn.Module, Xtr, ytr_noisy, ytr_clean, Xte, yte) -> dict:
    loss_fn = nn.BCEWithLogitsLoss()
    model.eval()
    with torch.no_grad():
        tr_logits = model(Xtr)
        te_logits = model(Xte)
        tr_loss = float(loss_fn(tr_logits, ytr_noisy).item())
        te_loss = float(loss_fn(te_logits, yte).item())
        tr_acc = accuracy_from_logits(tr_logits, ytr_noisy)
        te_acc = accuracy_from_logits(te_logits, yte)
        tr_acc_clean = accuracy_from_logits(tr_logits, ytr_clean)
    return dict(train_loss=tr_loss, test_loss=te_loss, train_acc=tr_acc, test_acc=te_acc, train_acc_clean=tr_acc_clean)


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    # dataset
    X, y = make_moons(n_samples=cfg.n_samples, noise=cfg.noise, random_state=cfg.seed)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train_clean, y_test = train_test_split(
        X, y, train_size=cfg.train_size, random_state=cfg.seed, stratify=y
    )

    # label noise on train
    rng = np.random.RandomState(cfg.seed)
    flip_mask = rng.rand(len(y_train_clean)) < cfg.flip_p
    y_train_noisy = y_train_clean.copy()
    y_train_noisy[flip_mask] = 1 - y_train_noisy[flip_mask]
    num_flipped = int(flip_mask.sum())

    # torch tensors
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    ytr = torch.tensor(y_train_noisy, dtype=torch.float32).view(-1, 1)
    ytr_clean = torch.tensor(y_train_clean, dtype=torch.float32).view(-1, 1)
    yte = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # train shallow, then convert
    shallow = ShallowNet(d=2, n=cfg.hidden_width)
    hist_sh = train_full_batch(
        shallow, Xtr, ytr, Xte, yte,
        epochs=cfg.shallow_epochs, lr=cfg.shallow_lr, grad_clip=cfg.grad_clip, log_every=100
    )
    df_sh = pd.DataFrame(hist_sh)
    df_sh.to_csv("results/history_shallow.csv", index=False)

    deep_conv = convert_shallow_to_paperdeep(shallow).eval()

    # Equivalence check in float64 for numerical stability
    shallow64 = ShallowNet(2, cfg.hidden_width).double()
    with torch.no_grad():
        shallow64.fc1.weight.copy_(shallow.fc1.weight.double())
        shallow64.fc1.bias.copy_(shallow.fc1.bias.double())
        shallow64.fc2.weight.copy_(shallow.fc2.weight.double())
        shallow64.fc2.bias.copy_(shallow.fc2.bias.double())
    deep_conv64 = convert_shallow_to_paperdeep(shallow64).double().eval()
    X_all = torch.tensor(X, dtype=torch.float64)
    with torch.no_grad():
        diff = (shallow64(X_all) - deep_conv64(X_all)).abs()
    equiv = dict(max_abs_diff_float64=float(diff.max().item()), mean_abs_diff_float64=float(diff.mean().item()))

    # train deep architecture from scratch
    deep_scratch = PaperDeepNetScratch(d=2, n=cfg.hidden_width, noise_std=1e-3)
    hist_deep = train_full_batch(
        deep_scratch, Xtr, ytr, Xte, yte,
        epochs=cfg.deep_epochs, lr=cfg.deep_lr, grad_clip=cfg.grad_clip, log_every=50
    )
    df_deep = pd.DataFrame(hist_deep)
    df_deep.to_csv("results/history_deep_scratch.csv", index=False)

    # plots
    plot_curves(df_sh, df_deep, Path("figures"))

    # summary
    def count_params(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    nz_conv, _ = nonzero_params(deep_conv64, tol=1e-12)

    metrics_sh = eval_metrics(shallow, Xtr, ytr, ytr_clean, Xte, yte)
    metrics_conv = eval_metrics(deep_conv, Xtr, ytr, ytr_clean, Xte, yte)
    metrics_deep = eval_metrics(deep_scratch, Xtr, ytr, ytr_clean, Xte, yte)

    summary = pd.DataFrame([
        dict(model="Shallow (trained)", depth=2, width=cfg.hidden_width,
             params_dense=count_params(shallow), params_nonzero=count_params(shallow), **metrics_sh),
        dict(model="Deep (converted from shallow)", depth=2*cfg.hidden_width+2, width=2*2+3,
             params_dense=count_params(deep_conv), params_nonzero=nz_conv, **metrics_conv),
        dict(model="Deep (trained from scratch)", depth=2*cfg.hidden_width+2, width=2*2+3,
             params_dense=count_params(deep_scratch), params_nonzero=count_params(deep_scratch), **metrics_deep),
    ])
    summary.to_csv("results/summary.csv", index=False)

    # config
    out_cfg = {
        "seed": cfg.seed,
        "dataset": {"name": "make_moons", "n_samples": cfg.n_samples, "noise": cfg.noise, "standardize": True},
        "split": {"train_size": int(cfg.train_size), "test_size": int(len(X_test)), "random_state": cfg.seed, "stratify": True},
        "label_noise": {"flip_probability": float(cfg.flip_p), "num_flipped": num_flipped},
        "models": {
            "shallow": {"hidden_width": cfg.hidden_width, "depth": 2, "params": count_params(shallow)},
            "deep_converted": {"width": 2*2+3, "depth": 2*cfg.hidden_width+2,
                               "params_dense": count_params(deep_conv), "params_nonzero": nz_conv},
            "deep_scratch": {"width": 2*2+3, "depth": 2*cfg.hidden_width+2, "params": count_params(deep_scratch)},
        },
        "training": {
            "shallow": {"optimizer": "Adam", "lr": cfg.shallow_lr, "epochs": cfg.shallow_epochs, "grad_clip": cfg.grad_clip},
            "deep_scratch": {"optimizer": "Adam", "lr": cfg.deep_lr, "epochs": cfg.deep_epochs, "grad_clip": cfg.grad_clip,
                             "init": "identity+noise(1e-3) internal layers"},
        },
        "equivalence_check": equiv,
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/config.json", "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2)

    print(summary)
    print("Equivalence check:", equiv)


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    main()
