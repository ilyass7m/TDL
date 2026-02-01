"""
Models and conversion utilities.

Implements:
- ShallowNet: 1-hidden-layer ReLU MLP (binary classifier, outputs logits).
- PaperDeepNet: narrow/deep architecture used to instantiate the width->depth conversion
  (Appendix C.1 in Vardi et al., 2022).
- convert_shallow_to_paperdeep: deterministic construction that maps a trained ShallowNet
  into a PaperDeepNet computing (numerically) the same function.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowNet(nn.Module):
    """Depth-2 ReLU MLP: Linear(d->n) -> ReLU -> Linear(n->1)."""
    def __init__(self, d: int, n: int):
        super().__init__()
        self.fc1 = nn.Linear(d, n)
        self.fc2 = nn.Linear(n, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class PaperDeepNet(nn.Module):
    """
    Width m = 2d+3, depth = 2n+2 (affine layers, ReLU after all but final output layer).

    State layout:
    [ReLU(x_1..x_d), ReLU(-x_1..-x_d), S_plus, S_minus, scratch_t]
    """
    def __init__(self, d: int, n: int):
        super().__init__()
        self.d = d
        self.n = n
        self.m = 2 * d + 3
        self.layers = nn.ModuleList([nn.Linear(d, self.m)] + [nn.Linear(self.m, self.m) for _ in range(2 * n)])
        self.out = nn.Linear(self.m, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.layers[0](x))
        for layer in self.layers[1:]:
            h = F.relu(layer(h))
        return self.out(h)


class PaperDeepNetScratch(PaperDeepNet):
    """
    Same architecture as PaperDeepNet, but with a training-friendly initialization
    (near-identity internal layers) to stabilize optimization.
    """
    def __init__(self, d: int, n: int, noise_std: float = 1e-3):
        super().__init__(d, n)
        self.reset_parameters(noise_std=noise_std)

    def reset_parameters(self, noise_std: float = 1e-3) -> None:
        # First layer: Kaiming init
        nn.init.kaiming_uniform_(self.layers[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.layers[0].bias)

        # Internal layers: identity + tiny noise (helps gradient flow in deep MLPs)
        for L in self.layers[1:]:
            with torch.no_grad():
                L.weight.zero_()
                L.weight += torch.eye(self.m)
                if noise_std > 0:
                    L.weight += noise_std * torch.randn_like(L.weight)
                L.bias.zero_()

        # Output layer: Kaiming init
        nn.init.kaiming_uniform_(self.out.weight, a=math.sqrt(5))
        nn.init.zeros_(self.out.bias)


def convert_shallow_to_paperdeep(shallow: ShallowNet) -> PaperDeepNet:
    """
    Deterministic construction converting a trained depth-2
    ReLU MLP into a width-(2d+3), depth-(2n+2) ReLU network that computes the same function.

    Notes:
    - Uses separate nonnegative accumulators S_plus and S_minus to avoid negative values
      (since ReLU is applied after every internal layer).
    - Uses a scratch unit t to compute each hidden activation sequentially.
    """
    d = shallow.fc1.in_features
    n = shallow.fc1.out_features
    deep = PaperDeepNet(d, n)
    m = deep.m

    idx_Sp = 2 * d
    idx_Sn = 2 * d + 1
    idx_t = 2 * d + 2

    W = shallow.fc1.weight.detach().clone()          # (n, d)
    b = shallow.fc1.bias.detach().clone()            # (n,)
    u = shallow.fc2.weight.detach().clone().view(-1) # (n,)
    b_out = shallow.fc2.bias.detach().clone().view(-1)  # (1,)

    with torch.no_grad():
        # Layer 0: [I; -I; 0; 0; 0] so after ReLU we get [ReLU(x), ReLU(-x), 0, 0, 0].
        L0 = deep.layers[0]
        L0.weight.zero_()
        L0.bias.zero_()
        L0.weight[0:d, :] = torch.eye(d)
        L0.weight[d:2 * d, :] = -torch.eye(d)

        # Initialize internal layers to identity.
        for L in deep.layers[1:]:
            L.weight.zero_()
            L.bias.zero_()
            L.weight.copy_(torch.eye(m))

        # For each hidden neuron i, use two layers:
        # compute t = ReLU(<w_i, x> + b_i)
        # accumulate u_i * t into S_plus or S_minus, then clear t
        for i in range(n):
            wi = W[i]
            bi = float(b[i].item())
            ui = float(u[i].item())

            term = deep.layers[1 + 2 * i]
            term.weight[idx_t, :] = 0.0
            term.bias[idx_t] = bi
            # <w, x> = <w, ReLU(x)> - <w, ReLU(-x)>
            term.weight[idx_t, 0:d] = wi
            term.weight[idx_t, d:2 * d] = -wi
            term.weight[idx_t, idx_t] = 0.0 

            acc = deep.layers[1 + 2 * i + 1]
            # Clear scratch after accumulation
            acc.weight[idx_t, :] = 0.0
            acc.bias[idx_t] = 0.0

            if ui >= 0:
                acc.weight[idx_Sp, idx_t] = ui
            else:
                acc.weight[idx_Sn, idx_t] = abs(ui)

        # Final output: S_plus - S_minus + b_out
        deep.out.weight.zero_()
        deep.out.bias.copy_(b_out)
        deep.out.weight[0, idx_Sp] = 1.0
        deep.out.weight[0, idx_Sn] = -1.0

    return deep


def nonzero_params(model: nn.Module, tol: float = 1e-12) -> tuple[int, int]:
    """Return (count_nonzero, count_total) across all parameters."""
    nz, total = 0, 0
    for p in model.parameters():
        arr = p.detach().cpu().numpy().ravel()
        total += arr.size
        nz += int(np.sum(np.abs(arr) > tol))
    return nz, total
