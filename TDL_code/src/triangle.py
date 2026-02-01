"""
Depth->width separation illustration (Telgarsky-style triangle function).

Produces:
- figures/triangle_L8.pdf
- results/depth_to_width_expensive_table.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def triangle(x: np.ndarray) -> np.ndarray:
    # ReLU representation of a triangular wave on [0,1]
    return np.maximum(0, 2 * x) - 2 * np.maximum(0, 2 * x - 1) + np.maximum(0, 2 * x - 2)


def compose_triangle(x: np.ndarray, L: int) -> np.ndarray:
    y = x.copy()
    for _ in range(L):
        y = triangle(y)
    return y


def main(out_dir: str = "figures", results_dir: str = "results", L: int = 8) -> None:
    out = Path(out_dir)
    res = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    xs = np.linspace(0, 1, 2000)
    ys = compose_triangle(xs, L)

    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel(f"g^{L}(x)")
    plt.tight_layout()
    plt.savefig(out / f"triangle_L{L}.pdf")
    plt.close()

    Ls = [4, 6, 8, 10]
    tbl = pd.DataFrame({
        "L": Ls,
        "linear_regions": [2 ** l for l in Ls],
        "min_width_depth2": [2 ** l - 1 for l in Ls],
    })
    tbl.to_csv(res / "depth_to_width_expensive_table.csv", index=False)


if __name__ == "__main__":
    main()
