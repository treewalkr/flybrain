"""Plot CEM training curves from a run's history.json.

    .venv/bin/python -m flybrain.plot data/runs/best --out data/out/train_curve.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/out/train_curve.png"))
    a = p.parse_args()
    hist = json.loads((a.run / "history.json").read_text())
    gens = [h["gen"] for h in hist]
    best = [h["best"] for h in hist]
    elite = [h["elite_mean"] for h in hist]
    mean = [h["mean"] for h in hist]
    sigma = [h["sigma"] for h in hist]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(gens, best, label="best", color="#f0c040")
    ax.plot(gens, elite, label="elite mean", color="#50d080")
    ax.plot(gens, mean, label="population mean", color="#6070a0")
    ax.axhline(0, color="w", lw=0.5, alpha=0.4)
    ax.set_ylabel("episode reward (20 balls, +1/-1)")
    ax.set_title("CEM training of the readout over the frozen MaleCNS circuit")
    ax.legend(frameon=False)
    ax2.plot(gens, sigma, color="#c06060")
    ax2.set_ylabel("sigma")
    ax2.set_xlabel("CEM generation")
    for ax_ in (ax, ax2):
        ax_.spines[["top", "right"]].set_visible(False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
