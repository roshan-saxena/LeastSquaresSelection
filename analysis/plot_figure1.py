"""Plot the constructed lower bound in Figure 1."""

import argparse
import os
from fractions import Fraction
from pathlib import Path

from experiments.paths import OUTPUT


def value(d, n):
    """Evaluate B_d(n) using balanced integer partitions."""
    if n == d:
        return Fraction(d + 1)
    if n == 2 * d:
        return Fraction(1)
    k = n - d
    return 1 + max(
        Fraction(g - k)
        / sum(
            [Fraction(1, d // g + 1)] * (d % g)
            + [Fraction(1, d // g)] * (g - d % g)
        )
        for g in range(k + 1, d + 1)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.93,
            "axes.titlesize": 7.14,
            "axes.labelsize": 7.93,
            "xtick.labelsize": 7.93,
            "ytick.labelsize": 7.93,
            "axes.linewidth": 0.635,
            "grid.linewidth": 0.635,
        }
    )
    fig, ax = plt.subplots(figsize=(232 / 72, 162 / 72))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.175, top=0.88)

    for d, color in zip([4, 6, 8], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        ns = np.arange(d, 2 * d + 1)
        ys = [float(value(d, int(n))) for n in ns]
        ax.plot(ns / d, ys, "-o", color=color, lw=1.2, ms=3, label=rf"$d={d}$")
        for n in [d + 1, 2 * d - 1]:
            ax.plot(
                n / d,
                float(value(d, n)),
                marker="*",
                ms=9.4,
                mec="black",
                mew=0.65,
                mfc=color,
                linestyle="none",
                zorder=6,
            )

    ax.set_xlim(0.95, 2.05)
    ax.set_ylim(0.6, 9.4)
    ax.set_xticks(np.arange(1, 2.01, 0.2))
    ax.set_yticks([2, 4, 6, 8])
    ax.grid(alpha=0.25)
    ax.axhline(1, color="black", ls=":", lw=0.635)
    ax.axvline(1.5, color="#888888", ls="--", lw=0.6, zorder=0)
    ax.set_xlabel(r"budget $n/d$", labelpad=3)
    ax.set_title(r"Frontier lower bound $B_d(n)$", pad=5)
    ax.text(
        1.28, 8.93,
        "Equality conjectured\n" + r"$(1<n/d<1.5)$",
        ha="center", va="top", fontsize=6.35, linespacing=1.3,
    )
    ax.text(
        1.77, 8.93,
        "Equality proved\n" + r"$(1.5\leq n/d\leq2)$",
        ha="center", va="top", fontsize=6.35, linespacing=1.3,
    )
    ax.legend(
        loc="upper right", bbox_to_anchor=(0.96, 0.76), fontsize=7.0,
        frameon=False, handlelength=1.5, labelspacing=0.35,
    )
    fig.savefig(output / "Figure1.png", dpi=400)
    fig.savefig(output / "Figure1.pdf")
    plt.close(fig)
    print(f"Saved Figure 1 to {output}.")


if __name__ == "__main__":
    main()
