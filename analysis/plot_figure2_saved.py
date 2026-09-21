"""Redraw the original Figure 2 from checked saved aggregates, original styling."""

from experiments.paths import OUTPUT
from pathlib import Path
import os, json
import matplotlib
import matplotlib.pyplot as plt


def main():
    R = Path(__file__).resolve().parents[1]
    os.environ["MPLCONFIGDIR"] = str(OUTPUT / "matplotlib")

    matplotlib.use("Agg")

    z = json.loads((R / "experiments/figure2/aggregates.json").read_text())
    col = {
        "uniform": "#888888",
        "leverage": "#1f77b4",
        "leverage-w": "#9467bd",
        "D-optimal": "#2ca02c",
        "SpanCancel": "#d62728",
    }
    lab = {
        "uniform": "uniform",
        "leverage": "leverage",
        "leverage-w": "leverage (weighted)",
        "D-optimal": "$D$-optimal",
        "SpanCancel": "SpanCancel (ours)",
    }
    plt.rcdefaults()
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.1))
    for p, panel, ti in [
        (ax[0], "training", r"training loss ratio $L_D(w_F)/L^\star_D$"),
        (ax[1], "test", "held out test-MSE ratio"),
    ]:
        for m in col:
            a = sorted((float(k), v) for k, v in z[panel][m].items())
            p.plot(
                [x for x, y in a],
                [y for x, y in a],
                marker="o",
                ms=3.5,
                lw=1.6,
                color=col[m],
                label=lab[m],
            )
        p.axhline(1, ls=":", c="k", lw=0.9)
        p.set_yscale("log")
        p.set_xlabel("budget $n/d$")
        p.set_title(ti, fontsize=9)
        p.grid(alpha=0.25, which="both")
    ax[0].set_ylabel("ratio (log scale)")
    ax[0].legend(fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure2_reproduced.pdf", bbox_inches="tight")
    print("Redrew Figure 2 from all 80 checked aggregate points.")


if __name__ == "__main__":
    main()
