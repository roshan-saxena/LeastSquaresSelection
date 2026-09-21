from experiments.paths import OUTPUT
from pathlib import Path
import os, json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def main():
    R = Path(__file__).resolve().parents[1]
    os.environ["MPLCONFIGDIR"] = str(OUTPUT / "matplotlib")

    matplotlib.use("Agg")

    q = json.loads((R / "experiments/figure2/correspondence.json").read_text())
    assert q["all_80_vertices_checked"] and q["maximum_relative_difference"] < 1e-4
    z = json.loads((R / "experiments/figure2/aggregates.json").read_text())
    xs = [1.5, 1.75, 2, 2.5, 3]
    plt.rcdefaults()
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.1))
    for p, panel, title in zip(
        ax,
        ["training", "test"],
        [
            r"(a) training loss ratio $L_D(w_F)/L_D^\star$",
            r"(b) held-out test MSE ratio",
        ],
    ):
        for m, vals in z[panel].items():
            if m != "SpanCancel":
                assert min(vals[str(float(x))] for x in xs) > 1.035
        ys = [z[panel]["SpanCancel"][str(float(x))] for x in xs]
        p.plot(
            xs,
            ys,
            marker="o",
            ms=3.5,
            lw=1.6,
            color="#d62728",
            label="SpanCancel (ours)",
        )
        p.axhline(1, ls=":", c="k", lw=0.9)
        p.set_xlabel(r"budget $n/d$")
        p.set_title(title, fontsize=9)
        p.grid(alpha=0.25, which="both")
        p.set_xlim(1.45, 3.05)
        p.set_ylim(0.995, 1.035)
        p.set_xticks(xs)
        p.set_yticks([1, 1.01, 1.02, 1.03])
    ax[0].set_ylabel("ratio (linear scale)")
    ax[0].legend(fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure3.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "Figure3.png", bbox_inches="tight", dpi=180)
    print("Saved Figure 3 using actual reproduction aggregates.")


if __name__ == "__main__":
    main()
