"""Plot loss ratios across budgets for the two fixed block constructions."""

from experiments.paths import OUTPUT
from pathlib import Path
import sys, os, json, hashlib
import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt


def main():
    sys.dont_write_bytecode = True
    R = Path(__file__).resolve().parents[1]
    E = OUTPUT
    os.environ["MPLCONFIGDIR"] = str(OUTPUT / "matplotlib")

    matplotlib.use("Agg")

    P = pd.read_csv(E / "tables/per_run_metrics.csv", low_memory=False)
    B = pd.read_csv(E / "tables/frontier_best_found.csv")
    # Plot styling matches the paper figures.
    COL = {
        "uniform": "#888888",
        "leverage": "#1f77b4",
        "leverage-w": "#9467bd",
        "D-optimal": "#2ca02c",
        "SpanCancel": "#d62728",
    }
    LAB = {
        "uniform": "uniform",
        "leverage": "leverage",
        "leverage-w": "leverage (weighted)",
        "D-optimal": "$D$-optimal",
        "SpanCancel": "SpanCancel (ours)",
    }
    plt.rcdefaults()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    records = []
    for ax, d, letter in zip(axes, [4, 6], ["a", "b"]):
        z = P[(P.data_id == f"frontier_blocks_{d}") & (P.record_type == "frontier")]
        assert set(z.n) == set(range(d, 2 * d + 1))
        assert z.N.nunique() == 1
        for m, c in COL.items():
            q = z[(z.method == m) & (z.protocol == "native")].sort_values("n")
            assert len(q) == d + 1
            ax.plot(q.n / d, q.ratio, marker="o", ms=3.5, lw=1.6, color=c, label=LAB[m])
            records += q.to_dict("records")
        best = z.groupby("n").ratio.min()
        bb = B[B.data_id == f"frontier_blocks_{d}"].set_index("n").sort_index()
        assert np.allclose(best.values, bb.best_ratio.values, rtol=0, atol=1e-14)
        ax.plot(
            best.index / d,
            best.values,
            marker="o",
            ms=5,
            markerfacecolor="none",
            markeredgewidth=0.8,
            lw=0.9,
            ls="--",
            color="k",
            label="best found",
            zorder=5,
        )
        ax.plot(
            bb.index / d,
            bb.conjectured_bound.values,
            ls="--",
            color="#555555",
            lw=0.9,
            label="worst-case reference",
            zorder=1,
        )
        ax.axhline(1, ls=":", c="k", lw=0.9)
        ax.set_yscale("log")
        ax.set_xlabel("budget $n/d$")
        ax.set_title(
            f"({letter}) orthogonal blocks: $d={d}$, $N={int(z.N.iloc[0])}$", fontsize=9
        )
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("ratio (log scale)")
    axes[0].legend(fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure4.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "Figure4.png", dpi=180, bbox_inches="tight")
    pd.DataFrame(records).to_csv(OUTPUT / "budget_frontier_rows.csv", index=False)
    print("Saved Figure 4 and its plotted values.")


if __name__ == "__main__":
    main()
