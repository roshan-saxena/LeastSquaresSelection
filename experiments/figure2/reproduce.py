"""Figure: training-loss and held-out test-MSE ratio versus budget n/d,
geometric mean over real datasets, for the five selectors. Writes
fig_selection.pdf in results/generated/figure2_rerun.
"""

import os
import json, hashlib, time, platform, sys
from pathlib import Path

INPUT = Path(__file__).parent
OUT = INPUT.resolve().parents[1] / "results/generated/figure2_rerun"
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from . import core

METHODS = {k: v for k, v in core.METHODS.items() if k != "vol-sampling"}

grid = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trace = (OUT / "per_run.jsonl").open("w")
    started = time.perf_counter()
    core.reset_rng()
    tr = {m: {g: [] for g in grid} for m in METHODS}
    te = {m: {g: [] for g in grid} for m in METHODS}
    for nm in ("diabetes", "california", "wine", "engel", "stackloss"):
        for seed in range(5):
            z = np.load(INPUT / f"input_{nm}_{seed}.npz")
            Xtr, Xte, ytr, yte = (z[k] for k in ("Xtr", "Xte", "ytr", "yte"))
            N, d = Xtr.shape
            w, Mf, bf, yy, L = core.stats(Xtr, ytr)
            np.savez_compressed(
                OUT / f"input_{nm}_{seed}.npz", Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte
            )
            print(
                f"start {nm} split={seed} N={N} d={d} elapsed={time.perf_counter()-started:.1f}",
                flush=True,
            )
            if L < 1e-9:
                continue
            mf = core.tmse(w, Xte, yte)
            for gv in grid:
                n = int(round(gv * d))
                if not (d <= n < N):
                    continue
                for mth, fn in METHODS.items():
                    reps = 5 if mth in core.STOCH else 1
                    a1, a2 = [], []
                    for _ in range(reps):
                        S, a = fn(Xtr, ytr, n, w)
                        wf = core.fit(Xtr, ytr, S, a)
                        a1.append(core.ratio(Xtr, ytr, S, a, Mf, bf, yy, L))
                        a2.append(core.tmse(wf, Xte, yte) / max(mf, 1e-12))
                        trace.write(
                            json.dumps(
                                dict(
                                    dataset=nm,
                                    split=seed,
                                    N=N,
                                    d=d,
                                    factor=gv,
                                    budget=n,
                                    method=mth,
                                    replicate=_,
                                    branch=(
                                        core.LAST_SC_BRANCH
                                        if mth == "SpanCancel"
                                        else None
                                    ),
                                    train_ratio=a1[-1],
                                    test_ratio=a2[-1],
                                    direct_train_ratio=float(
                                        np.sum((Xtr @ wf - ytr) ** 2) / L
                                    ),
                                    coefficient_error=float(
                                        np.linalg.norm(wf - w) / (1 + np.linalg.norm(w))
                                    ),
                                    support=list(map(int, S)),
                                    weights=list(map(float, a)),
                                )
                            )
                            + "\n"
                        )
                        trace.flush()
                    tr[mth][gv].append(np.mean(a1))
                    te[mth][gv].append(np.mean(a2))
    gm = lambda dd: {
        g: (np.exp(np.mean(np.log(np.maximum(v, 1e-9)))) if v else np.nan)
        for g, v in dd.items()
    }
    (OUT / "split_means.json").write_text(
        json.dumps(dict(training=tr, test=te), indent=2)
    )
    (OUT / "aggregates.json").write_text(
        json.dumps(
            {
                panel: {m: gm(dat[m]) for m in METHODS}
                for panel, dat in [("training", tr), ("test", te)]
            },
            indent=2,
        )
    )
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.1))
    for p, dat, ti in (
        (ax[0], tr, "training loss ratio $L_D(w_F)/L^\\star_D$"),
        (ax[1], te, "held out test-MSE ratio"),
    ):
        for mth in METHODS:
            g = gm(dat[mth])
            xs = [k for k in grid if not np.isnan(g[k])]
            p.plot(
                xs,
                [g[k] for k in xs],
                marker="o",
                ms=3.5,
                lw=1.6,
                color=col[mth],
                label=lab[mth],
            )
        p.axhline(1.0, ls=":", c="k", lw=0.9)
        p.set_yscale("log")
        p.set_xlabel("budget $n/d$")
        p.set_title(ti, fontsize=9)
        p.grid(alpha=0.25, which="both")
    ax[0].set_ylabel("ratio (log scale)")
    ax[0].legend(fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    out = str(OUT / "fig_selection.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", os.path.normpath(out), flush=True)
    trace.close()
    import scipy, sklearn, statsmodels

    (OUT / "completed.json").write_text(
        json.dumps(
            dict(
                seconds=time.perf_counter() - started,
                python=sys.version,
                platform=platform.platform(),
                numpy=np.__version__,
                scipy=scipy.__version__,
                sklearn=sklearn.__version__,
                statsmodels=statsmodels.__version__,
                matplotlib=matplotlib.__version__,
                threads={
                    k: os.environ.get(k)
                    for k in [
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                    ]
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
