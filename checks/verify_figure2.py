"""Recompute original-protocol fits and aggregation from stored supports and weights."""

from experiments.paths import OUTPUT
from pathlib import Path
import json, sys, importlib.util
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from experiments.figure2 import core


def main():
    sys.dont_write_bytecode = True

    ROOT = Path(__file__).resolve().parents[1]
    F = ROOT / "experiments/figure2"
    O = OUTPUT
    rows = [json.loads(l) for l in (F / "per_run.jsonl").read_text().splitlines()]
    cache = {}
    checks = []
    sc = []
    with threadpool_limits(1):
        for r in rows:
            key = (r["dataset"], r["split"])
            if key not in cache:
                z = np.load(F / f"input_{key[0]}_{key[1]}.npz")
                X, y, Xt, yt = (z[k] for k in ["Xtr", "ytr", "Xte", "yte"])
                w, M, b, yy, L = core.stats(X, y)
                cache[key] = (X, y, Xt, yt, w, M, b, yy, L, core.tmse(w, Xt, yt))
            X, y, Xt, yt, w, M, b, yy, L, testfull = cache[key]
            S = r["support"]
            a = np.array(r["weights"])
            wf = core.fit(X, y, S, a)
            tr = core.ratio(X, y, S, a, M, b, yy, L)
            te = core.tmse(wf, Xt, yt) / max(testfull, 1e-12)
            checks.append(
                max(
                    abs(tr - r["train_ratio"]) / max(1, abs(r["train_ratio"])),
                    abs(te - r["test_ratio"]) / max(1, abs(r["test_ratio"])),
                )
            )
            if r["method"] == "SpanCancel":
                a = a / a.sum()
                res = X @ w - y
                C = X[S] * res[S, None]
                cancel = np.linalg.norm(C.T @ a)
                tol = 1e-9 * max(1, np.linalg.norm(C, 2))
                coef = np.linalg.norm(wf - w) / (1 + np.linalg.norm(w))
                near = abs(np.sum((X @ wf - y) ** 2) / L - 1) <= 1e-7
                sc.append(
                    dict(
                        dataset=r["dataset"],
                        d=r["d"],
                        n=r["budget"],
                        branch=r["branch"],
                        near=near,
                        joint=core.rank(X[S]) == r["d"]
                        and cancel <= tol
                        and coef <= 1e-7,
                    )
                )
    df = pd.DataFrame(rows)
    saved = json.loads((F / "aggregates.json").read_text())
    maxagg = 0
    for panel, col in [("training", "train_ratio"), ("test", "test_ratio")]:
        split = (
            df.groupby(["method", "factor", "dataset", "split"])[col]
            .mean()
            .reset_index()
        )
        for (m, f), g in split.groupby(["method", "factor"]):
            value = float(np.exp(np.log(g[col]).mean()))
            maxagg = max(maxagg, abs(value - saved[panel][m][str(float(f))]))
    sc = pd.DataFrame(sc)
    out = dict(
        fits_checked=len(rows),
        maximum_relative_discrepancy=max(checks),
        aggregate_max_absolute_difference=maxagg,
        SpanCancel_calls=len(sc),
        LP=int(sc.branch.eq("lp").sum()),
        square=int(sc.branch.eq("square").sum()),
        fallback=int(sc.branch.eq("heuristic").sum()),
        LP_joint=int(sc[sc.branch.eq("lp")].joint.sum()),
        fallback_near=int(sc[sc.branch.eq("heuristic")].near.sum()),
        fallback_joint=int(sc[sc.branch.eq("heuristic")].joint.sum()),
    )
    sc.to_csv(O / "figure2_checks.csv", index=False)
    group = (
        df[df.factor.eq(1.5)]
        .groupby(["dataset", "method", "split"])
        .train_ratio.mean()
        .reset_index()
    )
    group.groupby(["dataset", "method"]).train_ratio.agg(["mean", "std"]).to_csv(
        O / "figure2_dataset_table.csv"
    )
    (O / "figure2_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    assert max(checks) < 1e-9 and maxagg < 1e-9


if __name__ == "__main__":
    main()
