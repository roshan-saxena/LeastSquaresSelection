"""Calculate the reported table values from numerical experiment records."""

from experiments.paths import RESULTS, OUTPUT
from pathlib import Path
import json, re, csv, os
import numpy as np, pandas as pd


def main():
    ROOT = Path(__file__).resolve().parents[1]
    T = OUTPUT / "tables"
    O = OUTPUT / "tables"
    O.mkdir(parents=True, exist_ok=True)
    raw = {
        p.stem: [json.loads(l) for l in p.read_text().splitlines()]
        for p in (RESULTS).glob("*.jsonl")
    }
    methods = [
        "uniform",
        "leverage",
        "leverage-w",
        "vol-sampling",
        "D-optimal",
        "SpanCancel",
    ]
    report = []
    differences = []
    num = re.compile(r"(?<![A-Za-z])[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?")

    def numbers(s):
        return [float(x) for x in num.findall(s.replace(r"\pm", " ± "))]

    def table(name, values):
        key = name.removesuffix(".tex")
        if (
            key.startswith("response_")
            and os.environ.get("MLWA_INCLUDE_RESPONSE") != "1"
        ):
            return
        expected = json.loads((ROOT / "checks/expected_tables.json").read_text())[key]
        if len(expected) != len(values):
            raise ValueError(f"{key}: unexpected row count")
        rows = []
        compare = RESULTS == ROOT / "results"
        for i, (old, newvals) in enumerate(zip(expected, values)):
            if len(old) != len(newvals) + 1:
                raise ValueError(f"{key}: unexpected column count in row {i}")
            cleaned = []
            for j, value in enumerate(newvals, 1):
                oldnum, newnum = numbers(old[j]), numbers(value)
                if compare and oldnum != newnum:
                    differences.append(
                        dict(table=key, row=i, column=j, old=old[j], new=value)
                    )
                    if len(oldnum) != len(newnum) or any(
                        abs(a - b) >= 5e-15 for a, b in zip(oldnum, newnum)
                    ):
                        raise ValueError(
                            f"{key}: value mismatch at row {i}, column {j}"
                        )
                cleaned.append(
                    re.sub(
                        r"[{}$^]",
                        "",
                        value.replace(r"\pm", " ± ").replace(r"\dagger", ""),
                    )
                )
            rows.append([old[0], *cleaned])
        headers = TABLE_COLUMNS[key]
        with (O / f"{key}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        report.append(
            dict(
                table=key,
                rows=len(values),
                numeric_cells=sum(map(len, values)),
                published_numbers_match=True if compare else None,
            )
        )

    TABLE_COLUMNS = {
        "table_benchmark": [
            "method",
            "native_1.25",
            "native_1.5",
            "native_2",
            "common_1.25",
            "common_1.5",
            "common_2",
        ],
        "table_distribution": [
            "method",
            "native_mean",
            "native_median",
            "native_geometric_mean",
            "common_mean",
            "common_median",
            "common_geometric_mean",
        ],
        "table_datasetmain": [
            "dataset",
            "weights",
            "uniform",
            "leverage",
            "weighted_leverage",
            "volume",
            "D-optimal",
            "SpanCancel",
        ],
        "table_rff": [
            "dimension",
            "budget",
            "uniform",
            "leverage",
            "weighted_leverage",
            "D-optimal",
            "SpanCancel_training",
            "SpanCancel_test",
        ],
        "table_reliability": [
            "cohort",
            "settings",
            "LP_passes",
            "fallback_calls",
            "fallback_near_one",
            "fallback_passes",
            "max_relative_coefficient_error",
        ],
        "response_cohort_table": [
            "cohort",
            "settings",
            "LP_passes",
            "fallback_calls",
            "fallback_near_one",
            "fallback_passes",
            "max_relative_coefficient_error",
        ],
        "table_runtime": [
            "rows",
            "dimension",
            "budget",
            "full_fit_ms",
            "residual_ms",
            "selection_ms",
            "LP_ms",
            "fallback_ms",
            "validation_ms",
            "total_mean_ms",
            "total_max_ms",
        ],
        "table_known": [
            "construction",
            "dimension",
            "rows",
            "budget",
            "analytical_reference",
            "best_found",
            "SpanCancel",
        ],
        "table_sensitivity": [
            "setting",
            "LP_passes",
            "max_ratio_difference",
            "minimum_support_rank",
        ],
    }

    a = pd.read_csv(T / "benchmark_macro.csv")
    b = pd.read_csv(T / "dataset_benchmarks.csv")
    q = pd.read_csv(T / "rff_summary.csv")
    rt = pd.read_csv(T / "runtime_components.csv")
    cell = lambda df, **kw: df.loc[
        np.logical_and.reduce([df[k].eq(v) for k, v in kw.items()])
    ].iloc[0]
    table(
        "table_benchmark.tex",
        [
            [
                f"{cell(a,method=m,protocol=p,budget_factor=f).geometric_mean_of_dataset_means:.3f}"
                for p in ["native", "common"]
                for f in [1.25, 1.5, 2.0]
            ]
            for m in methods
        ],
    )
    table(
        "table_distribution.tex",
        [
            [
                f"{cell(a,method=m,protocol=p,budget_factor=1.5)[k]:.3f}"
                for p in ["native", "common"]
                for k in [
                    "arithmetic_mean_of_dataset_means",
                    "median_of_dataset_means",
                    "geometric_mean_of_dataset_means",
                ]
            ]
            for m in methods
        ],
    )
    rows = []
    for ds in [
        "breast",
        "ccard",
        "copper",
        "diabetes",
        "grunfeld",
        "longley",
        "scotland",
        "stackloss",
        "statecrime",
        "wine",
    ]:
        for p in ["native", "common"]:
            vals = [p.title()]
            for m in methods:
                r = cell(b, dataset=ds, protocol=p, budget_factor=1.5, method=m)
                vals.append(
                    f'${r["mean"]:.3f}^{{\\dagger}}$'
                    if m == "SpanCancel" or (m == "D-optimal" and p == "native")
                    else f'${r["mean"]:.3f}\\pm{r.sd:.3f}$'
                )
            rows.append(vals)
    table("table_datasetmain.tex", rows)
    rows = []
    for d in [51, 101, 201]:
        for f in [1.25, 1.5, 2.0]:
            sc = cell(q, d=d, budget_factor=f, method="SpanCancel")
            vals = [str(int(sc.n))]
            for m in methods[:3] + ["D-optimal"]:
                r = cell(q, d=d, budget_factor=f, method=m)
                vals.append(f'${r["mean"]:.2f}\\pm{r.sd:.2f}$')
            vals += [f'${sc["mean"]:.3f}$', f"${sc.test_ratio_mean:.3f}$"]
            rows.append(vals)
    table("table_rff.tex", rows)
    ordinary = []
    for suite in ["controlled", "rff", "stress", "correlation_correction"]:
        seen = set()
        for r in raw[suite]:
            if (
                r.get("status") != "ok"
                or r.get("method") != "SpanCancel"
                or r.get("protocol") != "native"
                or r.get("record_type")
                or r["n"] <= r["d"]
            ):
                continue
            key = (r["data_id"], r["n"])
            if key in seen:
                continue
            seen.add(key)
            ordinary.append(dict(r, suite=suite))
    od = pd.DataFrame(ordinary)
    groups = [od[od.suite.eq("controlled")]] + [
        od[od.suite.eq("rff") & od.d.eq(d)] for d in [51, 101, 201]
    ]
    stress = od[od.suite.isin(["stress", "correlation_correction"])]
    groups += [
        stress[stress.dataset.eq("blocks")],
        stress[~stress.dataset.isin(["blocks", "overlap"])],
        stress[stress.dataset.eq("overlap")],
    ]

    def branch(g):
        h = g[g.branch.eq("heuristic")]
        lp = g[g.branch.eq("lp")]
        return [
            str(len(g)),
            str(int(lp.numerical_certificate.sum())),
            str(len(h)),
            str(int(h.near_one.sum())),
            str(int(h.numerical_certificate.sum())),
            f"{g.coefficient_error_relative.max():.2e}",
        ]

    table("table_reliability.tex", [branch(g) for g in groups])
    cg = [od[od.suite.eq("controlled") & od.d.eq(d)] for d in [4, 5, 6, 7, 8]] + [
        od[od.suite.eq("rff") & od.d.eq(d)] for d in [51, 101, 201]
    ]
    for ds in [
        "blocks",
        "correlated",
        "ill_conditioned",
        "overlap",
        "perturbed_blocks",
        "random",
        "structured_residual",
        "correlated_preserved",
    ]:
        cg.append(stress[stress.dataset.eq(ds)])
    front = pd.DataFrame(
        [
            r
            for r in raw["stress"]
            if r.get("record_type") == "frontier"
            and r.get("method") == "SpanCancel"
            and r.get("protocol") == "native"
            and r.get("data_id") in ["frontier_blocks_4", "frontier_blocks_6"]
        ]
    )
    cg += [front[front.d.eq(d) & (front.n > front.d)] for d in [4, 6]]
    table("response_cohort_table.tex", [branch(g) for g in cg])
    rows = []
    for N, d, n in [
        (200, 30, 45),
        (1000, 30, 45),
        (5000, 30, 45),
        (1000, 30, 38),
        (1000, 30, 60),
        (1000, 10, 15),
        (1000, 60, 90),
        (5000, 60, 90),
        (22, 20, 21),
        (500, 20, 30),
    ]:
        r = cell(rt, N=N, d=d, n=n)
        selection = "candidate_s_mean"
        rows.append(
            [str(d), str(n)]
            + [
                f"{1000*r[k]:.2f}"
                for k in [
                    "full_fit_s_mean",
                    "residual_s_mean",
                    selection,
                    "lp_s_mean",
                    "fallback_s_mean",
                    "validation_s_mean",
                    "total_s_mean",
                    "total_s_max",
                ]
            ]
        )
    table("table_runtime.tex", rows)
    known = raw["known_constructions"]
    rows = []
    for r in known:
        if r.get("method") != "SpanCancel":
            continue
        s = next(
            x
            for x in known
            if x.get("record_type") == "enumeration_summary"
            and x["data_id"] == r["data_id"]
        )
        value = r["analytical_reference"]
        rows.append(
            [
                str(r["d"]),
                str(r["N"]),
                str(r["n"]),
                "--" if value is None else f"{value:.3f}",
                f'{s["best_ratio"]:.3f}',
                f'{r["ratio"]:.3f}',
            ]
        )
    table("table_known.tex", rows)
    dg = pd.DataFrame(
        [
            r
            for suite in ["diagnostics", "correlation_correction"]
            for r in raw[suite]
            if r.get("protocol") == "sensitivity"
        ]
    )
    base = dg[dg.setting.eq("default")][["data_id", "replicate", "ratio"]].rename(
        columns={"ratio": "default_ratio"}
    )
    matched = dg.merge(base, on=["data_id", "replicate"], validate="many_to_one")
    matched["difference"] = matched.ratio - matched.default_ratio
    spec = [
        (["default"], "0"),
        (["pool_factor=1"], ".4f"),
        (["pool_factor=2"], ".4f"),
        (["pool_factor=4", "pool_factor=16", "pool_factor=32"], ".2e"),
        (["rank_atol=0.0001"], ".3f"),
        (["rank_atol=1e-06"], ".6f"),
        (["rank_atol=1e-10", "rank_atol=1e-12"], "0"),
        (["solver_rcond=1e-08", "solver_rcond=1e-10", "solver_rcond=1e-14"], "0"),
        (["lp_tol=1e-07", "lp_tol=1e-10"], "0"),
        (["support_tol=1e-07", "support_tol=1e-11"], "0"),
        (["coefficient_tol=1e-05", "coefficient_tol=1e-09"], "0"),
        (["restarts=0", "restarts=1", "restarts=3", "restarts=12"], ".1e"),
        (["initialization=lognormal"], ".2e"),
    ]
    rows = []
    for settings, fmt in spec:
        g = matched[matched.setting.isin(settings)]
        counts = g.groupby("setting").numerical_certificate.sum()
        assert counts.nunique() == 1 and len(counts) == len(settings)
        delta = g.difference.abs().max()
        assert fmt != "0" or delta < 1e-15
        rows.append(
            [
                str(int(counts.iloc[0])),
                "0" if fmt == "0" else format(delta, fmt),
                str(int(g.support_rank.min())),
            ]
        )
    table("table_sensitivity.tex", rows)
    matched.to_csv(T / "sensitivity_matched.csv", index=False)
    restarts = []
    ranges = []
    for r in ordinary:
        if r["branch"] != "heuristic":
            continue
        z = [z for z in r["restarts"] if z.get("kind") != "uniform_evaluation"]
        restarts.extend(dict(z, dataset=r["dataset"], key=r["key"]) for z in z)
        # Within-call variation concerns final optimized returns, not the unevaluated starting guess.
        vals = [v["ratio"] for v in z]
        ranges.append(
            dict(dataset=r["dataset"], key=r["key"], ratio_range=max(vals) - min(vals))
        )
    st = pd.DataFrame(restarts)
    rg = pd.DataFrame(ranges)
    summary = {
        "ordinary_cells": len(ordinary),
        "LP": int(od.branch.eq("lp").sum()),
        "fallback": len(ranges),
        "optimized_starts": len(st),
        "converged_starts": int(st.success.sum()),
        "families": {},
    }
    for family, g in st.groupby("dataset"):
        v = rg[rg.dataset.eq(family)].ratio_range
        summary["families"][family] = {
            "starts": len(g),
            "converged": int(g.success.sum()),
            "median_range": float(v.median()),
            "p90_range": float(v.quantile(0.9)),
            "nfev_mean": float(g.nfev.mean()),
            "nfev_max": int(g.nfev.max()),
            "seconds_mean": float(g.seconds.mean()),
            "seconds_max": float(g.seconds.max()),
        }
    (OUTPUT / "retained_cohorts.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT / "table_validation.json").write_text(
        json.dumps(
            dict(tables=report, roundoff_reaggregation_differences=differences),
            indent=2,
        )
    )
    print(
        "Regenerated and verified",
        len(report),
        "tables;",
        sum(r["numeric_cells"] for r in report),
        "cells",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
