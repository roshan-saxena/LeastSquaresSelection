"""Aggregate saved runs into dataset summaries, statistical comparisons, and diagnostics."""

from experiments.paths import RESULTS, OUTPUT
from pathlib import Path
import json, math, itertools, sys, hashlib
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn import datasets
import statsmodels.api as sm


def main():
    sys.dont_write_bytecode = True

    ROOT = Path(__file__).resolve().parents[1]
    OUT = OUTPUT / "tables"
    OUT.mkdir(exist_ok=True, parents=True)
    METHODS = [
        "uniform",
        "leverage",
        "leverage-w",
        "vol-sampling",
        "D-optimal",
        "SpanCancel",
    ]
    raw = {
        p.stem: [json.loads(l) for l in p.read_text().splitlines()]
        for p in (RESULTS).glob("*.jsonl")
    }

    def save(df, name):
        df.to_csv(OUT / (name + ".csv"), index=False)

    def summary(g):
        a = np.asarray(g["ratio"], float)
        n = len(a)
        sd = float(np.std(a, ddof=1)) if n > 1 else np.nan
        err = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        return dict(
            replicates=n,
            mean=float(np.mean(a)),
            median=float(np.median(a)),
            geometric_mean=float(np.exp(np.mean(np.log(a)))),
            sd=sd,
            min=float(min(a)),
            max=float(max(a)),
            mean_ci95_low=float(np.mean(a) - err),
            mean_ci95_high=float(np.mean(a) + err),
            n_exact_stored_one=int(np.sum(a == 1)),
            n_near_one=int(g.near_one.sum()) if "near_one" in g else 0,
            n_numerical_certificate=(
                int(g.numerical_certificate.sum())
                if "numerical_certificate" in g
                else 0
            ),
        )

    flat = []
    restarts = []
    lpattempts = []
    errors = []
    for suite, rows in raw.items():
        for r in rows:
            if r.get("status") == "error":
                errors.append(dict(suite=suite, key=r["key"], error=r.get("error")))
            if r.get("ratio") is not None and r.get("status") == "ok":
                rr = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                rr["suite"] = suite
                flat.append(rr)
            for j, z in enumerate(r.get("restarts", [])):
                rr = {k: v for k, v in z.items() if not isinstance(v, (dict, list))}
                rr.update(
                    suite=suite,
                    run_key=r["key"],
                    data_id=r.get("data_id"),
                    method=r.get("method"),
                    protocol=r.get("protocol"),
                    n=r.get("n"),
                    d=r.get("d"),
                    replicate=r.get("replicate"),
                    run_reused=bool(r.get("reused_from")),
                )
                restarts.append(rr)
            for j, z in enumerate(r.get("lp_attempts", [])):
                rr = dict(
                    z,
                    suite=suite,
                    run_key=r["key"],
                    attempt=j,
                    method=r.get("method"),
                    protocol=r.get("protocol"),
                    accepted_branch=r.get("branch"),
                    within_budget=z.get("support_size", float("inf")) <= r.get("n", 0),
                    run_reused=bool(r.get("reused_from")),
                )
                lpattempts.append(rr)
    f = pd.DataFrame(flat)
    if f.empty:
        raise ValueError(f"No completed numerical runs found in {RESULTS}")
    for column in ["record_type", "setting", "budget_factor", "conjectured_bound"]:
        if column not in f:
            f[column] = np.nan
    save(f, "per_run_metrics")
    save(pd.DataFrame(restarts), "fallback_restarts")
    save(pd.DataFrame(lpattempts), "lp_attempts")
    save(pd.DataFrame(errors, columns=["suite", "key", "error"]), "errors")

    if "controlled" in raw:
        # Preserve a complete dataset-level benchmark table for both weighting protocols.
        c = f[(f.suite == "controlled") & f.protocol.isin(["native", "common"])]
        rows = []
        for key, g in c.groupby(
            ["dataset", "d", "N", "n", "budget_factor", "method", "protocol"],
            dropna=False,
        ):
            rr = dict(
                zip(
                    ["dataset", "d", "N", "n", "budget_factor", "method", "protocol"],
                    key,
                )
            )
            rr.update(summary(g))
            rr["deterministic_result_reused"] = bool(
                g.get("reused_from", pd.Series(dtype=str)).notna().any()
            )
            rows.append(rr)
        ds = pd.DataFrame(rows)
        save(ds, "dataset_benchmarks")
        macro = []
        for key, g in ds.groupby(["budget_factor", "method", "protocol"]):
            a = g["mean"].to_numpy()
            macro.append(
                dict(
                    budget_factor=key[0],
                    method=key[1],
                    protocol=key[2],
                    dataset_count=len(a),
                    arithmetic_mean_of_dataset_means=float(a.mean()),
                    median_of_dataset_means=float(np.median(a)),
                    geometric_mean_of_dataset_means=float(np.exp(np.log(a).mean())),
                    min_dataset_mean=float(a.min()),
                    max_dataset_mean=float(a.max()),
                )
            )
        macro = pd.DataFrame(macro)
        save(macro, "benchmark_macro")

        # Friedman and Nemenyi on ten dataset means, at the preselected 1.5 budget factor.
        omni = []
        post = []
        wil = []
        for protocol in ["native", "common"]:
            tab = ds[(ds.protocol == protocol) & (ds.budget_factor == 1.5)].pivot(
                index="dataset", columns="method", values="mean"
            )[METHODS]
            assert tab.shape == (10, 6) and np.isfinite(tab.values).all()
            # Tolerance-level near-one ties are rounded at 12 decimals for the rank omnibus.
            vals = np.round(tab.to_numpy(), 12)
            test = stats.friedmanchisquare(*vals.T)
            omni.append(
                dict(
                    protocol=protocol,
                    N_datasets=10,
                    statistic=float(test.statistic),
                    p_raw=float(test.pvalue),
                    hypothesis="equal paired method distributions/ranks",
                    tie_rule="average ranks after decimal12 rounding",
                )
            )
            ranks = stats.rankdata(vals, axis=1).mean(axis=0)
            K = vals.shape[1]
            N = vals.shape[0]
            nem = pd.DataFrame(
                stats.studentized_range.sf(
                    np.abs(ranks[:, None] - ranks[None, :])
                    / np.sqrt(K * (K + 1) / (6 * N))
                    * np.sqrt(2),
                    K,
                    np.inf,
                )
            )
            for i in range(6):
                for j in range(i + 1, 6):
                    post.append(
                        dict(
                            protocol=protocol,
                            method_a=METHODS[i],
                            method_b=METHODS[j],
                            p_nemenyi=float(nem.iloc[i, j]),
                            p_two_protocols=min(1.0, 2 * float(nem.iloc[i, j])),
                        )
                    )
            for method in METHODS[:-1]:
                delta = np.log(tab[method].to_numpy()) - np.log(
                    tab["SpanCancel"].to_numpy()
                )
                zero = np.abs(delta) <= 1e-10
                d = np.round(delta[~zero], 12)
                ranks = stats.rankdata(np.abs(d))
                positive = float(ranks[d > 0].sum())
                negative = float(ranks[d < 0].sum())
                W = min(positive, negative)
                if len(d):
                    total = float(ranks.sum())
                    poss = np.array(
                        [
                            sum(rank for rank, b in zip(ranks, bits) if b)
                            for bits in itertools.product([0, 1], repeat=len(d))
                        ]
                    )
                    p = float(np.mean(np.minimum(poss, total - poss) <= W + 1e-12))
                    rbc = (positive - negative) / total
                else:
                    p = 1.0
                    rbc = 0.0
                wil.append(
                    dict(
                        protocol=protocol,
                        method=method,
                        comparator="SpanCancel",
                        datasets=len(delta),
                        nonzero_datasets=len(d),
                        wins_baseline=int(np.sum(delta < -1e-10)),
                        ties=int(np.sum(zero)),
                        wins_SpanCancel=int(np.sum(delta > 1e-10)),
                        W=W,
                        p_exact_sign_permutation=p,
                        rank_biserial_advantage_SC=rbc,
                        median_log_ratio_difference=float(np.median(delta)),
                        mean_log_ratio_difference=float(np.mean(delta)),
                        median_fold_baseline_over_SC=float(np.exp(np.median(delta))),
                        mean_absolute_ratio_difference=float(
                            (tab[method] - tab["SpanCancel"]).mean()
                        ),
                    )
                )
        om = pd.DataFrame(omni)
        om["p_Holm_two_protocols"] = multipletests(om.p_raw, method="holm")[1]
        save(om, "friedman")
        wi = pd.DataFrame(wil)
        wi["p_Holm_ten_comparisons"] = multipletests(
            wi.p_exact_sign_permutation, method="holm"
        )[1]
        save(wi, "wilcoxon_effects")
        save(pd.DataFrame(post), "nemenyi")

    # Branch rates: repeats that reuse an identical deterministic result are retained in nominal
    # per-setting summaries, and separately removed in actual-computation denominators.
    branch = []
    for key, g in f[f.method == "SpanCancel"].groupby(
        ["suite", "protocol"], dropna=False
    ):
        for denominator, gg in [
            ("recorded experiment cells", g),
            (
                "actually executed cells",
                g[g.reused_from.isna()] if "reused_from" in g else g,
            ),
        ]:
            if not len(gg):
                continue
            fb = gg[gg.branch == "heuristic"]
            lp = gg[gg.branch == "lp"]
            sq = gg[gg.branch == "square_interpolation"]
            branch.append(
                dict(
                    suite=key[0],
                    protocol=key[1],
                    denominator=denominator,
                    total=len(gg),
                    LP_accepted=len(lp),
                    LP_percent=100 * len(lp) / len(gg),
                    iterative_fallback=len(fb),
                    iterative_fallback_percent=100 * len(fb) / len(gg),
                    square_interpolation=len(sq),
                    fallback_near_one=int(fb.near_one.sum()),
                    fallback_coefficient_recovery=int(fb.coefficient_recovery.sum()),
                    fallback_numerical_certificate=int(fb.numerical_certificate.sum()),
                    fallback_near_one_percent_of_fallback=(
                        100 * fb.near_one.sum() / len(fb) if len(fb) else np.nan
                    ),
                    fallback_coefficient_percent_of_fallback=(
                        100 * fb.coefficient_recovery.sum() / len(fb)
                        if len(fb)
                        else np.nan
                    ),
                )
            )
    save(pd.DataFrame(branch), "branch_rates")
    setting = []
    for key, g in f[
        (f.method == "SpanCancel")
        & (f.suite.isin(["controlled", "rff", "diagnostics", "correlation_correction"]))
    ].groupby(
        ["suite", "dataset", "protocol", "budget_factor", "setting"], dropna=False
    ):
        rr = dict(
            zip(["suite", "dataset", "protocol", "budget_factor", "setting"], key)
        )
        rr.update(
            runs=len(g),
            lp=int((g.branch == "lp").sum()),
            fallback=int((g.branch == "heuristic").sum()),
            square=int((g.branch == "square_interpolation").sum()),
            ratio_mean=g.ratio.mean(),
            ratio_max=g.ratio.max(),
            coefficient_error_relative_max=g.coefficient_error_relative.max(),
            cancellation_scaled_max=g.residual_cancellation_scaled.max(),
            condition_X_max=g.condition_X.max(),
            condition_H_estimate=g.condition_X.max() ** 2,
            support_size_min=g.support_size.min(),
            support_size_max=g.support_size.max(),
            support_rank_min=g.support_rank.min(),
            rank_detected_min=g.rank_detected.min(),
            rank_solver_min=g.rank_solver.min(),
            pool_size_mean=g.pool_size.mean() if "pool_size" in g else np.nan,
            pool_saturation_percent=(
                100 * np.mean(g.pool_size == g.N) if "pool_size" in g else np.nan
            ),
        )
        setting.append(rr)
    save(pd.DataFrame(setting), "branch_by_setting")

    # Held-out and RFF statistics use splits (one stochastic draw per split), no extra dataset count.
    for suite in ["rff"]:
        if suite not in raw:
            continue
        rows = []
        for key, g in f[f.suite == suite].groupby(
            ["dataset", "d", "N", "n", "budget_factor", "method"]
        ):
            rr = dict(zip(["dataset", "d", "N", "n", "budget_factor", "method"], key))
            rr.update(summary(g))
            rr.update(
                test_ratio_mean=g.test_ratio.mean(),
                test_ratio_sd=g.test_ratio.std(ddof=1),
                test_prediction_error_max=g.test_prediction_error.max(),
                coefficient_error_relative_max=g.coefficient_error_relative.max(),
                full_test_mse_mean=g.full_test_mse.mean(),
            )
            rows.append(rr)
        save(pd.DataFrame(rows), suite + "_summary")

    # Small searches and frontier best-found results remain distinct from global optima.
    search = []
    for suite in [
        "stress",
        "correlation_correction",
        "adversarial",
        "known_constructions",
    ]:
        for r in raw.get(suite, []):
            if r.get("record_type") in ["enumeration_summary", "adversarial_summary"]:
                search.append(r)
    save(
        pd.DataFrame(
            [
                {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                for r in search
            ]
        ),
        "search_coverage",
    )
    sr = f[
        (f.suite.isin(["stress", "correlation_correction"]))
        & (f.protocol == "native")
        & (f.method == "SpanCancel")
        & f.record_type.isna()
    ].copy()
    sr["gap_to_conjectured_bound"] = sr.conjectured_bound - sr.ratio
    save(sr, "low_budget_instances")
    front = f[(f.suite == "stress") & (f.record_type == "frontier")]
    br = []
    for key, g in front.groupby(["data_id", "d", "n"]):
        j = g.ratio.idxmin()
        r = g.loc[j]
        br.append(
            dict(
                data_id=key[0],
                d=key[1],
                n=key[2],
                best_ratio=r.ratio,
                best_key=r.key,
                best_method=r.method,
                best_protocol=r.protocol,
                conjectured_bound=r.conjectured_bound,
            )
        )
    save(pd.DataFrame(br), "frontier_best_found")

    if "runtime" in raw:
        rt = f[f.suite == "runtime"]
        rs = []
        for key, g in rt.groupby(["protocol", "N", "d", "n", "method"]):
            rr = dict(zip(["protocol", "N", "d", "n", "method"], key))
            rr["replicates"] = len(g)
            for col in [
                "full_fit_s",
                "residual_s",
                "candidate_s",
                "lp_s",
                "fallback_s",
                "validation_s",
                "selection_s",
                "total_s",
            ]:
                rr[col + "_mean"] = g[col].mean()
                rr[col + "_median"] = g[col].median()
                rr[col + "_max"] = g[col].max()
                rr[col + "_sd"] = g[col].std(ddof=1)
            rs.append(rr)
        save(pd.DataFrame(rs), "runtime_components")

    print(
        "Derived summaries:",
        len(list(OUT.glob("*.csv"))),
        f"CSV files in {OUT}",
    )


if __name__ == "__main__":
    main()
