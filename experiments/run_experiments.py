"""Reproducible experiment runner. Writes only within this isolated experiment folder."""

from pathlib import Path
import sys, os, time, json, hashlib, itertools, math, traceback, dataclasses, argparse

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import differential_evolution
from threadpoolctl import threadpool_limits
from spancancel.engine import (
    Config,
    Problem,
    asdict,
    cancellation,
    fallback,
    greedy_cancel,
    reweight,
    seed_for,
    select,
    vol,
)
from spancancel.datasets_local import (
    blocks,
    controlled,
    frontier,
    holdout,
    synthetic,
    whiten_svd,
)

ROOT = Path(
    os.environ.get(
        "MLWA_RUN_DIR",
        str(Path(__file__).resolve().parents[1] / "results/generated/rerun"),
    )
)
METHODS = [
    "uniform",
    "leverage",
    "leverage-w",
    "vol-sampling",
    "D-optimal",
    "SpanCancel",
]


class Log:
    def __init__(self, suite):
        for name in ("data", "results", "config"):
            (ROOT / name).mkdir(parents=True, exist_ok=True)
        self.suite = suite
        self.path = ROOT / "results" / f"{suite}.jsonl"
        self.data = ROOT / "data"
        self.data.mkdir(exist_ok=True)
        self.done = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r = json.loads(line)
                self.done[r["key"]] = r
        self.file = self.path.open("a", buffering=1)

    def save(self, key, row):
        if key in self.done:
            return self.done[key]
        row.update(
            suite=self.suite,
            key=key,
            recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self.file.write(json.dumps(row, allow_nan=True) + "\n")
        self.file.flush()
        self.done[key] = row
        return row

    def problem(self, key, X, y, meta, Xe=None, ye=None, cfg=Config()):
        path = self.data / f"{key}.npz"
        mpath = self.data / f"{key}.json"
        if not path.exists():
            arrays = dict(X=X, y=y)
            if Xe is not None:
                arrays.update(Xtest=Xe, ytest=ye)
            np.savez_compressed(path, **arrays)
            meta.update(
                shape=list(X.shape),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            mpath.write_text(json.dumps(meta, indent=2))
        return Problem(X, y, cfg, Xe, ye)

    def run(self, key, P, n, method, seed, extra, cfg=Config(), force=False):
        if key in self.done:
            return self.done[key]
        try:
            r = select(P, n, method, cfg, seed, force)
            r.update(extra, n=n, status="ok")
        except Exception as e:
            r = dict(
                extra,
                n=n,
                method=method,
                seed=seed,
                status="error",
                error=repr(e),
                traceback=traceback.format_exc(),
            )
        return self.save(key, r)


def controlled_suite():
    log = Log("controlled")
    for nm, X, y, meta in controlled():
        dataid = f"controlled_{nm}"
        P = log.problem(dataid, X, y, meta)
        for nd in (1.0, 1.25, 1.5, 2.0):
            n = round(nd * P.d)
            for rep in range(10):
                for method in METHODS:
                    key = f"{dataid}:{nd}:{rep}:{method}:native"
                    extra = dict(
                        dataset=nm,
                        data_id=dataid,
                        budget_factor=nd,
                        replicate=rep,
                        protocol="native",
                        replicate_unit="method draw on fixed dataset",
                        deterministic_native=method == "D-optimal",
                    )
                    prev = log.done.get(f"{dataid}:{nd}:0:{method}:native")
                    if (
                        key not in log.done
                        and rep > 0
                        and prev
                        and (
                            method == "D-optimal"
                            or (method == "SpanCancel" and prev.get("branch") == "lp")
                        )
                    ):
                        rr = dict(prev)
                        rr.update(
                            extra,
                            reused_from=prev["key"],
                            seed=seed_for("controlled", nm, nd, method, rep),
                        )
                        native = log.save(key, rr)
                    else:
                        native = log.run(
                            key,
                            P,
                            n,
                            method,
                            seed_for("controlled", nm, nd, method, rep),
                            extra,
                        )
                    key = f"{dataid}:{nd}:{rep}:{method}:common"
                    if key in log.done or native["status"] != "ok":
                        continue
                    ex = dict(extra, protocol="common")
                    if method == "SpanCancel":
                        r = dict(native)
                        r.update(
                            ex,
                            reused_from=native["key"],
                            common_note="already uses common optimizer; reused unchanged",
                        )
                    else:
                        try:
                            r = reweight(
                                P,
                                native,
                                Config(),
                                seed_for("common", nm, nd, method, rep),
                            )
                            r.update(ex, n=n, status="ok")
                        except Exception as e:
                            r = dict(
                                ex, n=n, method=method, status="error", error=repr(e)
                            )
                    log.save(key, r)
        # Ablation uses the same deterministic span and the same instrumented full fit.
        for nd in (1.25, 1.5, 2.0):
            n = round(nd * P.d)
            S = greedy_cancel(P, vol(P.X, P.d), n)
            r = P.evaluate(S, np.ones(n))
            log.save(
                f"{dataid}:{nd}:ablation",
                dict(
                    r,
                    dataset=nm,
                    data_id=dataid,
                    budget_factor=nd,
                    n=n,
                    d=P.d,
                    N=P.N,
                    method="greedy-cancel-unit",
                    protocol="ablation",
                    replicate=0,
                    status="ok",
                ),
            )
        print("controlled complete", nm, flush=True)


def rff_suite():
    log = Log("rff")
    for M, reps in ((50, 10), (100, 10), (200, 5)):
        for split in range(reps):
            X, y, Xe, ye, meta = holdout(
                "california", split, 12000 if M == 200 else 6000, M
            )
            dataid = f"rff_{M}_{split}"
            P = log.problem(dataid, X, y, meta, Xe, ye)
            for nd in (1.0, 1.25, 1.5, 2.0, 3.0):
                n = round(nd * P.d)
                for method in METHODS:
                    if method == "vol-sampling":
                        continue
                    r = log.run(
                        f"{dataid}:{nd}:{method}",
                        P,
                        n,
                        method,
                        seed_for("rff", M, split, nd, method),
                        dict(
                            dataset="california",
                            data_id=dataid,
                            M=M,
                            split=split,
                            replicate=split,
                            replicate_unit="fixed split and one method draw; fixed feature seed 7",
                            budget_factor=nd,
                            protocol="native",
                            full_test_mse=P.full_test_mse,
                        ),
                    )
            print("RFF complete", M, split, flush=True)


def get_stress(family, d, N, n, seed):
    if family in ("blocks", "perturbed_blocks"):
        k = n - d
        gs = []
        for g in range(k + 1, d + 1):
            q, r = divmod(d, g)
            gs.append((1 + (g - k) / (r / (q + 1) + (g - r) / q), g))
        g = max(gs)[1]
        q, r = divmod(d, g)
        dims = [q + 1] * r + [q] * (g - r)
        X, y, meta = blocks(dims, seed)
        res = X @ np.linalg.lstsq(X, y, rcond=1e-12)[0] - y
        w = np.linalg.lstsq(X, y, rcond=1e-12)[0]
        rng = np.random.default_rng(seed)
        while len(y) < N:
            j = int(rng.integers(len(y)))
            x = X[j].copy() / np.sqrt(2)
            rr = res[j] / np.sqrt(2)
            X[j] = x
            res[j] = rr
            X = np.vstack([X, x])
            res = np.r_[res, rr]
            y = X @ w - res
        if family == "perturbed_blocks":
            X += rng.normal(scale=1e-3, size=X.shape)
            X, _ = whiten_svd(X)
            res -= X @ (X.T @ res)
            y = X @ w - res
        meta.update(
            family=family,
            split_duplicate_rows_to_N=N,
            perturbation=1e-3 if family == "perturbed_blocks" else 0,
        )
        return X, y, meta
    return synthetic(
        family, d, N, seed, kappa=1e6 if family == "ill_conditioned" else 1.0
    )


def enumerated_search(
    log, P, n, basekey, seed, max_supports=None, full_enumeration=False
):
    """Enumerate discrete supports and optimize their weights with multiple local starts."""
    cfg = Config(restarts=8, maxiter=800)
    rng = np.random.default_rng(seed)
    best = float("inf")
    bestrow = None
    seen = 0
    pruned = 0
    all_count = sum(math.comb(P.N, m) for m in range(1, n + 1))
    t = time.perf_counter()
    for m in range(n, 0, -1):
        for S0 in itertools.combinations(range(P.N), m):
            if max_supports is not None and seen >= max_supports:
                break
            S = np.array(S0)
            seen += 1
            key = f"{basekey}:support:{seen}"
            if key in log.done:
                rr = log.done[key]
                if rr.get("ratio", float("inf")) < best:
                    best = rr["ratio"]
                    bestrow = rr
                continue
            # Relaxed minimum over the selected row space supplies a numerical pruning bound.
            _, sv, Vt = np.linalg.svd(P.X[S], full_matrices=False)
            rk = int(np.sum(sv > 1e-12 * sv[0]))
            B = Vt[:rk].T
            z = np.linalg.lstsq(P.X @ B, P.y, rcond=1e-12)[0]
            lb = float(np.linalg.norm(P.X @ B @ z - P.y) ** 2 / P.L)
            if lb > best + 1e-7:
                pruned += 1
                log.save(
                    key,
                    dict(
                        record_type="support",
                        data_id=basekey,
                        n=n,
                        indices=S.tolist(),
                        relaxed_lower_bound=lb,
                        status="projection_pruned",
                        rank=rk,
                    ),
                )
                continue
            found, atts, lps = cancellation(P, S, cfg)
            if found is not None:
                SS, a = found
                restarts = []
                branch = "lp"
            else:
                SS = S
                a, restarts, _, branch = fallback(P, S, cfg, rng)
            rr = P.evaluate(SS, a, cfg)
            rr.update(
                record_type="support",
                data_id=basekey,
                n=n,
                enumeration_support=S.tolist(),
                relaxed_lower_bound=lb,
                lp_attempts=atts,
                restarts=restarts,
                branch=branch,
                status="ok",
                config=asdict(cfg),
            )
            log.save(key, rr)
            if rr["ratio"] < best:
                best = rr["ratio"]
                bestrow = rr
            if rr["numerical_certificate"] and not full_enumeration:
                break
        if (max_supports is not None and seen >= max_supports) or (
            bestrow and bestrow.get("numerical_certificate") and not full_enumeration
        ):
            break
    out = dict(
        record_type="enumeration_summary",
        data_id=basekey,
        n=n,
        N=P.N,
        d=P.d,
        total_qualifying_supports=all_count,
        supports_visited=seen,
        all_supports_visited=seen == all_count,
        projection_pruned=pruned,
        best_ratio=best,
        best_key=None if bestrow is None else bestrow["key"],
        best_indices=None if bestrow is None else bestrow["indices"],
        best_weights=None if bestrow is None else bestrow["weights"],
        gap_to_conjectured_bound=frontier(P.d, n) - best,
        numerical_certificate=(
            False if bestrow is None else bestrow["numerical_certificate"]
        ),
        seconds=time.perf_counter() - t,
        stopping_rule=(
            "complete enumeration"
            if full_enumeration
            else f"numerical recovery or {max_supports} supports"
        ),
        continuous_search_globally_certified=False,
    )
    return log.save(f"{basekey}:enumeration_summary", out)


def stress_suite():
    log = Log("stress")
    for d, ns, sizes in [
        (3, [4], [6, 10, 18]),
        (4, [5], [8, 12, 24]),
        (6, [7, 8], [10, 18, 36]),
    ]:
        for n in ns:
            for N in sizes:
                for family in (
                    "random",
                    "correlated",
                    "ill_conditioned",
                    "structured_residual",
                    "overlap",
                    "blocks",
                    "perturbed_blocks",
                ):
                    for rep in range(3):
                        key = f"stress_{family}_{d}_{N}_{n}_{rep}"
                        X, y, meta = get_stress(family, d, N, n, seed_for(key))
                        P = log.problem(key, X, y, meta)
                        ex = dict(
                            dataset=family,
                            data_id=key,
                            replicate=rep,
                            N_over_n_plus_1=P.N / (n + 1),
                            conjectured_bound=frontier(d, n),
                            protocol="native",
                        )
                        r = log.run(
                            key + ":SpanCancel",
                            P,
                            n,
                            "SpanCancel",
                            seed_for(key, "method"),
                            ex,
                        )
                        if rep == 0 and N in (6, 8):
                            if key + ":enumeration_summary" not in log.done:
                                enumerated_search(
                                    log,
                                    P,
                                    n,
                                    key,
                                    seed_for(key, "enum"),
                                    full_enumeration=True,
                                )
                        elif r.get("ratio", 0) > frontier(d, n) + 1e-6:
                            if key + ":enumeration_summary" not in log.done:
                                enumerated_search(
                                    log,
                                    P,
                                    n,
                                    key,
                                    seed_for(key, "enum"),
                                    max_supports=120,
                                )
            print("stress completed d,n", d, n, flush=True)
    # Full frontier on representative structured and random instances, methods matched for every n.
    for d in (3, 4, 6):
        for family in ("random", "blocks", "perturbed_blocks", "overlap"):
            key = f"frontier_{family}_{d}"
            X, y, meta = get_stress(family, d, 2 * d + 2, d + 1, 20260 + d)
            P = log.problem(key, X, y, meta)
            for n in range(d, 2 * d + 1):
                rows = []
                for method in METHODS:
                    r = log.run(
                        f"{key}:{n}:{method}",
                        P,
                        n,
                        method,
                        seed_for(key, n, method),
                        dict(
                            dataset=family,
                            data_id=key,
                            record_type="frontier",
                            replicate=0,
                            protocol="native",
                            conjectured_bound=frontier(d, n),
                            solved_regime=n >= math.ceil(1.5 * d),
                        ),
                    )
                    rows.append(r)
                # Best-found pool combines native, reweighted baseline supports, and a stronger SpanCancel fallback.
                for base in rows:
                    if base.get("status") != "ok" or base["method"] == "SpanCancel":
                        continue
                    kk = f'{key}:{n}:{base["method"]}:common'
                    if kk not in log.done:
                        r = reweight(
                            P, base, Config(restarts=8, maxiter=800), seed_for(kk)
                        )
                        r.update(
                            dataset=family,
                            data_id=key,
                            n=n,
                            record_type="frontier",
                            protocol="common",
                            status="ok",
                            conjectured_bound=frontier(d, n),
                        )
                        log.save(kk, r)
            print("frontier completed", family, d, flush=True)


def diagnostics_suite():
    log = Log("diagnostics")
    cases = []
    for family, d, N, n in [
        ("random", 6, 240, 8),
        ("correlated", 6, 240, 8),
        ("structured_residual", 6, 240, 8),
        ("overlap", 6, 90, 8),
        ("blocks", 4, 20, 5),
        ("perturbed_blocks", 4, 20, 5),
        ("ill_conditioned", 6, 120, 8),
    ]:
        X, y, meta = get_stress(family, d, N, n, 2026)
        cases.append((family, X, y, meta, n))
    settings = [("default", Config())]
    for field, values in [
        ("pool_factor", [1, 2, 4, 16, 32]),
        ("rank_atol", [1e-4, 1e-6, 1e-10, 1e-12]),
        ("solver_rcond", [1e-8, 1e-10, 1e-14]),
        ("lp_tol", [1e-7, 1e-10]),
        ("support_tol", [1e-7, 1e-11]),
        ("coefficient_tol", [1e-5, 1e-9]),
        ("restarts", [0, 1, 3, 12]),
        ("initialization", ["lognormal"]),
    ]:
        for v in values:
            settings.append(
                (f"{field}={v}", dataclasses.replace(Config(), **{field: v}))
            )
    for name, X, y, meta, n in cases:
        for label, cfg in settings:
            key = f"diag_{name}"
            P = log.problem(key, X, y, meta, cfg=cfg)
            for rep in range(3):
                log.run(
                    f"{key}:{label}:{rep}",
                    P,
                    n,
                    "SpanCancel",
                    seed_for(key, rep),
                    dict(
                        dataset=name,
                        data_id=key,
                        setting=label,
                        replicate=rep,
                        protocol="sensitivity",
                    ),
                    cfg,
                )
        P = Problem(X, y)
        base = select(P, n, seed=2026)
        if base["branch"] == "lp":
            for rep in range(5):
                log.run(
                    f"diag_{name}:forced:{rep}",
                    P,
                    n,
                    "SpanCancel",
                    seed_for(name, "forced", rep),
                    dict(
                        dataset=name,
                        data_id=f"diag_{name}",
                        setting="forced_fallback_on_LP_solved_instance",
                        lp_comparator_ratio=base["ratio"],
                        replicate=rep,
                        protocol="forced_fallback",
                    ),
                    force=True,
                )
        print("diagnostics completed", name, flush=True)
    # Controlled conditioning sweep, same X geometry and residual family, nominal d=6.
    for kappa in (1.0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12):
        X, y, meta = synthetic("ill_conditioned", 6, 120, 812, kappa)
        key = f"conditioning_{kappa:g}"
        for rt in (1e-4, 1e-8, 1e-12):
            cfg = Config(rank_atol=rt)
            P = log.problem(key, X, y, meta, cfg=cfg)
            log.run(
                f"{key}:{rt}",
                P,
                8,
                "SpanCancel",
                42,
                dict(
                    dataset="conditioning",
                    data_id=key,
                    setting=f"rank_atol={rt}",
                    kappa_requested=kappa,
                    replicate=0,
                    protocol="conditioning",
                ),
                cfg,
            )
    # Common reweighting on the larger Fourier-feature support sizes.
    X, y, Xe, ye, meta = holdout("california", 0, 6000, 50)
    key = "large_common_rff50"
    P = log.problem(key, X, y, meta, Xe, ye)
    for nd in (1.25, 1.5, 2.0):
        n = round(nd * P.d)
        base = select(P, n, "D-optimal", seed=41)
        kk = f"{key}:{nd}:common"
        if kk not in log.done:
            r = reweight(P, base, Config(), 42)
            r.update(
                dataset="california",
                data_id=key,
                n=n,
                budget_factor=nd,
                replicate=0,
                protocol="large_common",
                status="ok",
            )
            log.save(kk, r)


def adversarial_suite():
    log = Log("adversarial")
    rng = np.random.default_rng(77221)
    for d, n, N in ((4, 5, 8), (6, 7, 10)):
        X0, y0, meta = get_stress("blocks", d, N, n, 718)
        P0 = Problem(X0, y0)
        r0 = P0.r.copy()
        incumbent = (X0, y0)
        score = -np.inf
        for trial in range(24):
            level = float(10 ** rng.uniform(-8, -0.3))
            Xbase, ybase = incumbent if trial % 3 == 0 else (X0, y0)
            X = Xbase + rng.normal(scale=level, size=Xbase.shape)
            X, _ = whiten_svd(X)
            r = r0 + rng.normal(scale=level, size=N)
            r -= X @ (X.T @ r)
            r /= np.linalg.norm(r)
            w = P0.w
            y = X @ w - r
            key = f"adversarial_{d}_{n}_{N}_{trial}"
            P = log.problem(
                key,
                X,
                y,
                dict(
                    family="adaptive block perturbation",
                    trial=trial,
                    noise=level,
                    outer_seed=77221,
                    parent="incumbent" if trial % 3 == 0 else "initial",
                    objective="maximize best feasible sampled-support ratio; heuristic adversary",
                ),
            )
            row = log.run(
                key + ":sc",
                P,
                n,
                "SpanCancel",
                seed_for(key),
                dict(
                    dataset="adaptive_perturbation",
                    data_id=key,
                    replicate=trial,
                    protocol="adversarial",
                    conjectured_bound=frontier(d, n),
                ),
            )
            vals = [row.get("ratio", np.inf)]
            for j in range(8):
                S = rng.choice(N, n, replace=False)
                cfg = Config(restarts=2, maxiter=300)
                a, logs, tt, branch = fallback(P, S, cfg, rng)
                rr = P.evaluate(S, a)
                rr.update(
                    data_id=key,
                    dataset="adaptive_perturbation",
                    n=n,
                    d=d,
                    N=N,
                    method="random-support-strong-weights",
                    protocol="adversarial",
                    config=asdict(cfg),
                    restarts=logs,
                    status="ok",
                )
                log.save(f"{key}:sample:{j}", rr)
                vals.append(rr["ratio"])
            best = min(vals)
            if best > score:
                score = best
                incumbent = (X, y)
            log.save(
                key + ":summary",
                dict(
                    record_type="adversarial_summary",
                    data_id=key,
                    d=d,
                    n=n,
                    N=N,
                    trial=trial,
                    best_ratio=best,
                    conjectured_bound=frontier(d, n),
                    gap_to_conjectured_bound=frontier(d, n) - best,
                    incumbent_score=score,
                ),
            )
            if (
                best > frontier(d, n) + 1e-6
                and key + ":enumeration_summary" not in log.done
            ):
                enumerated_search(
                    log, P, n, key, seed_for(key, "strong"), max_supports=150
                )
        print("adversarial completed", d, n, N, flush=True)


def runtime_suite():
    log = Log("runtime")
    # Warm-up uses the same numerical libraries and is recorded, excluded from summaries.
    X, y, meta = synthetic("random", 10, 100, 700)
    P = Problem(X, y)
    select(P, 15, seed=7)
    log.save(
        "warmup",
        dict(
            record_type="warmup",
            description="one N=100,d=10,n=15 full fit, LP selection and validation",
            status="ok",
        ),
    )
    for N in (200, 1000, 5000):
        for d in (10, 30, 60):
            for nd in (1.25, 1.5, 2.0):
                for rep in range(3):
                    key = f"runtime_{N}_{d}_{nd}_{rep}"
                    X, y, meta = synthetic("random", d, N, 871)
                    P = log.problem(f"runtime_data_{N}_{d}", X, y, meta)
                    n = round(nd * d)
                    log.run(
                        key,
                        P,
                        n,
                        "SpanCancel",
                        seed_for(key),
                        dict(
                            dataset="random",
                            data_id=f"runtime_data_{N}_{d}",
                            replicate=rep,
                            budget_factor=nd,
                            protocol="runtime",
                            timing_context="single experiment process; BLAS threads=1",
                        ),
                    )
        print("runtime scaling complete", N, flush=True)
    for d in (4, 10, 20):
        X, y, meta = blocks([d // 2, d - d // 2], 0)
        P = Problem(X, y)
        n = d + 1
        for rep in range(5):
            P = log.problem(f"runtime_fallback_{d}", X, y, meta)
            log.run(
                f"runtime_fallback_{d}_{rep}",
                P,
                n,
                "SpanCancel",
                seed_for(d, rep),
                dict(
                    dataset="two_blocks",
                    data_id=f"runtime_fallback_{d}",
                    replicate=rep,
                    protocol="runtime_fallback",
                    timing_context="single experiment process; BLAS threads=1",
                ),
            )
    for N, d in ((80, 6), (200, 10), (500, 20)):
        X, y, meta = synthetic("random", d, N, 717)
        for rep in range(3):
            P = log.problem(f"runtime_volume_{N}_{d}", X, y, meta)
            log.run(
                f"runtime_volume_{N}_{d}_{rep}",
                P,
                round(1.5 * d),
                "vol-sampling",
                seed_for(N, d, rep),
                dict(
                    dataset="random",
                    data_id=f"runtime_volume_{N}_{d}",
                    replicate=rep,
                    protocol="runtime_volume",
                    timing_context="single experiment process; BLAS threads=1",
                ),
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "suite",
        choices=[
            "controlled",
            "rff",
            "stress",
            "diagnostics",
            "adversarial",
            "runtime",
        ],
    )
    args = p.parse_args()
    with threadpool_limits(limits=1):
        globals()[args.suite + "_suite"]()
