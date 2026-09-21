from experiments.paths import RESULTS, OUTPUT
from pathlib import Path
import sys, json, time, dataclasses, hashlib

sys.dont_write_bytecode = True
import numpy as np
from decimal import Decimal, localcontext
from .run_experiments import Log, enumerated_search
from spancancel.engine import Config, Problem, seed_for, select
from spancancel.datasets_local import load_input, frontier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def correlated(d, N, seed):
    rng = np.random.default_rng(seed)
    X = 0.95 * rng.normal(size=(N, 1)) + np.sqrt(1 - 0.95**2) * rng.normal(size=(N, d))
    X /= np.sqrt(np.sum(X * X, axis=0))
    b = rng.normal(size=N)
    r = b - X @ np.linalg.lstsq(X, b, rcond=1e-12)[0]
    r /= np.linalg.norm(r)
    w = rng.normal(size=d) * 20
    cc = np.corrcoef(X, rowvar=False)
    return (
        X,
        X @ w - r,
        dict(
            family="correlated_preserved",
            pairwise_column_correlation_mean=float(cc[np.triu_indices(d, 1)].mean()),
            whitening=False,
            scale="column unit Euclidean norm; correlation retained",
            seed=seed,
        ),
    )


def correlation_runs():
    log = Log("correlation_correction")
    for d, ns, sizes in [
        (3, [4], [6, 10, 18]),
        (4, [5], [8, 12, 24]),
        (6, [7, 8], [10, 18, 36]),
    ]:
        for n in ns:
            for N in sizes:
                for rep in range(3):
                    key = f"correlated_preserved_{d}_{N}_{n}_{rep}"
                    X, y, m = correlated(
                        d, N, seed_for(f"stress_correlated_{d}_{N}_{n}_{rep}")
                    )
                    P = log.problem(key, X, y, m)
                    rr = log.run(
                        key + ":sc",
                        P,
                        n,
                        "SpanCancel",
                        seed_for(key),
                        dict(
                            dataset="correlated_preserved",
                            data_id=key,
                            replicate=rep,
                            protocol="native",
                            conjectured_bound=frontier(d, n),
                            N_over_n_plus_1=N / (n + 1),
                            analysis_role="Unwhitened correlated-input stress and sensitivity experiments",
                        ),
                    )
                    if rep == 0 and N in (6, 8):
                        enumerated_search(
                            log, P, n, key, seed_for(key, "enum"), full_enumeration=True
                        )
    X, y, m = correlated(6, 240, 2026)
    key = "diag_correlated_preserved"
    settings = [("default", Config())]
    for field, vals in [
        ("pool_factor", [1, 2, 4, 16, 32]),
        ("rank_atol", [1e-4, 1e-6, 1e-10, 1e-12]),
        ("solver_rcond", [1e-8, 1e-10, 1e-14]),
        ("lp_tol", [1e-7, 1e-10]),
        ("support_tol", [1e-7, 1e-11]),
        ("coefficient_tol", [1e-5, 1e-9]),
        ("restarts", [0, 1, 3, 12]),
        ("initialization", ["lognormal"]),
    ]:
        for v in vals:
            settings.append(
                (f"{field}={v}", dataclasses.replace(Config(), **{field: v}))
            )
    for setting, cfg in settings:
        P = log.problem(key, X, y, m, cfg=cfg)
        for rep in range(3):
            log.run(
                f"{key}:{setting}:{rep}",
                P,
                8,
                "SpanCancel",
                seed_for(key, rep),
                dict(
                    dataset="correlated_preserved",
                    data_id=key,
                    setting=setting,
                    replicate=rep,
                    protocol="sensitivity",
                ),
                cfg,
            )
    P = Problem(X, y)
    base = select(P, 8)
    if base["branch"] == "lp":
        for rep in range(5):
            log.run(
                f"{key}:forced:{rep}",
                P,
                8,
                "SpanCancel",
                seed_for(key, "forced", rep),
                dict(
                    dataset="correlated_preserved",
                    data_id=key,
                    setting="forced_fallback_on_LP_solved_instance",
                    lp_comparator_ratio=base["ratio"],
                    replicate=rep,
                    protocol="forced_fallback",
                ),
                force=True,
            )
    print("Correlated-input experiments completed", flush=True)


def solve_decimal(A, b):
    n = len(b)
    T = [list(a) + [bb] for a, bb in zip(A, b)]
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(T[i][j]))
        T[j], T[pivot] = T[pivot], T[j]
        if T[j][j] == 0:
            raise ValueError("exact zero pivot in Decimal solve")
        for i in range(j + 1, n):
            z = T[i][j] / T[j][j]
            for k in range(j, n + 1):
                T[i][k] -= z * T[j][k]
    out = [Decimal(0)] * n
    for i in range(n - 1, -1, -1):
        out[i] = (T[i][-1] - sum(T[i][j] * out[j] for j in range(i + 1, n))) / T[i][i]
    return out


def decfit(X, y, weights):
    d = len(X[0])
    A = [
        [sum(weights[k] * X[k][i] * X[k][j] for k in range(len(y))) for j in range(d)]
        for i in range(d)
    ]
    b = [sum(weights[k] * X[k][i] * y[k] for k in range(len(y))) for i in range(d)]
    return solve_decimal(A, b)


def loss(X, y, w):
    return sum((sum(a * b for a, b in zip(x, w)) - yy) ** 2 for x, yy in zip(X, y))


def decimal_checks():
    log = Log("high_precision")
    rows = [
        json.loads(x) for x in (RESULTS / "diagnostics.jsonl").read_text().splitlines()
    ]
    for r in rows:
        if r.get("protocol") != "conditioning":
            continue
        z = load_input(r["data_id"])
        X = z["X"]
        y = z["y"]
        S = r["indices"]
        a = r["weights"]
        with localcontext() as ctx:
            ctx.prec = 80
            XX = [[Decimal.from_float(float(x)) for x in row] for row in X]
            yy = [Decimal.from_float(float(v)) for v in y]
            aa = [Decimal.from_float(float(v)) for v in a]
            try:
                w = decfit(XX, yy, [Decimal(1)] * len(yy))
                wf = decfit([XX[i] for i in S], [yy[i] for i in S], aa)
                L = loss(XX, yy, w)
                LF = loss(XX, yy, wf)
                err = sum((a - b) ** 2 for a, b in zip(w, wf)).sqrt()
                rel = err / (1 + sum(a * a for a in w).sqrt())
                log.save(
                    r["key"],
                    dict(
                        data_id=r["data_id"],
                        parent_key=r["key"],
                        kappa_requested=r["kappa_requested"],
                        rank_atol=r["config"]["rank_atol"],
                        double_ratio=r["ratio"],
                        decimal_ratio=str(LF / L),
                        decimal_coefficient_error_relative=str(rel),
                        double_coefficient_error_relative=r[
                            "coefficient_error_relative"
                        ],
                        decimal_full_loss=str(L),
                        status="ok",
                        decimal_precision=80,
                        qualification="80-digit evaluation of the untruncated full-rank solution on stored binary64 inputs and weights",
                    ),
                )
            except Exception as e:
                log.save(
                    r["key"], dict(parent_key=r["key"], status="error", error=str(e))
                )
    print("High precision checks completed", flush=True)


if __name__ == "__main__":
    protocol = dict(
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        reason="Evaluate unwhitened correlated inputs on the matched experiment grid and recompute the conditioning cases using 80-digit decimal arithmetic.",
        correlation="same dimensions, budgets, reps as initial family; normalized correlated columns retained",
        high_precision="all 21 conditioning settings; evaluate stored supports and weights without reoptimization",
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (OUTPUT / "conditioning_protocol.json").write_text(json.dumps(protocol, indent=2))
    with threadpool_limits(1):
        correlation_runs()
        decimal_checks()
