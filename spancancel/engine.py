"""SpanCancel selection with numerical recovery checks and per-stage diagnostics."""

from dataclasses import dataclass, asdict
import time, hashlib, json
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.linalg import qr


@dataclass(frozen=True)
class Config:
    """Candidate-pool size, numerical tolerances and fallback search settings."""

    pool_factor: float = 8.0
    rank_atol: float = 1e-8
    solver_rcond: float = 1e-12
    lp_tol: float = 1e-9
    support_tol: float = 1e-9
    coefficient_tol: float = 1e-7
    restarts: int = 6
    maxiter: int = 400
    fatol: float = 1e-12
    xatol: float = 1e-8
    initialization: str = "dirichlet"


def seed_for(*parts):
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little"
    )


def rank(X, tol=1e-8):
    return int(np.sum(np.linalg.svd(X, compute_uv=False) > tol))


def fit(X, y, S, a, cfg=Config()):
    """Minimum-norm weighted fit on row indices S with nonnegative weights a."""
    S = np.asarray(S, int)
    a = np.asarray(a, float)
    if not len(S) or not np.isfinite(a).all() or np.any(a < 0) or a.sum() <= 0:
        raise ValueError("Invalid selected support/weights")
    z = np.sqrt(a / a.max())
    return np.linalg.lstsq(X[S] * z[:, None], y[S] * z, rcond=cfg.solver_rcond)[0]


class Problem:
    """Full-data fit for finite X of shape (N, d) and scalar targets y of shape (N,).

    Add an intercept column explicitly. Optional test arrays use the same features.
    """

    def __init__(self, X, y, cfg=Config(), Xtest=None, ytest=None):
        t = time.perf_counter()
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        if self.X.ndim != 2 or not all(self.X.shape):
            raise ValueError("X must be a nonempty two-dimensional matrix.")
        if self.y.shape != (self.X.shape[0],):
            raise ValueError("y must be one-dimensional with one target per row of X.")
        if not np.isfinite(self.X).all() or not np.isfinite(self.y).all():
            raise ValueError("X and y must contain only finite values.")
        if (Xtest is None) != (ytest is None):
            raise ValueError("Provide both Xtest and ytest, or neither.")
        if Xtest is not None:
            Xtest, ytest = np.asarray(Xtest, float), np.asarray(ytest, float)
            if (
                Xtest.ndim != 2
                or Xtest.shape[1] != self.X.shape[1]
                or ytest.shape != (len(Xtest),)
            ):
                raise ValueError(
                    "Test arrays must match the training feature and target dimensions."
                )
            if not np.isfinite(Xtest).all() or not np.isfinite(ytest).all():
                raise ValueError("Test arrays must contain only finite values.")
        self.cfg = cfg
        self.N, self.d = self.X.shape
        self.w = np.linalg.lstsq(self.X, self.y, rcond=cfg.solver_rcond)[0]
        self.full_fit_s = time.perf_counter() - t
        t = time.perf_counter()
        self.r = self.X @ self.w - self.y
        self.L = float(self.r @ self.r)
        self.C = self.r[:, None] * self.X
        self.residual_s = time.perf_counter() - t
        self.singular = np.linalg.svd(self.X, compute_uv=False)
        self.condition = (
            float(self.singular[0] / self.singular[-1])
            if self.singular[-1] > 0
            else float("inf")
        )
        self.rank_solver = int(
            np.sum(self.singular > cfg.solver_rcond * self.singular[0])
        )
        self.Xtest = Xtest
        self.ytest = ytest
        self.full_test_mse = (
            None if Xtest is None else float(np.mean((Xtest @ self.w - ytest) ** 2))
        )
        self.Q = None

    def evaluate(self, S, a, cfg=None):
        cfg = cfg or self.cfg
        t = time.perf_counter()
        S = np.asarray(S, int)
        a = np.asarray(a, float)
        a = a / a.sum()
        wf = fit(self.X, self.y, S, a, cfg)
        err = wf - self.w
        residual = self.X @ wf - self.y
        loss = float(residual @ residual)
        ratio = loss / self.L if self.L > 0 else (1.0 if loss == 0 else float("inf"))
        weighted_s = np.linalg.svd(self.X[S] * np.sqrt(a)[:, None], compute_uv=False)
        pos = a > cfg.support_tol
        rank_support = rank(self.X[S[pos]], cfg.rank_atol) if np.any(pos) else 0
        g = self.C[S].T @ a
        g_scale = max(1.0, float(np.linalg.norm(self.C[S], ord=2)))
        coefficient_error = float(np.linalg.norm(err))
        normalized = coefficient_error / (1 + np.linalg.norm(self.w))
        out = dict(
            ratio=ratio,
            loss=loss,
            full_loss=self.L,
            coefficient_error=coefficient_error,
            coefficient_error_relative=float(normalized),
            residual_cancellation_error=float(np.linalg.norm(g)),
            residual_cancellation_scaled=float(np.linalg.norm(g) / g_scale),
            support_size=int(pos.sum()),
            selected_count=len(S),
            support_rank=rank_support,
            weighted_solver_rank=int(
                np.sum(weighted_s > cfg.solver_rcond * weighted_s[0])
            ),
            weighted_condition=(
                float(weighted_s[0] / weighted_s[-1])
                if weighted_s[-1] > 0
                else float("inf")
            ),
            near_one=bool(abs(ratio - 1) <= 1e-7),
            coefficient_recovery=bool(normalized <= cfg.coefficient_tol),
            numerical_certificate=bool(
                rank_support == self.d
                and np.linalg.norm(g) <= cfg.lp_tol * g_scale
                and normalized <= cfg.coefficient_tol
            ),
            min_weight=float(a.min()),
            max_weight=float(a.max()),
            indices=S.tolist(),
            weights=a.tolist(),
            full_stationarity=float(np.linalg.norm(self.X.T @ self.r)),
            loss_identity_error=float(
                abs((loss - self.L) - np.linalg.norm(self.X @ err) ** 2)
            ),
        )
        if self.Xtest is not None:
            out["test_mse"] = float(np.mean((self.Xtest @ wf - self.ytest) ** 2))
            out["test_ratio"] = (
                out["test_mse"] / self.full_test_mse if self.full_test_mse > 0 else None
            )
            out["test_prediction_error"] = float(np.linalg.norm(self.Xtest @ err))
        out["validation_s"] = time.perf_counter() - t
        return out


def vol(X, n):
    """Same regularized greedy gain rule as baseline, with symmetric updates."""
    N, d = X.shape
    A = np.eye(d) * 1e9
    S = []
    taken = np.zeros(N, bool)
    for _ in range(n):
        g = np.einsum("ij,ij->i", X @ A, X)
        g[taken] = -np.inf
        i = int(np.argmax(g))
        S.append(i)
        taken[i] = True
        v = X[i]
        Av = A @ v
        den = 1 + v @ Av
        if den <= 0 or not np.isfinite(den):
            A = np.linalg.pinv(X[S].T @ X[S] + np.eye(d) * 1e-9, rcond=1e-15)
        else:
            A -= np.outer(Av, Av) / den
            A = (A + A.T) * 0.5
    return np.asarray(S, int)


def cancellation(P, S, cfg):
    """Recheck both support rank and cancellation after thresholding."""
    t = time.perf_counter()
    S = np.asarray(S, int)
    C = P.C[S]
    m = len(S)
    attempts = []
    A = np.vstack([C.T, np.ones(m)])
    b = np.r_[np.zeros(P.d), 1.0]
    for oi, obj in enumerate((np.zeros(m), -np.linalg.norm(C, axis=1))):
        t0 = time.perf_counter()
        res = linprog(
            obj,
            A_eq=A,
            b_eq=b,
            bounds=(0, None),
            method="highs",
            options={
                "primal_feasibility_tolerance": max(1e-10, cfg.lp_tol),
                "dual_feasibility_tolerance": max(1e-10, cfg.lp_tol),
            },
        )
        info = dict(
            objective=oi,
            status=int(res.status),
            success=bool(res.success),
            message=res.message,
            iterations=int(res.nit),
            seconds=time.perf_counter() - t0,
        )
        attempts.append(info)
        if not res.success or not np.isfinite(res.x).all():
            continue
        info["min_raw_weight"] = float(res.x.min())
        if res.x.min() < -1e-10:
            continue
        a = np.maximum(res.x, 0)
        a /= a.sum()
        idx = a > cfg.support_tol
        if not idx.any():
            continue
        SS = S[idx]
        aa = a[idx]
        aa /= aa.sum()
        ev = P.evaluate(SS, aa, cfg)
        info.update(
            {
                k: ev[k]
                for k in [
                    "support_size",
                    "support_rank",
                    "residual_cancellation_scaled",
                    "coefficient_error_relative",
                    "numerical_certificate",
                ]
            }
        )
        if ev["numerical_certificate"]:
            return (SS, aa), attempts, time.perf_counter() - t
    return None, attempts, time.perf_counter() - t


def fallback(P, S, cfg, rng):
    t = time.perf_counter()
    S = np.asarray(S, int)
    m = len(S)
    Xs = P.X[S]
    ys = P.y[S]
    # Evaluate full loss through stable QR sufficient statistics: L + ||R(w-w*)||^2.
    # Final results always recomputed using raw residuals, never clipped to >=1.
    R = np.linalg.qr(P.X, mode="r")

    def weights(v):
        a = np.exp(v - np.max(v))
        return a / a.sum()

    def f(v):
        try:
            wf = fit(P.X, P.y, S, weights(v), cfg)
            return 1 + float(np.linalg.norm(R @ (wf - P.w)) ** 2) / P.L
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return float("inf")

    a = np.ones(m) / m
    best = f(np.zeros(m))
    logs = [
        dict(
            start=-1,
            kind="uniform_evaluation",
            initial_weights=a.tolist(),
            ratio=best,
            success=True,
            nfev=1,
            nit=0,
            seconds=0.0,
        )
    ]
    if m == P.d and rank(Xs, cfg.rank_atol) == P.d:
        return a, logs, time.perf_counter() - t, "square_interpolation"
    starts = [np.zeros(m)]
    for _ in range(cfg.restarts):
        starts.append(
            rng.normal(0, 2, m)
            if cfg.initialization == "lognormal"
            else np.log(rng.dirichlet(np.ones(m)) + 1e-9)
        )
    for j, start in enumerate(starts):
        tt = time.perf_counter()
        rr = minimize(
            f,
            start,
            method="Nelder-Mead",
            options=dict(maxiter=cfg.maxiter, fatol=cfg.fatol, xatol=cfg.xatol),
        )
        aa = weights(rr.x)
        raw = P.evaluate(S, aa, cfg)["ratio"]
        logs.append(
            dict(
                start=j,
                kind="uniform_optimized" if j == 0 else cfg.initialization,
                initial_weights=weights(start).tolist(),
                result_weights=aa.tolist(),
                ratio=float(raw),
                objective=float(rr.fun),
                success=bool(rr.success),
                message=rr.message,
                nfev=int(rr.nfev),
                nit=int(rr.nit),
                seconds=time.perf_counter() - tt,
            )
        )
        if rr.fun < best:
            best = float(rr.fun)
            a = aa
    return a, logs, time.perf_counter() - t, "heuristic"


def greedy_cancel(P, S, n):
    S = list(map(int, S))
    g = P.C[S].sum(0)
    available = np.ones(P.N, bool)
    available[S] = False
    while len(S) < n:
        scores = np.linalg.norm(P.C + g, axis=1)
        scores[~available] = np.inf
        j = int(scores.argmin())
        S.append(j)
        available[j] = False
        g += P.C[j]
    return np.asarray(S, int)


def select(P, n, method="SpanCancel", cfg=None, seed=0, force_fallback=False):
    """Select at most n rows and return weights, loss and recovery diagnostics.

    Methods: SpanCancel, uniform, leverage, leverage-w, D-optimal, vol-sampling.
    n is an integer between d and N. seed controls stochastic selections.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer support budget.")
    cfg = cfg or P.cfg
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    if not (P.d <= n <= P.N):
        raise ValueError("Budget outside full-rank experiment range")
    attempts = []
    restarts = []
    lp_s = fallback_s = 0.0
    pool_size = 0
    repair = False
    branch = "native"
    S = None
    a = None
    t = time.perf_counter()
    if method == "SpanCancel":
        span = vol(P.X, P.d)
        if rank(P.X[span], cfg.rank_atol) < P.d and rank(P.X, cfg.rank_atol) == P.d:
            span = qr(P.X.T, pivoting=True, mode="economic")[2][: P.d]
            repair = True
        count = min(P.N, max(int(cfg.pool_factor * P.d), n + P.d))
        pool = np.array(
            list(
                dict.fromkeys(
                    span.tolist()
                    + np.argsort(-np.linalg.norm(P.C, axis=1), kind="stable")[
                        :count
                    ].tolist()
                )
            )
        )
        pool_size = len(pool)
        candidate_s = time.perf_counter() - t
        if not force_fallback:
            found, attempts, lp_s = cancellation(P, pool, cfg)
            if found is not None and len(found[0]) <= n:
                S, a = found
                branch = "lp"
        if S is None:
            t = time.perf_counter()
            S = greedy_cancel(P, span, n)
            candidate_s += time.perf_counter() - t
            a, restarts, fallback_s, branch = fallback(P, S, cfg, rng)
    else:
        if method == "uniform":
            S = rng.choice(P.N, n, replace=False)
            a = np.ones(n)
        elif method in ("leverage", "leverage-w"):
            U, s, _ = np.linalg.svd(P.X, full_matrices=False)
            keep = s > cfg.solver_rcond * s[0]
            h = np.sum(U[:, keep] ** 2, axis=1)
            p = h / h.sum()
            S = rng.choice(P.N, n, replace=False, p=p)
            a = 1 / p[S] if method == "leverage-w" else np.ones(n)
        elif method == "D-optimal":
            S = vol(P.X, n)
            a = np.ones(n)
        elif method == "vol-sampling":
            S = np.arange(P.N)
            # Recompute leverage by SVD: avoids accumulated rank-one inverse drift.
            while len(S) > n:
                U, s, _ = np.linalg.svd(P.X[S], full_matrices=False)
                h = (U[:, s > cfg.solver_rcond * s[0]] ** 2).sum(1)
                pr = 1 - h
                if pr.min() < -1e-8:
                    raise ArithmeticError("Invalid deletion probability")
                pr = np.maximum(pr, 0)
                pr /= pr.sum()
                j = rng.choice(len(S), p=pr)
                S = np.delete(S, j)
            a = np.ones(n)
        else:
            raise ValueError(method)
        candidate_s = time.perf_counter() - t
    ev = P.evaluate(S, a, cfg)
    selection_s = time.perf_counter() - start
    ev.update(
        method=method,
        branch=branch,
        fallback_used=branch in ("heuristic", "square_interpolation"),
        lp_attempts=attempts,
        restarts=restarts,
        pool_size=pool_size,
        span_repair=repair,
        candidate_s=candidate_s,
        lp_s=lp_s,
        fallback_s=fallback_s,
        selection_s=selection_s,
        full_fit_s=P.full_fit_s,
        residual_s=P.residual_s,
        total_s=P.full_fit_s + P.residual_s + selection_s,
        N=P.N,
        d=P.d,
        condition_X=P.condition,
        rank_solver=P.rank_solver,
        rank_detected=rank(P.X, cfg.rank_atol),
        seed=int(seed),
        config=asdict(cfg),
    )
    return ev


def reweight(P, selection, cfg, seed):
    t = time.perf_counter()
    S = np.asarray(selection["indices"])
    found, attempts, lp_s = cancellation(P, S, cfg)
    restarts = []
    fb_s = 0.0
    if found is not None:
        S, a = found
        branch = "lp"
    else:
        a, restarts, fb_s, branch = fallback(P, S, cfg, np.random.default_rng(seed))
    ev = P.evaluate(S, a, cfg)
    ev.update(
        method=selection["method"],
        branch=branch,
        fallback_used=branch in ("heuristic", "square_interpolation"),
        lp_attempts=attempts,
        restarts=restarts,
        lp_s=lp_s,
        fallback_s=fb_s,
        reweight_s=time.perf_counter() - t,
        selection_s=selection["selection_s"],
        candidate_s=selection["candidate_s"],
        full_fit_s=P.full_fit_s,
        residual_s=P.residual_s,
        N=P.N,
        d=P.d,
        condition_X=P.condition,
        rank_solver=P.rank_solver,
        rank_detected=rank(P.X, cfg.rank_atol),
        seed=int(seed),
        config=asdict(cfg),
        original_support=selection["indices"],
    )
    ev["total_s"] = (
        P.full_fit_s + P.residual_s + selection["selection_s"] + ev["reweight_s"]
    )
    return ev
