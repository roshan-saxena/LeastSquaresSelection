"""SpanCancel and the baselines.

Each selector returns ``(indices, weights)`` and :func:`ratio` evaluates the
full-data loss ratio ``L_D(w_F) / L*``. SpanCancel first spans by greedy
max-volume selection, then looks for nonnegative weights that cancel the
full-fit residual cross moment. The LP often returns a Caratheodory support;
when that support does not span, the implementation minimizes the full loss
ratio on a greedily augmented spanning subset.
"""

import numpy as np
from scipy.optimize import minimize, linprog
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

DEFAULT_SEED = 0
RANK_TOL = 1e-8
PINV_RCOND = 1e-12
rng = np.random.default_rng(DEFAULT_SEED)


def reset_rng(seed=DEFAULT_SEED):
    """Reset selector randomness so every script is reproducible in isolation."""
    global rng
    rng = np.random.default_rng(seed)


def rank(X):
    """Numerical matrix rank using the paper's singular-value threshold."""
    s = np.linalg.svd(np.asarray(X), compute_uv=False)
    return int(np.count_nonzero(s > RANK_TOL))


def pinv(M):
    return np.linalg.pinv(M, rcond=PINV_RCOND)


def whiten(X):
    v, Q = np.linalg.eigh(X.T @ X)
    inv = np.zeros_like(v)
    keep = v > RANK_TOL**2
    inv[keep] = 1 / np.sqrt(v[keep])
    W = Q @ np.diag(inv) @ Q.T
    return X @ W, W


def vol(X, n):
    N, d = X.shape
    S = []
    A = np.eye(d) * 1e9
    taken = np.zeros(N, bool)
    for _ in range(n):
        g = ((X @ A) * X).sum(1)
        g[taken] = -np.inf
        i = int(np.argmax(g))
        S.append(i)
        taken[i] = True
        v = X[i]
        Av = A @ v
        A = A - np.outer(Av, Av) / (1 + v @ Av)
    return S


def fit(X, y, S, a):
    Xs = X[S]
    M = (Xs * a[:, None]).T @ Xs
    b = (Xs * (a * y[S])[:, None]).sum(0)
    return pinv(M) @ b


def stats(X, y):
    Mf = X.T @ X
    bf = X.T @ y
    w = pinv(Mf) @ bf
    r = X @ w - y
    return w, Mf, bf, float(y @ y), float(r @ r)


def ratio(X, y, S, a, Mf, bf, yy, L):
    w = fit(X, y, S, a)
    loss = float(w @ Mf @ w - 2 * bf @ w + yy)
    loss = max(loss, 0.0)
    if L <= 1e-12:
        return 1.0 if loss <= 1e-12 else np.inf
    return max(1.0, loss / L)


def cancellation_weights(C):
    """Return numerically valid nonnegative weights with ``C.T @ a == 0``."""
    m = len(C)
    A = np.vstack([C.T, np.ones(m)])
    b = np.r_[np.zeros(C.shape[1]), 1.0]
    # Keep the zero-objective feasibility LP used in the paper as the fast
    # path. If HiGHS returns a tolerance-level degenerate vertex, retry with a
    # deterministic objective favoring substantial cross-moment vectors.
    objectives = (np.zeros(m), -np.linalg.norm(C, axis=1))
    for objective in objectives:
        res = linprog(objective, A_eq=A, b_eq=b, bounds=[(0, None)] * m, method="highs")
        if res is None or not res.success or not np.isfinite(res.x).all():
            continue
        # HiGHS may report success with small negative entries inside its
        # default feasibility tolerance. Those entries can be essential to
        # the equalities, so clipping without validation is unsafe.
        if np.min(res.x) < -1e-10:
            continue
        a = np.maximum(res.x, 0.0)
        total = float(a.sum())
        if total <= 0:
            continue
        a /= total
        scale = max(1.0, float(np.linalg.norm(C, ord=2)))
        if np.linalg.norm(C.T @ a) <= 1e-9 * scale:
            return a
    return None


def tmse(w, X, y):
    return float(np.mean((X @ w - y) ** 2))


def unif(X, y, n, w):
    return list(rng.choice(len(y), n, replace=False)), np.ones(n)


def lev(X, y, n, w):
    h = np.einsum("ij,jk,ik->i", X, pinv(X.T @ X), X)
    p = h / h.sum()
    return list(rng.choice(len(y), n, replace=False, p=p)), np.ones(n)


def levw(X, y, n, w):
    h = np.einsum("ij,jk,ik->i", X, pinv(X.T @ X), X)
    p = h / h.sum()
    S = list(rng.choice(len(y), n, replace=False, p=p))
    return S, 1.0 / p[S]


def vsel(X, y, n, w):
    return vol(X, n), np.ones(n)


def vsamp(X, y, n, w):
    S = list(range(len(y)))
    A = pinv(X.T @ X)
    while len(S) > n:
        XS = X[S]
        h = np.einsum("ij,jk,ik->i", XS, A, XS)
        p = np.clip(1 - h, 1e-12, None)
        p /= p.sum()
        j = int(rng.choice(len(S), p=p))
        x = X[S[j]]
        Ax = A @ x
        A = A + np.outer(Ax, Ax) / max(1 - h[j], 1e-12)
        S.pop(j)
    return S, np.ones(n)


def cancel(X, r, span, n):
    S = list(span)
    g = sum(r[j] * X[j] for j in S)
    seen = set(S)
    while len(S) < n:
        c = [i for i in range(len(r)) if i not in seen]
        i = c[int(np.argmin([np.linalg.norm(g + r[j] * X[j]) for j in c]))]
        S.append(i)
        seen.add(i)
        g += r[i] * X[i]
    return S


LAST_SC_BRANCH = None


def sc(X, y, n, w):
    global LAST_SC_BRANCH
    LAST_SC_BRANCH = None
    N, d = X.shape
    r = X @ w - y
    C = r[:, None] * X
    span = vol(X, d)
    mag = np.linalg.norm(C, axis=1)
    pool = list(
        dict.fromkeys(
            span + list(map(int, np.argsort(mag)[::-1][: min(N, max(8 * d, n + d))]))
        )
    )

    def exact_from(P):
        P = np.asarray(P, dtype=int)
        ap = cancellation_weights(C[P])
        if ap is None:
            return None
        nz = np.nonzero(ap > 1e-9)[0]
        S = [int(P[i]) for i in nz]
        a = ap[nz]
        a /= a.sum()
        if not (0 < len(S) <= n and rank(X[S]) == d):
            return None
        wf = fit(X, y, S, a)
        if np.linalg.norm(wf - w) > 1e-7 * (1 + np.linalg.norm(w)):
            return None
        return S, a

    exact = exact_from(pool)
    if exact is not None:
        LAST_SC_BRANCH = "lp"
        return exact
    # With exactly d independent rows, the square interpolating fit does not
    # depend on the positive row weights, so nonlinear weight optimization is
    # mathematically redundant. Consume the six documented Dirichlet starts so
    # later stochastic methods retain the paper's exact random-number stream.
    if n == d:
        LAST_SC_BRANCH = "square"
        for _ in range(6):
            rng.dirichlet(np.ones(d))
        return span, np.ones(d) / d
    LAST_SC_BRANCH = "heuristic"
    S = cancel(X, r, span, n)
    Mf, bf, yy, L = X.T @ X, X.T @ y, float(y @ y), float(y @ y - (X.T @ y) @ w)
    Xs, ys, m = X[S], y[S], len(S)

    def f(t):
        a = np.exp(t - t.max())
        a /= a.sum()
        M = (Xs * a[:, None]).T @ Xs
        bb = (Xs * (a * ys)[:, None]).sum(0)
        wf = pinv(M) @ bb
        return float(wf @ Mf @ wf - 2 * bf @ wf + yy) / max(L, 1e-12)

    a = np.ones(m) / m
    best = f(np.zeros(m))
    for _ in range(6):
        rr = minimize(
            f,
            np.log(rng.dirichlet(np.ones(m)) + 1e-9),
            method="Nelder-Mead",
            options={"maxiter": 400, "fatol": 1e-12},
        )
        if rr.fun < best:
            best = rr.fun
            a = np.exp(rr.x - rr.x.max())
            a /= a.sum()
    return S, a


def wopt(X, y, S, w):
    S = list(S)
    d = X.shape[1]
    m = len(S)
    r = X[S] @ w - y[S]
    C = r[:, None] * X[S]
    if rank(X[S]) == d:
        a = cancellation_weights(C)
        if a is not None:
            wf = fit(X, y, S, a)
            if np.linalg.norm(wf - w) <= 1e-7 * (1 + np.linalg.norm(w)):
                return a
    if m > 60:
        return np.ones(m)
    Mf, bf, yy = X.T @ X, X.T @ y, float(y @ y)
    L = float(yy - bf @ w)
    Xs, ys = X[S], y[S]

    def f(t):
        a = np.exp(t - t.max())
        a /= a.sum()
        M = (Xs * a[:, None]).T @ Xs
        bb = (Xs * (a * ys)[:, None]).sum(0)
        wf = pinv(M) @ bb
        return float(wf @ Mf @ wf - 2 * bf @ wf + yy) / max(L, 1e-12)

    a = np.ones(m) / m
    best = f(np.zeros(m))
    for _ in range(6):
        rr = minimize(
            f,
            np.log(rng.dirichlet(np.ones(m)) + 1e-9),
            method="Nelder-Mead",
            options={"maxiter": 400, "fatol": 1e-12},
        )
        if rr.fun < best:
            best = rr.fun
            a = np.exp(rr.x - rr.x.max())
            a /= a.sum()
    return a


METHODS = {
    "uniform": unif,
    "leverage": lev,
    "leverage-w": levw,
    "vol-sampling": vsamp,
    "D-optimal": vsel,
    "SpanCancel": sc,
}
STOCH = {"uniform", "leverage", "leverage-w", "vol-sampling"}


def _sm(nm):
    t = (
        getattr(sm.datasets, nm)
        .load_pandas()
        .data.select_dtypes(include=[np.number])
        .dropna()
    )
    return t.iloc[:, 1:].to_numpy(float), t.iloc[:, 0].to_numpy(float)


def reals():
    """Controlled low-dimensional regression tasks used by the training tables."""
    out = []
    for nm, ld, k in [
        ("diabetes", datasets.load_diabetes, 5),
        ("wine", datasets.load_wine, 5),
        ("breast", datasets.load_breast_cancer, 6),
    ]:
        d = ld()
        out.append(
            (nm, np.asarray(d.data, float)[:, :k], np.asarray(d.target, float).ravel())
        )
    for nm in [
        "longley",
        "stackloss",
        "ccard",
        "copper",
        "statecrime",
        "scotland",
        "grunfeld",
    ]:
        out.append((nm, *_sm(nm)))
    return out


def splits():
    out = []
    for nm, ld, k in [
        ("diabetes", datasets.load_diabetes, 10),
        ("california", datasets.fetch_california_housing, 8),
        ("wine", datasets.load_wine, 13),
    ]:
        d = ld()
        out.append((nm, np.asarray(d.data, float)[:, :k], np.asarray(d.target, float)))
    for nm in ["engel", "stackloss"]:
        out.append((nm, *_sm(nm)))
    return out


def prep(X, y, cap=80):
    """Standardize an intercept model after deterministic evenly spaced capping."""
    m = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[m], y[m]
    if len(y) > cap:
        idx = np.linspace(0, len(y) - 1, cap).round().astype(int)
        X, y = X[idx], y[idx]
    X = np.column_stack([np.ones(len(y)), StandardScaler().fit_transform(X)])
    return X, (y - y.mean()) / (y.std() + 1e-12)
