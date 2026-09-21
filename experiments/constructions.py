"""Structured inputs used by the paper's construction experiments."""

import numpy as np

RANK_TOL = 1e-8


def whiten(X):
    v, Q = np.linalg.eigh(X.T @ X)
    inv = np.zeros_like(v)
    keep = v > RANK_TOL**2
    inv[keep] = 1 / np.sqrt(v[keep])
    W = Q @ np.diag(inv) @ Q.T
    return X @ W, W


def simplex(s):
    C = np.eye(s + 1) - np.ones((s + 1, s + 1)) / (s + 1)
    U, S, _ = np.linalg.svd(C)
    Y = U[:, :s] * S[:s]
    v, Q = np.linalg.eigh(Y.T @ Y)
    return Y @ (Q @ np.diag(1 / np.sqrt(np.clip(v, 1e-14, None))) @ Q.T)


def blocks(dims):
    d = sum(dims)
    N = d + len(dims)
    X = np.zeros((N, d))
    r = np.zeros(N)
    i = j = 0
    for s in dims:
        e = 1.0 / s
        X[i : i + s + 1, j : j + s] = simplex(s)
        r[i : i + s + 1] = np.sqrt(e / (s + 1))
        i += s + 1
        j += s
    return X, -r


def chain(seed):
    V = np.array([[1, 0, 0, -1, 0], [0, 1, 0, -1, -1], [0, 0, 1, 0, -1]], float)
    V = V * np.array([1.0, 2, 1, 1, 1])
    r = np.random.default_rng(seed).standard_normal(5)
    r += np.sign(r) * 0.3
    X = (V / r).T
    X, _ = whiten(X)
    return X, r


def merge(delta, om):
    d = delta + 1
    X = np.zeros((delta + 2, d))
    r = np.zeros(delta + 2)
    X[: delta + 1, 1:] = simplex(delta)
    X[: delta + 1, 0] = np.sqrt(om / (delta + 1))
    r[: delta + 1] = 1 / np.sqrt(delta + 1)
    X[delta + 1, 0] = np.sqrt(1 - om)
    r[delta + 1] = -np.sqrt(om / (1 - om))
    return X, r
