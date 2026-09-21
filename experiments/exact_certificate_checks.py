"""Verify finite positive-weight recovery certificates on the exact stored binary inputs.
Fractions are used only to check fixed small systems, not to prove the conjecture.
"""

from experiments.paths import OUTPUT


from pathlib import Path
from fractions import Fraction as F
import json, time, sys, hashlib

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from .conditioning import solve_decimal
from spancancel.datasets_local import load_input

ROOT = Path(__file__).resolve().parents[1]


def gauss(A, b):
    n = len(b)
    T = [list(a) + [bb] for a, bb in zip(A, b)]
    for j in range(n):
        q = max(range(j, n), key=lambda i: abs(T[i][j]))
        T[j], T[q] = T[q], T[j]
        if T[j][j] == 0:
            raise ValueError("singular exact system")
        for i in range(j + 1, n):
            z = T[i][j] / T[j][j]
            for k in range(j, n + 1):
                T[i][k] -= z * T[j][k]
    out = [F(0)] * n
    for i in range(n - 1, -1, -1):
        out[i] = (T[i][-1] - sum(T[i][j] * out[j] for j in range(i + 1, n))) / T[i][i]
    return out


def verify(dataid, n):
    t = time.perf_counter()
    z = load_input(dataid)
    X = z["X"]
    y = z["y"]
    N, d = X.shape
    XX = [[F(float(v)) for v in row] for row in X]
    yy = [F(float(v)) for v in y]
    A = [[sum(x[i] * x[j] for x in XX) for j in range(d)] for i in range(d)]
    b = [sum(x[i] * y for x, y in zip(XX, yy)) for i in range(d)]
    w = gauss(A, b)
    r = [sum(a * b for a, b in zip(x, w)) - y for x, y in zip(XX, yy)]
    assert all(sum(x[j] * rr for x, rr in zip(XX, r)) == 0 for j in range(d))
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Z = U @ Vt
    C = Z * np.array([float(v) for v in r])[:, None]
    attempts = []
    for obj in [np.zeros(N), -np.linalg.norm(C, axis=1), np.arange(N) / N]:
        lp = linprog(
            obj,
            A_eq=np.vstack([C.T, np.ones(N)]),
            b_eq=np.r_[np.zeros(d), 1],
            bounds=(0, None),
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-10,
                "dual_feasibility_tolerance": 1e-10,
            },
        )
        at = dict(LP_status=lp.status, success=bool(lp.success))
        attempts.append(at)
        if not lp.success:
            continue
        S = np.flatnonzero(lp.x > 1e-9)
        at["candidate_support"] = S.tolist()
        if len(S) != d + 1 or len(S) > n:
            at["rejection"] = "support size"
            continue
        try:
            E = [[r[i] * XX[i][j] for i in S] for j in range(d)] + [[F(1)] * len(S)]
            aa = gauss(E, [F(0)] * d + [F(1)])
            assert sum(aa) == 1 and all(v > 0 for v in aa)
            assert all(
                sum(aa[k] * r[i] * XX[i][j] for k, i in enumerate(S)) == 0
                for j in range(d)
            )
            G = [
                [sum(XX[i][j] * XX[i][k] for i in S) for k in range(d)]
                for j in range(d)
            ]
            gauss(G, [F(1)] * d)
            return dict(
                data_id=dataid,
                d=d,
                N=N,
                n=n,
                indices=S.tolist(),
                weights_rational=[str(v) for v in aa],
                full_coefficients_rational=[str(v) for v in w],
                full_loss_rational=str(sum(v * v for v in r)),
                exact_positive_weights=True,
                exact_cancellation=True,
                exact_full_support_rank=True,
                exact_loss_ratio="1",
                certificate_domain="exact binary64 input values interpreted as rationals",
                discovery_variant="independent full-pool LP with invertible feature preconditioning, then exact rational equality solve; not the main algorithm",
                attempts=attempts,
                seconds=time.perf_counter() - t,
                status="certified",
            )
        except (AssertionError, ValueError) as e:
            at["rejection"] = repr(e)
    return dict(
        data_id=dataid,
        n=n,
        status="not_certified",
        attempts=attempts,
        seconds=time.perf_counter() - t,
    )


if __name__ == "__main__":
    targets = [
        ("conditioning_1e+06", 8),
        ("conditioning_1e+08", 8),
        ("conditioning_1e+10", 8),
        ("conditioning_1e+12", 8),
        ("controlled_diabetes", 9),
    ]
    p = dict(
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        reason="Verify fixed recovery witnesses using exact rational arithmetic.",
        targets=targets,
        discovery_objectives=3,
        selection_tolerance=1e-9,
        verification="exact Fraction linear solves and positivity/cancellation/full-rank checks",
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (OUTPUT / "exact_certificate_protocol.json").write_text(json.dumps(p, indent=2))
    rows = []
    with threadpool_limits(1):
        for dataid, n in targets:
            r = verify(dataid, n)
            rows.append(r)
            print(dataid, r["status"], r["seconds"], flush=True)
    (OUTPUT / "exact_certificates.json").write_text(json.dumps(rows, indent=2))
