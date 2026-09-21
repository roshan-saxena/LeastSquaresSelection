"""Exact rational witnesses and numerical solver checks."""

from experiments.paths import OUTPUT
from pathlib import Path
from fractions import Fraction
import itertools, json, sys
import numpy as np
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits
from spancancel.engine import Problem, fit, qr


def main():
    sys.dont_write_bytecode = True

    ROOT = Path(__file__).resolve().parents[1]
    checks = []
    with threadpool_limits(1):
        # Exact rational antipodal-block model: all 14 nonempty supports of at most three rows.
        # On a represented coordinate a lone row forces error +/-1. A pair can achieve error0;
        # on an absent coordinate minimum norm forces coefficient0. Orthogonal loss decomposes.
        X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        w = np.array([20.0, 37.0])
        y = X @ w - np.ones(4)
        P = Problem(X, y)
        entries = []
        for m in (1, 2, 3):
            for S in itertools.combinations(range(4), m):
                penalty = 0
                for axis in (0, 1):
                    c = sum(i // 2 == axis for i in S)
                    penalty += 0 if c == 2 else (1 if c == 1 else int(w[axis]) ** 2)
                optimum = Fraction(4 + 2 * penalty, 4)
                entries.append(
                    dict(
                        indices=list(S),
                        exact_infimum_ratio=str(optimum),
                        attained=True,
                        justification="independent coordinates; paired rows equal positive weights cancel, singleton fits exactly, absent coordinate minimum norm zero",
                    )
                )
        witness = P.evaluate([0, 1, 2], np.ones(3))
        assert min(Fraction(e["exact_infimum_ratio"]) for e in entries) == Fraction(
            3, 2
        )
        assert abs(witness["ratio"] - 1.5) < 1e-12
        np.savez_compressed(OUTPUT / "exact_antipodal_d2_n3.npz", X=X, y=y)
        checks.append(
            dict(
                name="exact_rational_d2_n3",
                N=4,
                d=2,
                n=3,
                global_optimum="3/2",
                all_qualifying_supports=14,
                supports=entries,
                numerical_witness=witness,
                certificate_type="exact rational coordinate decomposition for this fixed construction",
            )
        )
        # Independent LAPACK drivers and explicit pseudoinverse on full and deficient selected designs.
        rng = np.random.default_rng(944)
        for shape in ((9, 4), (2, 4), (7, 4)):
            X = rng.normal(size=shape)
            X[-1] = X[0] if shape[0] == 7 else X[-1]
            y = rng.normal(size=shape[0])
            a = rng.uniform(0.01, 2, size=len(y))
            A = X * np.sqrt(a)[:, None]
            b = y * np.sqrt(a)
            ours = fit(X, y, np.arange(len(y)), a)
            svd = lstsq(A, b, cond=1e-12, lapack_driver="gelsd")[0]
            qr = lstsq(A, b, cond=1e-12, lapack_driver="gelsy")[0]
            pinv = np.linalg.pinv(A, rcond=1e-12) @ b
            diffs = [np.linalg.norm(ours - z) for z in (svd, qr, pinv)]
            assert max(diffs) < 1e-10
            checks.append(
                dict(
                    name="weighted_minimum_norm",
                    shape=shape,
                    driver_differences=list(map(float, diffs)),
                )
            )
        # Explicit cancellation witness: all full rows equally weighted recovers the full fit.
        X = rng.normal(size=(14, 4))
        y = rng.normal(size=14)
        P = Problem(X, y)
        rr = P.evaluate(np.arange(14), np.ones(14))
        assert rr["numerical_certificate"]
        checks.append(dict(name="full_support_cancellation_check", result=rr))
        # Normal equations cutoff 1e-12 on X^TX drops directions already at kappa(X)>1e6.
        Q = np.linalg.qr(rng.normal(size=(20, 4)))[0]
        X = Q @ np.diag([1, 1e-3, 1e-6, 1e-8])
        w = np.array([1, 2, 3, 4.0])
        y = X @ w
        wd = np.linalg.lstsq(X, y, rcond=1e-12)[0]
        wn = np.linalg.pinv(X.T @ X, rcond=1e-12) @ (X.T @ y)
        checks.append(
            dict(
                name="normal_equation_cutoff",
                condition_X=float(np.linalg.cond(X)),
                direct_coefficient_error=float(np.linalg.norm(wd - w)),
                normal_equation_coefficient_error=float(np.linalg.norm(wn - w)),
                direct_retained_rank=4,
                normal_equation_retained_rank=int(
                    np.linalg.matrix_rank(X.T @ X, tol=1e-12)
                ),
            )
        )
    (OUTPUT / "independent_checks.json").write_text(json.dumps(checks, indent=2))
    print("Independent checks completed:", len(checks))


if __name__ == "__main__":
    main()
