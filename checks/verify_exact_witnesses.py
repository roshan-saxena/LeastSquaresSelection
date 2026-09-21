"""Check the five saved rational witnesses without repeating LP discovery."""

from experiments.paths import RESULTS, OUTPUT
from pathlib import Path
from fractions import Fraction as F
import json, numpy as np
from spancancel.datasets_local import load_input
from experiments.exact_certificate_checks import gauss


def main():
    R = Path(__file__).resolve().parents[1]
    out = []
    for z in json.loads((RESULTS / "exact_certificates.json").read_text()):
        q = load_input(z["data_id"])
        X = [[F(float(v)) for v in row] for row in q["X"]]
        y = [F(float(v)) for v in q["y"]]
        w = list(map(F, z["full_coefficients_rational"]))
        a = list(map(F, z["weights_rational"]))
        S = z["indices"]
        d = len(w)
        assert (
            len(S) <= z["n"]
            and len(set(S)) == len(S)
            and all(v > 0 for v in a)
            and sum(a) == 1
        )
        r = [sum(v * t for v, t in zip(x, w)) - b for x, b in zip(X, y)]
        assert all(sum(x[j] * e for x, e in zip(X, r)) == 0 for j in range(d))
        assert all(
            sum(aa * X[i][j] * r[i] for i, aa in zip(S, a)) == 0 for j in range(d)
        )
        assert sum(e * e for e in r) == F(z["full_loss_rational"])
        gauss(
            [[sum(X[i][j] * X[i][k] for i in S) for k in range(d)] for j in range(d)],
            [F(1)] * d,
        )
        out.append(
            dict(
                data_id=z["data_id"],
                verified=True,
                domain="stored binary64 inputs interpreted as exact rationals",
            )
        )
    (OUTPUT / "exact_witness_validation.json").write_text(json.dumps(out, indent=2))
    print("Verified", len(out), "fixed rational witnesses.")


if __name__ == "__main__":
    main()
