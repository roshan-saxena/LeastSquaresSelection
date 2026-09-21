"""Exact rational evaluation of the eight partition-formula entries in Table 3."""

from experiments.paths import OUTPUT
from pathlib import Path
from fractions import Fraction as F
import json


def main():
    R = Path(__file__).resolve().parents[1]
    cases = [
        (2, 3, F(3, 2)),
        (3, 4, F(5, 3)),
        (3, 5, F(4, 3)),
        (4, 5, F(2)),
        (4, 6, F(3, 2)),
        (4, 7, F(5, 4)),
        (5, 6, F(11, 5)),
        (6, 7, F(5, 2)),
    ]
    out = []
    for d, n, expected in cases:
        vals = []
        for g in range(n - d + 1, d + 1):
            q, r = divmod(d, g)
            vals.append(1 + F(g - (n - d), 1) / (F(r, q + 1) + F(g - r, q)))
        value = max(vals)
        assert value == expected
        out.append(dict(d=d, n=n, value=str(value), verified=True))
    (OUTPUT / "formula_table_validation.json").write_text(json.dumps(out, indent=2))
    print("Verified eight Table 3 formula entries exactly.")


if __name__ == "__main__":
    main()
