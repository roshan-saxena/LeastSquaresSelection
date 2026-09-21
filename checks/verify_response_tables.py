"""Check response-only table values against the same saved runs used by the paper."""

from experiments.paths import RESULTS, OUTPUT


from pathlib import Path
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    return json.loads((ROOT / "checks/expected_tables.json").read_text())[
        name.removesuffix(".tex")
    ]


def numeric(cell):
    return [float(v) for v in re.findall(r"[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?", cell)]


def main():
    summaries = pd.read_csv(OUTPUT / "figure2_dataset_table.csv")
    checks = pd.read_csv(OUTPUT / "figure2_checks.csv")
    methods = ["uniform", "leverage", "leverage-w", "D-optimal", "SpanCancel"]
    report = {}
    name = "response_figure2_dataset.tex"
    for row in rows(name):
        dataset = row[0]
        d = int(checks.loc[checks.dataset.eq(dataset), "d"].iloc[0])
        assert numeric(row[1]) == [d]
        assert numeric(row[2]) == [round(1.5 * d)]
        for method, cell in zip(methods, row[3:]):
            value = summaries.loc[
                summaries.dataset.eq(dataset) & summaries.method.eq(method)
            ].iloc[0]
            assert numeric(cell) == [
                float(f'{value["mean"]:.3f}'),
                float(f'{value["std"]:.3f}'),
            ], (dataset, method)
    report[name] = {"rows": len(rows(name)), "passed": True}

    name = "response_figure2_checks.tex"
    for row in rows(name):
        dataset, d, n = row[:3]
        group = checks[
            checks.dataset.eq(dataset) & checks.d.eq(int(d)) & checks.n.eq(int(n))
        ]
        h = group[group.branch.eq("heuristic")]
        expected = [
            len(group),
            int((group.branch.eq("lp") & group.joint).sum()),
            int(group.branch.eq("square").sum()),
            len(h),
            int(h.near.sum()),
            int(h.joint.sum()),
        ]
        assert [int(c) for c in row[3:]] == expected, (dataset, n, expected)
    report[name] = {"rows": len(rows(name)), "passed": True}

    records = [
        json.loads(line) for line in (RESULTS / "stress.jsonl").read_text().splitlines()
    ]
    frontier = [
        r
        for r in records
        if r.get("record_type") == "frontier"
        and r.get("method") == "SpanCancel"
        and r.get("protocol") == "native"
    ]
    name = "response_frontier_checks.tex"
    for row in rows(name):
        d, N, n = map(int, row[:3])
        matches = [
            r
            for r in frontier
            if r["data_id"] == f"frontier_blocks_{d}" and r["n"] == n
        ]
        assert len(matches) == 1
        r = matches[0]
        assert r["N"] == N and int(row[3]) == r["support_rank"]
        assert row[4] == r["branch"].replace("_", " ")
        assert float(row[5]) == float(f'{r["ratio"]:.6f}')
        assert row[6] == ("Yes" if r["numerical_certificate"] else "No")
    report[name] = {"rows": len(rows(name)), "passed": True}
    (OUTPUT / "response_table_validation.json").write_text(json.dumps(report, indent=2))
    print(
        "Verified",
        sum(r["rows"] for r in report.values()),
        "rows in three response tables.",
    )


if __name__ == "__main__":
    main()
