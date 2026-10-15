"""Summarize the saved case-study results."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import math

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    groups = defaultdict(list)
    timing_groups = defaultdict(list)
    outcomes = Counter()
    for path in sorted((ROOT / "records/runs").glob("*.json")):
        r = json.loads(path.read_text())
        outcomes[(r.get("kind", "unknown"), r.get("status", "unknown"))] += 1
        key = (r.get("task"), r.get("n"), r.get("protocol"), r.get("method"))
        if r.get("status") == "complete":
            timing_groups[key].append(r)
        if r.get("kind") != "science":
            continue
        groups[key].append(r)
    for (task, n, protocol, method), records in sorted(groups.items()):
        complete = [r for r in records if r.get("status") == "complete"]
        undefined = [r for r in complete if r.get("metrics", {}).get("ratio_undefined", False)]
        raw_ratios = [r["engine_result"]["ratio"] for r in complete if not r.get("metrics", {}).get("ratio_undefined", False)]
        ratios = [v for v in raw_ratios if isinstance(v, (int, float)) and math.isfinite(v)]
        cert = [r["engine_result"].get("numerical_certificate", False) for r in complete]
        branch = Counter(r["engine_result"].get("branch", "unknown") for r in complete)
        row = dict(task=task, n=n, protocol=protocol, method=method,
                   recorded=len(records), complete=len(complete),
                   incomplete=len(records)-len(complete),
                   nonfinite_ratio_count=len(raw_ratios)-len(ratios),
                   undefined_ratio_count=len(undefined),
                   mean_ratio=statistics.mean(ratios) if ratios else None,
                   sd_ratio=statistics.stdev(ratios) if len(ratios)>1 else None,
                   min_ratio=min(ratios) if ratios else None,
                   max_ratio=max(ratios) if ratios else None,
                   numerical_recovery_count=sum(cert),
                   branches=json.dumps(dict(branch), sort_keys=True))
        for field in ["coefficient_error_relative", "residual_cancellation_scaled", "support_size"]:
            vals = [r["engine_result"][field] for r in complete]
            row["max_"+field] = max(vals) if vals else None
        for field in ["test_rmse_original", "test_prediction_difference_rmse_original"]:
            vals = [r["metrics"][field] for r in complete]
            row["mean_"+field] = statistics.mean(vals) if vals else None
            row["max_"+field] = max(vals) if vals else None
        times = [r["timing"]["native_method_s"] for r in timing_groups[(task,n,protocol,method)]]
        row["timing_count"] = len(times)
        row["mean_native_method_s"] = statistics.mean(times) if times else None
        row["min_native_method_s"] = min(times) if times else None
        row["max_native_method_s"] = max(times) if times else None
        rows.append(row)
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    if rows:
        with (out / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summary = {"record_outcomes": {"/".join(k):v for k,v in outcomes.items()},
               "groups":rows,
               "interpretation":"Summaries by task and budget. Standard deviations describe stochastic repetitions. Counts include every recorded run; metric summaries use completed runs."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary["record_outcomes"]))


if __name__ == "__main__":
    main()
