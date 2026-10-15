"""Independently verify every JSON/CSV aggregate from saved run records.

Standard-library only: no experiment, array loading, engine, summarizer, or
report-builder imports. Run after the campaign and summarizer finish.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


RTOL, ATOL = 1e-10, 1e-12
MAX_RESULT_FIELDS = ("coefficient_error_relative", "residual_cancellation_scaled", "support_size")
METRIC_FIELDS = ("test_rmse_original", "test_prediction_difference_rmse_original")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(row):
    return row["task"], int(row["n"]), row["protocol"], row["method"]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def average(values):
    return math.fsum(values) / len(values) if values else None


def sample_sd(values):
    if len(values) < 2:
        return None
    center = average(values)
    return math.sqrt(math.fsum((value - center) ** 2 for value in values) / (len(values) - 1))


class Audit:
    def __init__(self):
        self.issues = []
        self.checked = 0

    def require(self, condition, check, **details):
        self.checked += 1
        if not condition:
            self.issues.append({"check": check, **details})

    def compare(self, actual, expected, check, **details):
        if isinstance(expected, float):
            good = finite(actual) and math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL)
        elif expected is None:
            good = actual is None
        elif isinstance(expected, int):
            good = isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
        else:
            good = actual == expected
        self.require(good, check, actual=actual, expected=expected, **details)


def values(records, section, field, audit, group):
    result = []
    for filename, record in records:
        value = record.get(section, {}).get(field)
        audit.require(finite(value), "finite_metric_for_aggregate", group=group, record=filename,
                      section=section, field=field, value=value)
        if finite(value):
            result.append(value)
    return result


def expected_rows(records, audit):
    groups, timings = defaultdict(list), defaultdict(list)
    outcomes = Counter()
    for filename, record in records:
        group = identity(record)
        audit.require(record.get("kind") in {"science", "timing"}, "known_run_kind", record=filename)
        outcomes[f'{record.get("kind", "unknown")}/{record.get("status", "unknown")}'] += 1
        if record.get("status") == "complete":
            timings[group].append((filename, record))
        if record.get("kind") == "science":
            groups[group].append((filename, record))
        if record.get("kind") == "timing":
            d = record.get("engine_result", {}).get("d")
            if isinstance(d, int):
                audit.require(record["n"] == math.ceil(1.5 * d), "timing_only_at_primary_budget", record=filename)
            audit.require(record.get("rep") == 0 and record.get("trial") in (1, 2),
                          "timing_only_trial_identity", record=filename)
    result = {}
    timing_membership = {}
    for group, observed in sorted(groups.items()):
        complete = [(name, r) for name, r in observed if r.get("status") == "complete"]
        ratios, undefined, nonfinite = [], 0, 0
        branches, certificates = Counter(), 0
        for filename, record in complete:
            engine = record["engine_result"]
            denominator = engine.get("full_loss")
            audit.require(finite(denominator) and denominator >= 0, "valid_ratio_denominator", record=filename)
            is_undefined = not finite(denominator) or denominator <= 0
            audit.require(record.get("metrics", {}).get("ratio_undefined") is is_undefined,
                          "undefined_flag_matches_denominator", record=filename)
            audit.require(record.get("metrics", {}).get("ratio_denominator_positive") is (not is_undefined),
                          "denominator_positive_flag", record=filename)
            if is_undefined:
                undefined += 1
            elif finite(engine.get("ratio")):
                ratios.append(engine["ratio"])
            else:
                nonfinite += 1
            branches[engine.get("branch", "unknown")] += 1
            certificates += engine.get("numerical_certificate") is True
        task, n, protocol, method = group
        row = {"task": task, "n": n, "protocol": protocol, "method": method,
               "recorded": len(observed), "complete": len(complete), "incomplete": len(observed) - len(complete),
               "nonfinite_ratio_count": nonfinite, "undefined_ratio_count": undefined,
               "mean_ratio": average(ratios), "sd_ratio": sample_sd(ratios),
               "min_ratio": min(ratios) if ratios else None, "max_ratio": max(ratios) if ratios else None,
               "numerical_recovery_count": certificates, "branches": dict(branches)}
        for field in MAX_RESULT_FIELDS:
            metric = values(complete, "engine_result", field, audit, group)
            row["max_" + field] = max(metric) if metric else None
        for field in METRIC_FIELDS:
            metric = values(complete, "metrics", field, audit, group)
            row["mean_" + field], row["max_" + field] = average(metric), max(metric) if metric else None
        metric = values(timings[group], "timing", "native_method_s", audit, group)
        row.update(timing_count=len(timings[group]), mean_native_method_s=average(metric),
                   min_native_method_s=min(metric) if metric else None, max_native_method_s=max(metric) if metric else None)
        timing_membership[str(group)] = {
            "science_count": sum(r["kind"] == "science" for _, r in timings[group]),
            "timing_only_count": sum(r["kind"] == "timing" for _, r in timings[group]),
            "records": [filename for filename, _ in timings[group]],
        }
        result[group] = row
    return result, dict(outcomes), timing_membership


def check_rows(rows, expected, audit, source, from_csv=False):
    keyed = {}
    for row in rows:
        key = identity(row)
        audit.require(key not in keyed, "unique_aggregate_group", source=source, group=key)
        keyed[key] = row
    audit.require(set(keyed) == set(expected), "all_and_only_expected_aggregate_groups", source=source,
                  missing=[list(k) for k in sorted(set(expected) - set(keyed))],
                  extra=[list(k) for k in sorted(set(keyed) - set(expected))])
    for group in set(keyed) & set(expected):
        saved, row = keyed[group], expected[group]
        audit.require(set(saved) == set(row), "all_aggregate_fields_checked", source=source, group=group,
                      missing=sorted(set(row) - set(saved)), extra=sorted(set(saved) - set(row)))
        for field in set(saved) & set(row):
            actual, wanted = saved[field], row[field]
            try:
                if field == "branches" and isinstance(actual, str):
                    actual = json.loads(actual)
                elif from_csv:
                    if wanted is None:
                        actual = None if actual == "" else actual
                    elif isinstance(wanted, int):
                        actual = int(actual)
                    elif isinstance(wanted, float):
                        actual = float(actual)
            except (ValueError, TypeError) as error:
                audit.require(False, "aggregate_value_parse", source=source, group=group,
                              field=field, value=actual, error=str(error))
                continue
            audit.compare(actual, wanted, "aggregate_value", source=source, group=group, field=field)


def run(root):
    audit = Audit()
    paths = sorted((root / "records/runs").glob("*.json"))
    summary_path, csv_path = root / "analysis/summary.json", root / "analysis/summary.csv"
    tracked = paths + [summary_path, csv_path]
    hashes = {str(path.relative_to(root)): digest(path) for path in tracked}
    state = json.loads((root / "records/campaign_state.json").read_text())
    audit.require(state.get("active_attempt") is None and state.get("status") in
                  {"complete", "execution_complete_pending_independent_verification", "execution_ended_with_failures"},
                  "campaign_ended_before_aggregate_verification", status=state.get("status"))
    if audit.issues:
        return {"issues": audit.issues, "summary": {"passed": False, "issue_count": len(audit.issues)}}
    records = [(path.name, json.loads(path.read_text())) for path in paths]
    expected, outcomes, timing_membership = expected_rows(records, audit)
    summary = json.loads(summary_path.read_text())
    audit.compare(summary.get("record_outcomes"), outcomes, "all_record_outcome_counts")
    check_rows(summary.get("groups", []), expected, audit, "summary.json")
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        csv_rows = list(reader)
        audit.require(len(reader.fieldnames or []) == len(set(reader.fieldnames or [])), "unique_csv_columns")
    check_rows(csv_rows, expected, audit, "summary.csv", from_csv=True)
    audit.require(paths == sorted((root / "records/runs").glob("*.json")), "record_set_unchanged_during_aggregate_audit")
    for path in tracked:
        audit.require(hashes[str(path.relative_to(root))] == digest(path), "source_unchanged_during_aggregate_audit",
                      path=str(path.relative_to(root)))
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "independence": "No engine, summarizer, or report-builder imports; means and sample SD independently recomputed with math.fsum.",
            "comparison_tolerances": {"relative": RTOL, "absolute": ATOL},
            "source_hashes": hashes, "expected_aggregates": list(expected.values()),
            "timing_membership": timing_membership, "record_outcomes": outcomes, "issues": audit.issues,
            "summary": {"passed": not audit.issues, "issue_count": len(audit.issues),
                        "scientific_groups_checked": len(expected), "run_records_read": len(records),
                        "json_and_csv_values_checked": audit.checked,
                        "scope": "Aggregate arithmetic and membership audit; no new experiment or scientific fit."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        report = run(root)
    except Exception as error:
        report = {"issues": [{"check": "aggregate_verification_exception", "type": type(error).__name__, "message": str(error)}],
                  "summary": {"passed": False, "issue_count": 1}}
    output = args.output or root / "analysis/summary_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"aggregate_verification_report": str(output), **report["summary"]}))
    if report["issues"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
