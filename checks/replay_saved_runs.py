"""Replay saved selection and reweighting runs."""

from experiments.paths import RESULTS
from pathlib import Path
import argparse, json, sys, time
import numpy as np
from threadpoolctl import threadpool_limits
from spancancel.engine import Config, Problem, reweight, select
from spancancel.datasets_local import load_input


def main():
    sys.dont_write_bytecode = True

    ROOT = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="controlled")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--branch", default=None)
    p.add_argument("--tolerance", type=float, default=1e-9)
    p.add_argument("--output", default="results/generated/replay_smoke.json")
    args = p.parse_args()
    rows = [
        json.loads(x)
        for x in (RESULTS / f"{args.suite}.jsonl").read_text().splitlines()
    ]
    out = []
    cache = {}
    with threadpool_limits(1):
        for r in rows:
            if (
                r.get("status") != "ok"
                or r.get("reused_from")
                or r.get("protocol") not in ("native", "common")
                or "config" not in r
            ):
                continue
            if args.branch and r.get("branch") != args.branch:
                continue
            cfg = Config(**r["config"])
            key = (r["data_id"], cfg.solver_rcond)
            if key not in cache:
                z = load_input(r["data_id"])
                cache[key] = Problem(
                    z["X"],
                    z["y"],
                    cfg,
                    z["Xtest"] if "Xtest" in z else None,
                    z["ytest"] if "ytest" in z else None,
                )
            P = cache[key]
            if r["protocol"] == "common":
                rr = reweight(
                    P,
                    dict(
                        indices=r["original_support"],
                        method=r["method"],
                        selection_s=0.0,
                        candidate_s=0.0,
                    ),
                    cfg,
                    r["seed"],
                )
            else:
                rr = select(P, r["n"], r["method"], cfg, r["seed"])
            out.append(
                dict(
                    key=r["key"],
                    stored_ratio=r["ratio"],
                    replayed_ratio=rr["ratio"],
                    relative_difference=abs(rr["ratio"] - r["ratio"])
                    / max(1.0, abs(r["ratio"])),
                )
            )
            if args.limit and len(out) >= args.limit:
                break
    f = ROOT / args.output
    f.parent.mkdir(exist_ok=True, parents=True)
    f.write_text(json.dumps(out, indent=2))
    print(
        "Replayed",
        len(out),
        "runs; max relative difference",
        max((x["relative_difference"] for x in out), default=0.0),
    )

    if not out:
        raise SystemExit("No saved runs match the requested suite and branch.")
    if any(
        not np.isfinite(x["relative_difference"])
        or x["relative_difference"] > args.tolerance
        for x in out
    ):
        raise SystemExit(
            f"Replay differences exceed tolerance {args.tolerance:g}; see {f}."
        )


if __name__ == "__main__":
    main()
