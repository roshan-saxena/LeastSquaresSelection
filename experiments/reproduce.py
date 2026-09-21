"""Reproduce paper results or summarize a fresh experiment run."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify saved inputs, fits and table values",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="prepare inputs and rerun representative selections",
    )
    parser.add_argument(
        "--download-california",
        action="store_true",
        help="allow the upstream dataset download",
    )
    parser.add_argument(
        "--include-response",
        action="store_true",
        help="also reproduce the response-specific numerical tables",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="summarize this experiment result directory instead of supplied results",
    )
    parser.add_argument(
        "--output", type=Path, help="destination for generated summaries and figures"
    )
    args = parser.parse_args()
    if args.results_dir and (
        args.verify or args.replay or args.download_california or args.include_response
    ):
        parser.error(
            "Use --results-dir for new-run summaries; verification and response checks apply to the supplied paper records."
        )
    if args.download_california and not (args.verify or args.replay):
        parser.error("Use --download-california together with --verify or --replay.")
    results = (args.results_dir or ROOT / "results").resolve()
    if not results.is_dir() or not any(results.glob("*.jsonl")):
        parser.error(f"No experiment logs found in {results}")
    output = (
        args.output
        or (
            results.parent / "analysis"
            if args.results_dir
            else ROOT / "results/generated"
        )
    ).resolve()
    if output == results:
        parser.error(
            "The output directory must differ from the saved result directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        MLWA_RESULTS_DIR=str(results),
        MLWA_OUTPUT_DIR=str(output),
        MLWA_INCLUDE_RESPONSE="1" if args.include_response else "0",
    )
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"

    def run(module, *arguments):
        print(f"\nRunning {module}", flush=True)
        subprocess.run(
            [sys.executable, "-m", module, *arguments], cwd=ROOT, env=env, check=True
        )

    if args.verify or args.replay:
        run(
            "checks.prepare_inputs",
            *(["--download-california"] if args.download_california else []),
        )
    run("analysis.analyze_results")
    if args.results_dir:
        print(
            f"\nNew-run summaries saved to {output}. Only the selected logs were used."
        )
        return
    for module in (
        "analysis.reproduce_tables",
        "checks.verify_formula_table",
        "analysis.plot_figure2_saved",
        "analysis.plot_figure3",
        "analysis.plot_budget_frontier",
    ):
        run(module)
    if args.verify:
        for module in (
            "checks.validate_results",
            "checks.verify_figure2",
            "checks.verify_exact_witnesses",
            "checks.independent_checks",
        ):
            run(module)
    if args.include_response:
        if not args.verify:
            run("checks.prepare_inputs")
            run("checks.verify_figure2")
        run("checks.verify_response_tables")
    if args.replay:
        for suite, limit, branch, filename in (
            ("controlled", 12, None, "controlled"),
            ("stress", 8, None, "stress"),
            ("stress", 2, "heuristic", "fallback"),
            ("rff", 1, "lp", "rff"),
        ):
            run(
                "checks.replay_saved_runs",
                "--suite",
                suite,
                "--limit",
                str(limit),
                "--output",
                str(output / f"replay_{filename}.json"),
                *(["--branch", branch] if branch else []),
            )
    print(f"\nResults and checks saved to {output}.")


if __name__ == "__main__":
    main()
