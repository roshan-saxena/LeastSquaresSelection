# Additional regression case studies

Code and complete records for the Gas CO and Superconductivity case studies in “Data Selection for Least Squares: Worst-Case Ratios and Residual-Cancelling Coresets.”

## Contents

- `analysis/primary_comparison.csv`: the six methods at `ceil(1.5d)` reported in Table 7, with test RMSE, variability, timings and solver diagnostics.
- `analysis/summary.csv` and `summary.json`: all four budgets. `records/runs/` contains 276 scientific runs and eight additional timing trials, including supports, weights, coefficients and solver outcomes. `records/full_models/` contains the two reference fits.
- `protocol.json`: splits, preprocessing, methods, seeds and computational limits. `data/prepared/*.json` records the exact training and test indices' hashes, transformations and data checks.
- `records/environment.json` and `records/data_manifest.json`: software, hardware and dataset provenance. `file_hashes.json` lists checksums for every supplied file. `provenance.json` links the recorded source checksums to the supplied files.

All budgets, repetitions and baseline outcomes for both case studies are included. Timings include the scope identified in the manuscript; the full-fit reference costs are recorded separately. Raw data and generated fit caches are obtained during reproduction.

## Check the saved summaries

With Python 3.12, from this folder:

```sh
python source/verify_summary.py --root . --output summary_check.json
```

This independent check uses the saved records and requires no data download or new model selections.

## Reproduce the case studies

Run these commands on macOS or Linux; on Windows, use WSL.

Create a separate working directory and copy only `source/` and `protocol.json` into it. Run the following commands there:

```sh
python3.12 -m venv venv
source venv/bin/activate
python -m pip install numpy==1.26.4 scipy==1.13.1 scikit-learn==1.4.2 threadpoolctl==3.6.0
python source/prepare_data.py --tasks gas_co superconductivity
python source/run_study.py
python source/run_study.py --execute
python source/summarize.py
python source/verify_results.py --root . --output analysis/verification_report.json
python source/verify_summary.py --root . --output analysis/summary_verification.json
```

The first runner command displays the planned grid. Execution uses one worker and one numerical thread. The same execution command resumes an interrupted run. The numerical verifier independently reconstructs the selected fits and checks recovery using SVD and pivoted QR. The summarizer produces `summary.json` and `summary.csv`; the primary comparison is their `ceil(1.5d)` subset with the corresponding reference-fit and solver diagnostics.

The selection and fitting code is unchanged. The engine matches repository commit `cef3d15b2274a92a5b882fc2edf61e10a9ada933`. Numerical values and timings can vary with platform and libraries. Keep new runs separate from the supplied records so their provenance remains consistent.

## Data sources

- [Gas Turbine CO and NOx Emission Data Set (2019)](https://doi.org/10.24432/C5WC95).
- [Hamidieh (2018), Superconductivty Data](https://doi.org/10.24432/C53P47).

Both datasets are available from UCI under CC BY 4.0. Downloads, attribution and archive checksums are recorded in `records/data_manifest.json`. The base implementation and other manuscript reproduction materials are in [LeastSquaresSelection](https://github.com/roshan-saxena/LeastSquaresSelection).
