# Data Selection for Least Squares

Code for **Data Selection for Least Squares: Worst-Case Ratios and Residual-Cancelling Coresets**, by Roshan Saxena.

SpanCancel selects weighted rows using the full-data least-squares residual. This repository contains the method, comparison methods, experiment runners, and records needed to reproduce the reported numerical results.

## Install

Use Python 3.12. Run these commands from the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install numpy==1.26.4 scipy==1.13.1 scikit-learn==1.4.2 statsmodels==0.14.2 pandas==2.2.2 matplotlib==3.8.4 threadpoolctl==3.6.0
```

## Use the method

```python
import numpy as np
from spancancel import Problem, select, fit

rng = np.random.default_rng(0)
X = np.column_stack((np.ones(40), rng.normal(size=(40, 3))))
y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(size=40)

problem = Problem(X, y)
selected = select(problem, n=8, seed=0)
coefficients = fit(X, y, selected["indices"], selected["weights"])
print(selected["ratio"])
```

Supply a finite matrix `X` with shape `(N, d)` and a one-dimensional target `y`. Add an intercept column explicitly if needed. The integer budget `n` lies between `d` and `N`. `Problem` computes the full-data fit; `select` returns row indices, weights, loss ratio, selected branch, and recovery diagnostics. `Config` exposes the numerical tolerances and fallback settings. Available methods are `SpanCancel`, `uniform`, `leverage`, `leverage-w`, `D-optimal`, and `vol-sampling`.

## Reproduce the paper

Regenerate the numerical tables, statistical summaries, and Figures 1–4. Figure 1 is calculated directly from the constructed-bound formula; the tables and Figures 2–4 use the supplied records:

```sh
python -m experiments.reproduce
```

To generate Figure 1 alone:

```sh
python -m analysis.plot_figure1
```

Outputs are written to `results/generated/`. Tables are CSV files; uncertainty cells use mean ± sample SD. To reconstruct dataset inputs, check the saved fits and table values, and rerun 23 representative selections:

```sh
python -m experiments.reproduce --verify --replay --download-california
```

The download flag permits scikit-learn to fetch California housing. Other public datasets use the installed library loaders. These commands use one numerical thread. Add `--include-response` to also check the detailed numerical tables cited in the reviewer response. All generated files remain separate from supplied results.

## Run new experiments

After preparing inputs with the verification command above:

```sh
python -m experiments.run_experiments controlled
python -m experiments.reproduce --results-dir results/generated/rerun/results
```

The second command summarizes only the new logs, writing to `results/generated/rerun/analysis/`. It does not substitute the supplied paper results. Other main suites are `rff`, `stress`, `diagnostics`, `adversarial`, and `runtime`. Set `MLWA_RUN_DIR` to choose a different run destination. Repeating a suite resumes completed run keys; use a new destination when changing its settings. Full suites can take substantially longer than saved-result verification.

The construction and conditioning studies have separate runners:

```sh
python -m experiments.known_constructions
python -m experiments.conditioning
python -m experiments.stronger_chain_search
python -m experiments.exact_certificate_checks
```

These follow-up commands use the supplied fixed inputs and, where required, saved conditioning records. The complete Figure 2 experiment uses:

```sh
python -m experiments.figure2.reproduce
```

## Files and protocols

- `spancancel/`: selection methods, weighted fitting, and dataset routines.
- `experiments/`: runners and structured-input generators. `protocols/` records the reported settings and tested environment; the runners implement those settings.
- `analysis/`: statistical summaries, CSV tables, and plotting.
- `checks/`: numerical verification and expected table values.
- `data/`: `inputs.npz` stores arrays as `input_id/array_name`; `metadata.json` holds generation and preprocessing records; `input_manifest.json` lists identifiers and array hashes.
- `results/`: per-run records used by the paper. Generated outputs are ignored by Git.

Figures 2–3 use the Figure 2 protocol: pre-split standardization, a shared random stream, weighted normal equations, and training ratios floored at one. The other experiments use direct SVD fits and unclipped losses; Fourier-feature preprocessing is fitted on training rows. Native weights and common post-selection reweighting are separate protocols. The table-generation code selects the reported cohorts.

The JSONL records retain Python's `NaN` and `Infinity` values for undefined or non-finite numerical quantities. Python's `json` reader accepts these; strict JSON readers require conversion. Historical timings describe the recorded machine and runs. Fresh timings depend on the local environment.

## Citation and license

```bibtex
@unpublished{saxena2026selection,
  author = {Roshan Saxena},
  title = {Data Selection for Least Squares: Worst-Case Ratios and Residual-Cancelling Coresets},
  year = {2026}
}
```

Original code is under the [MIT license](LICENSE). Dependencies and datasets retain their own terms. Public dataset observations are loaded locally; synthetic inputs and numerical results are included. See the [statsmodels dataset notices](https://github.com/statsmodels/statsmodels/tree/main/statsmodels/datasets) and [scikit-learn dataset documentation](https://scikit-learn.org/stable/datasets.html). Statsmodels retains the original authors' rights for Ccard, Copper, and Scotland.
