"""Independent, read-only numerical audit of the frozen large-regression study.

Independently of engine.py and run_study.py, this module solves each
saved scientific fit from X/y/indices/weights, reconstructs raw residual vectors,
and cross-checks full fits and recovery claims with LAPACK's pivoted-QR gelsy.
Invoke only after the primary campaign stops, to avoid timing contention.
"""

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.linalg import lstsq as qr_lstsq
from threadpoolctl import threadpool_info, threadpool_limits


FROZEN_ENGINE_SHA256 = "bb7b73406a97fb523b5c7b5b23af2e04aaa2de001678568bee4e3472521020ba"
FROZEN_CONFIG = {
    "pool_factor": 8.0, "rank_atol": 1e-8, "solver_rcond": 1e-12,
    "lp_tol": 1e-9, "support_tol": 1e-9, "coefficient_tol": 1e-7,
    "restarts": 6, "maxiter": 400, "fatol": 1e-12, "xatol": 1e-8,
    "initialization": "dirichlet",
}
NEAR_ONE_TOL = 1e-7
SEED_NAMESPACE = "large_regression_v1"
# These are audit comparison tolerances, never scientific decision thresholds.
COMPARE_RTOL = 5e-9
COMPARE_ATOL = 5e-11


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        if math.isfinite(value):
            return float(value)
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


class Audit:
    def __init__(self, context):
        self.context = context
        self.issues = []

    def require(self, condition, check, **details):
        if not bool(condition):
            self.issues.append({"context": self.context, "check": check, **clean(details)})

    def scalar(self, saved, expected, check, rtol=COMPARE_RTOL, atol=COMPARE_ATOL):
        if expected is None:
            self.require(saved is None, check, saved=saved, expected=expected)
            return
        if math.isinf(float(expected)):
            # JSON normalization is allowed to represent nonfinite diagnostics as null.
            good = (saved == ("Infinity" if expected > 0 else "-Infinity")
                    or (isinstance(saved, (int, float)) and saved == expected))
        else:
            good = (isinstance(saved, (int, float)) and not isinstance(saved, bool)
                    and math.isfinite(float(saved))
                    and math.isclose(float(saved), float(expected), rel_tol=rtol, abs_tol=atol))
        self.require(good, check, saved=saved, expected=expected, rtol=rtol, atol=atol)

    def array(self, saved, expected, check, rtol=COMPARE_RTOL, atol=COMPARE_ATOL):
        arr = np.asarray(saved, dtype=float)
        expected = np.asarray(expected, dtype=float)
        good = arr.shape == expected.shape and np.allclose(arr, expected, rtol=rtol, atol=atol)
        self.require(good, check, saved_shape=arr.shape, expected_shape=expected.shape,
                     maximum_absolute_error=(float(np.max(np.abs(arr - expected)))
                                             if arr.shape == expected.shape and arr.size else None))


def norm(value) -> float:
    return float(np.linalg.norm(value))


def rank_abs(X):
    return int(np.count_nonzero(np.linalg.svd(X, compute_uv=False) > FROZEN_CONFIG["rank_atol"]))


def recovery_diagnostics(X, y, selected, weights, fitted, full_coef, full_residual):
    # Saved weights are already normalized by engine.evaluate. Their sum is
    # audited separately: another division changes the recorded floating-point
    # problem and can perturb nearly singular directions or near-zero errors.
    a = np.asarray(weights, float)
    S = np.asarray(selected, int)
    delta = fitted - full_coef
    C_selected = full_residual[S, None] * X[S]
    g = X[S].T @ (a * full_residual[S])
    scale = max(1.0, float(np.linalg.norm(C_selected, ord=2)))
    singular = np.linalg.svd(X[S] * np.sqrt(a)[:, None], compute_uv=False)
    pos = a > FROZEN_CONFIG["support_tol"]
    support_rank = rank_abs(X[S[pos]]) if np.any(pos) else 0
    coefficient_relative = norm(delta) / (1 + norm(full_coef))
    cancellation_relative = norm(g) / scale
    residual = X @ fitted - y  # Raw residual reconstruction for EVERY fit.
    loss = float(residual @ residual)
    full_loss = float(full_residual @ full_residual)
    ratio = loss / full_loss if full_loss > 0 else None
    prediction_delta = X @ delta
    quadratic = float(prediction_delta @ prediction_delta)
    cross_term = float(2 * (full_residual @ prediction_delta))
    return {
        "ratio": ratio, "loss": loss, "full_loss": full_loss,
        "coefficient_error": norm(delta),
        "coefficient_error_relative": coefficient_relative,
        "residual_cancellation_error": norm(g),
        "residual_cancellation_scaled": cancellation_relative,
        "support_size": int(pos.sum()), "selected_count": len(S),
        "support_rank": support_rank,
        "weighted_solver_rank": int(np.count_nonzero(singular > FROZEN_CONFIG["solver_rcond"] * singular[0])),
        "weighted_condition": float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf"),
        "near_one": ratio is not None and abs(ratio - 1) <= NEAR_ONE_TOL,
        "coefficient_recovery": coefficient_relative <= FROZEN_CONFIG["coefficient_tol"],
        "numerical_certificate": (support_rank == X.shape[1]
                                  and cancellation_relative <= FROZEN_CONFIG["lp_tol"]
                                  and coefficient_relative <= FROZEN_CONFIG["coefficient_tol"]),
        "min_weight": float(a.min()), "max_weight": float(a.max()),
        "loss_identity_error": abs((loss - full_loss) - quadratic),
        "loss_decomposition_quadratic": quadratic,
        "loss_decomposition_cross_term": cross_term,
        "loss_decomposition_roundoff": abs((loss - full_loss) - quadratic - cross_term),
        "train_prediction_difference_l2": norm(prediction_delta),
    }, residual, delta


def dataset_arrays(root, task, audit):
    path = root / "data/prepared" / f"{task}.npz"
    with np.load(path, allow_pickle=False) as data:
        required = ("X", "y", "Xtest", "ytest", "train_indices", "test_indices",
                    "x_mean", "x_scale", "y_mean", "y_scale")
        for key in required:
            audit.require(key in data.files, f"prepared_key:{key}")
        if any(key not in data.files for key in required):
            raise ValueError("Missing prepared-data keys")
        arrays = {key: data[key].copy() for key in required}
    X, y, Xt, yt = (arrays[k] for k in ("X", "y", "Xtest", "ytest"))
    audit.require(X.ndim == 2 and X.size > 0, "train_shape")
    audit.require(Xt.ndim == 2 and Xt.shape[1] == X.shape[1], "test_shape")
    audit.require(y.shape == (len(X),) and yt.shape == (len(Xt),), "target_shapes")
    for key in ("X", "y", "Xtest", "ytest", "x_mean", "x_scale", "y_mean", "y_scale"):
        audit.require(np.isfinite(arrays[key]).all(), f"finite_prepared:{key}")
    audit.require(np.all(arrays["x_scale"] > 0) and float(arrays["y_scale"].item()) > 0, "positive_scales")
    audit.require(arrays["x_mean"].shape == (X.shape[1] - 1,), "predictor_mean_shape")
    audit.require(arrays["x_scale"].shape == (X.shape[1] - 1,), "predictor_scale_shape")
    for key, rows in (("train_indices", len(X)), ("test_indices", len(Xt))):
        index = arrays[key]
        audit.require(index.shape == (rows,), f"{key}_shape")
        audit.require(index.dtype.kind in "iu" and np.all(index >= 0), f"{key}_integer_nonnegative")
        audit.require(len(np.unique(index)) == rows, f"{key}_unique")
    audit.require(np.intersect1d(arrays["train_indices"], arrays["test_indices"]).size == 0,
                  "train_test_disjoint")
    # The actual saved preparation metadata places its explicit intercept first.
    audit.require(np.all(X[:, 0] == 1) and np.all(Xt[:, 0] == 1), "explicit_intercept_first")
    audit.array(X[:, 1:].mean(axis=0), np.zeros(X.shape[1] - 1), "train_predictor_centering", atol=1e-9)
    audit.array(X[:, 1:].std(axis=0, ddof=0), np.ones(X.shape[1] - 1), "train_predictor_population_scaling", atol=1e-9)
    audit.scalar(float(y.mean()), 0.0, "train_target_centering", atol=1e-9)
    audit.scalar(float(y.std(ddof=0)), 1.0, "train_target_population_scaling", atol=1e-9)
    return arrays


def audit_original_scaling(root, task, arrays, audit):
    """Reload originals independently; no preparation-module/scaler imports."""
    if task == "gas_co":
        rows = []
        ntrain = 0
        for year in range(2011, 2016):
            path = root / "data/raw/gas" / f"gt_{year}.csv"
            with path.open(newline="") as stream:
                header = next(csv.reader(stream))
            audit.require(header == ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO", "NOX"],
                          f"original_gas_header:{year}")
            value = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
            rows.append(value)
            if year <= 2013:
                ntrain += len(value)
        original = np.concatenate(rows)
        x, y = original[:, :9], original[:, 9]
        audit.require(len(original) == 36733, "original_gas_row_count")
        audit.array(arrays["train_indices"], np.arange(ntrain), "original_gas_train_years", rtol=0, atol=0)
        audit.array(arrays["test_indices"], np.arange(ntrain, len(original)), "original_gas_test_years", rtol=0, atol=0)
    elif task == "superconductivity":
        original = np.loadtxt(root / "data/raw/superconductivity/train.csv", delimiter=",", skiprows=1, dtype=np.float64)
        composition = np.loadtxt(root / "data/raw/superconductivity/unique_m.csv", delimiter=",", skiprows=1,
                                 usecols=range(87), dtype=np.float64)
        x, y = original[:, :-1], original[:, -1]
        audit.require(original.shape == (21263, 82) and composition.shape == (21263, 87), "original_super_shapes")
        audit.array(composition[:, -1], y, "super_cross_file_exact_target_alignment", rtol=0, atol=0)
        audit.array(np.count_nonzero(composition[:, :-1], axis=1), x[:, 0],
                    "super_cross_file_element_count_alignment", rtol=0, atol=0)
        tr, te = arrays["train_indices"], arrays["test_indices"]
        parent = np.arange(len(x))
        def find(row):
            while parent[row] != row:
                parent[row] = parent[parent[row]]
                row = int(parent[row])
            return row
        def join(a, b):
            ra, rb = find(a), find(b)
            parent[max(ra, rb)] = min(ra, rb)
        for name, features in (("composition", composition[:, :-1]), ("predictors", x)):
            _, membership = np.unique(features, axis=0, return_inverse=True)
            audit.require(np.intersect1d(membership[tr], membership[te]).size == 0,
                          f"super_no_cross_split_exact_{name}_group")
            first = {}
            for row, group in enumerate(membership):
                if int(group) in first:
                    join(first[int(group)], row)
                else:
                    first[int(group)] = row
        component = np.array([find(row) for row in range(len(x))], dtype=np.int64)
        groups = np.unique(component)
        shuffled = np.random.RandomState(0).permutation(len(groups))
        test_groups = groups[shuffled[:math.ceil(0.2 * len(groups))]]
        expected_test = np.flatnonzero(np.isin(component, test_groups))
        expected_train = np.flatnonzero(~np.isin(component, test_groups))
        audit.array(tr, expected_train, "super_union_group_shuffle_train_indices", rtol=0, atol=0)
        audit.array(te, expected_test, "super_union_group_shuffle_test_indices", rtol=0, atol=0)
        component_hash = hashlib.sha256(component.astype("<i8", copy=False).tobytes()).hexdigest()
    else:
        raise ValueError(f"Unknown case study: {task}")
    train, test = arrays["train_indices"], arrays["test_indices"]
    x_mean, x_scale = x[train].mean(axis=0), x[train].std(axis=0, ddof=0)
    y_mean, y_scale = float(y[train].mean()), float(y[train].std(ddof=0))
    audit.array(arrays["x_mean"], x_mean, "original_train_only_predictor_mean", rtol=1e-11, atol=1e-10)
    audit.array(arrays["x_scale"], x_scale, "original_train_only_predictor_population_std", rtol=1e-11, atol=1e-10)
    audit.scalar(float(arrays["y_mean"].item()), y_mean, "original_train_only_target_mean", rtol=1e-11, atol=1e-10)
    audit.scalar(float(arrays["y_scale"].item()), y_scale, "original_train_only_target_population_std", rtol=1e-11, atol=1e-10)
    for label, index, design, target in (("train", train, arrays["X"], arrays["y"]),
                                       ("test", test, arrays["Xtest"], arrays["ytest"])):
        audit.array(design[:, 1:], (x[index] - x_mean) / x_scale, f"all_original_{label}_predictor_rows", rtol=1e-10, atol=1e-9)
        audit.array(target, (y[index] - y_mean) / y_scale, f"all_original_{label}_target_rows", rtol=1e-10, atol=1e-9)
    result = {"original_row_reconstruction": "Every gas/superconductivity original row reconstructed with independently computed training moments.",
            "train_only_scalers_verified": not audit.issues,
            "n_original": len(y), "n_train": len(train), "n_test": len(test)}
    if task == "superconductivity":
        result.update(union_components=len(groups), component_minimum_row_label_sha256=component_hash)
    return result




def source_hashes(root, task, audit):
    paths = {
        "engine_sha256": root / "source/engine.py",
        "runner_sha256": root / "source/run_study.py",
        "prepared_npz_sha256": root / "data/prepared" / f"{task}.npz",
        "prepared_metadata_sha256": root / "data/prepared" / f"{task}.json",
        "protocol_sha256": root / "protocol.json",
    }
    out = {}
    for key, path in paths.items():
        audit.require(path.is_file(), f"source_exists:{key}", path=str(path))
        if path.is_file():
            out[key] = sha256(path)
    return out


def verify_provenance(record, hashes, audit, recorded_protocol_sha256=None, recorded_runner_sha256=None):
    saved = record.get("source_hashes", {})
    for key, expected in hashes.items():
        matches = saved.get(key) == expected
        if key == "protocol_sha256" and recorded_protocol_sha256 is not None:
            matches = matches or saved.get(key) == recorded_protocol_sha256
        if key == "runner_sha256" and recorded_runner_sha256 is not None:
            matches = matches or saved.get(key) == recorded_runner_sha256
        audit.require(matches, f"source_hash:{key}", saved=saved.get(key), expected=expected)
    audit.require(record.get("config") == FROZEN_CONFIG, "frozen_top_level_config")


def full_fit(root, task, arrays, hashes, audit, recorded_protocol_sha256=None, recorded_runner_sha256=None):
    X, y, Xt, yt = (arrays[k] for k in ("X", "y", "Xtest", "ytest"))
    w, _, rank, singular = np.linalg.lstsq(X, y, rcond=FROZEN_CONFIG["solver_rcond"])
    qr_w, _, qr_rank, _ = qr_lstsq(X, y, cond=FROZEN_CONFIG["solver_rcond"], lapack_driver="gelsy")
    residual = X @ w - y
    qr_residual = X @ qr_w - y
    test_residual = Xt @ w - yt
    loss = float(residual @ residual)
    y_scale = float(arrays["y_scale"].item())
    stationarity = norm(X.T @ residual)
    stationarity_scaled = stationarity / max(1.0, float(singular[0]) * norm(residual))
    qr_relative = norm(qr_w - w) / (1 + norm(w))
    audit.require(qr_relative <= FROZEN_CONFIG["coefficient_tol"], "full_qr_svd_coefficients", relative_error=qr_relative)
    audit.require(rank == qr_rank, "full_qr_svd_solver_rank", svd_rank=int(rank), qr_rank=int(qr_rank))
    audit.require(stationarity_scaled <= FROZEN_CONFIG["lp_tol"], "full_fit_scaled_stationarity", value=stationarity_scaled)
    audit.scalar(float(qr_residual @ qr_residual), loss, "full_qr_svd_loss")
    saved_path = root / "records/full_models" / f"{task}.json"
    audit.require(saved_path.is_file(), "full_model_record_exists")
    if saved_path.is_file():
        saved = json.loads(saved_path.read_text())
        audit.require(saved.get("status") == "complete", "full_model_complete")
        verify_provenance(saved, hashes, audit, recorded_protocol_sha256, recorded_runner_sha256)
        audit.array(saved.get("coefficients", []), w, "saved_full_coefficients")
        for key, expected in {
            "N": len(X), "d": X.shape[1], "full_loss": loss,
            "full_test_mse": float(np.mean(test_residual ** 2)),
            "train_rmse_original": math.sqrt(loss / len(X)) * y_scale,
            "test_rmse_original": math.sqrt(float(np.mean(test_residual ** 2))) * y_scale,
            "rank_solver": int(rank),
            "condition_X": float(singular[0] / singular[-1]) if singular[-1] else float("inf"),
        }.items():
            audit.scalar(saved.get(key), expected, f"saved_full:{key}")
    cache_path = root / "work/cache" / f"{task}.npz"
    audit.require(cache_path.is_file(), "full_cache_exists")
    if cache_path.is_file():
        if saved_path.is_file():
            audit.require(saved.get("cache_sha256") == sha256(cache_path), "saved_full_cache_hash")
        with np.load(cache_path, allow_pickle=False) as cache:
            for key, expected in (("w", w), ("r", residual), ("singular", singular)):
                audit.array(cache[key], expected, f"saved_full_cache:{key}")
            cached_C = cache["C"]
            audit.require(cached_C.shape == X.shape, "saved_full_cache:C_shape")
            if cached_C.shape == X.shape:
                for start in range(0, len(X), 10000):
                    end = min(start + 10000, len(X))
                    audit.array(cached_C[start:end], residual[start:end, None] * X[start:end],
                                f"saved_full_cache:C_rows_{start}_{end}")
    summary = {
        "N": len(X), "d": X.shape[1], "full_loss": loss,
        "train_rmse_original": math.sqrt(loss / len(X)) * y_scale,
        "test_rmse_original": math.sqrt(float(np.mean(test_residual ** 2))) * y_scale,
        "rank_svd": int(rank), "rank_qr": int(qr_rank),
        "rank_absolute": int(np.count_nonzero(singular > FROZEN_CONFIG["rank_atol"])),
        "qr_svd_coefficient_error_relative": qr_relative,
        "stationarity": stationarity, "stationarity_scaled": stationarity_scaled,
        "ratio_denominator_positive": loss > 0, "ratio_scientifically_undefined": loss <= 0,
    }
    return {"w": w, "qr_w": qr_w, "r": residual, "qr_r": qr_residual,
            "full_loss": loss, "stationarity": stationarity, "summary": summary}


def verify_run(record, arrays, full, hashes, audit, check_metrics=True, recorded_protocol_sha256=None, recorded_runner_sha256=None):
    verify_provenance(record, hashes, audit, recorded_protocol_sha256, recorded_runner_sha256)
    result = record.get("engine_result", {})
    audit.require(result.get("config") == FROZEN_CONFIG, "frozen_engine_result_config")
    X, y, Xt, yt = (arrays[k] for k in ("X", "y", "Xtest", "ytest"))
    selected_raw = np.asarray(result.get("indices", []))
    weights = np.asarray(result.get("weights", []), dtype=float)
    n = record.get("n")
    audit.require(isinstance(n, int) and X.shape[1] <= n <= len(X), "valid_frozen_budget", n=n)
    audit.require(selected_raw.ndim == 1 and selected_raw.size > 0, "nonempty_one_dimensional_indices")
    audit.require(selected_raw.dtype.kind in "iu", "indices_integer")
    if selected_raw.ndim != 1 or selected_raw.size == 0 or selected_raw.dtype.kind not in "iu":
        return {"verified": False}
    S = selected_raw.astype(np.int64)
    audit.require(len(np.unique(S)) == len(S), "selected_indices_unique")
    audit.require(np.all((S >= 0) & (S < len(X))), "selected_indices_train_only")
    audit.require(weights.shape == S.shape, "weights_match_indices")
    audit.require(np.isfinite(weights).all() and np.all(weights > 0), "weights_finite_strictly_positive")
    if (weights.shape != S.shape or not np.isfinite(weights).all()
            or np.any(weights <= 0) or np.any((S < 0) | (S >= len(X)))):
        return {"verified": False}
    audit.scalar(float(weights.sum()), 1.0, "saved_weights_sum_one", rtol=0, atol=1e-12)
    a = weights
    audit.require(len(S) <= n and np.count_nonzero(a > FROZEN_CONFIG["support_tol"]) <= n,
                  "selected_and_positive_support_within_budget", selected=len(S), positive=int(np.count_nonzero(a > 1e-9)), n=n)
    if record.get("protocol") == "common":
        original = result.get("original_support", [])
        audit.require(len(set(original)) == len(original), "common_original_support_unique")
        audit.require(set(S.tolist()).issubset(set(original)), "common_reweighting_preserves_support_subset")
    z = np.sqrt(a / a.max())
    A, b = X[S] * z[:, None], y[S] * z
    fitted, _, fit_rank, _ = np.linalg.lstsq(A, b, rcond=FROZEN_CONFIG["solver_rcond"])
    refit_expected, refit_residual, refit_delta = recovery_diagnostics(X, y, S, a, fitted, full["w"], full["r"])
    saved_coefficients = np.asarray(record.get("model", {}).get("coefficients", []), dtype=float)
    audit.array(saved_coefficients, fitted, "saved_selected_coefficients")
    audit.require(saved_coefficients.shape == fitted.shape and np.isfinite(saved_coefficients).all(),
                  "saved_selected_coefficients_valid")
    if saved_coefficients.shape != fitted.shape or not np.isfinite(saved_coefficients).all():
        return {"verified": False}
    # First audit the arithmetic claimed for the stored coefficient vector;
    # independently audit its reconstruction and recovery decisions as well.
    expected, residual, delta = recovery_diagnostics(X, y, S, a, saved_coefficients, full["w"], full["r"])
    for key in ("near_one", "coefficient_recovery", "numerical_certificate"):
        if key == "near_one" and expected["ratio"] is None:
            continue  # The engine's zero-denominator convention is not a scientific ratio.
        audit.require(result.get(key) is expected[key], f"saved_decision:{key}", saved=result.get(key), expected=expected[key])
        audit.require(result.get(key) is refit_expected[key], f"independent_refit_decision:{key}",
                      saved=result.get(key), independently_refitted=refit_expected[key])
    for key in ("ratio", "loss", "full_loss", "coefficient_error", "coefficient_error_relative",
                "residual_cancellation_error", "residual_cancellation_scaled", "support_size",
                "selected_count", "support_rank", "weighted_solver_rank", "weighted_condition",
                "min_weight", "max_weight", "loss_identity_error"):
        if key == "ratio" and expected["ratio"] is None:
            continue
        atol = (1e-10 * max(1.0, expected["loss"], expected["full_loss"])
                if key == "loss_identity_error" else COMPARE_ATOL)
        audit.scalar(result.get(key), expected[key], f"saved_engine:{key}", atol=atol)
    audit.scalar(result.get("full_stationarity"), full["stationarity"], "saved_engine:full_stationarity")
    for key, expected_value in (("N", len(X)), ("d", X.shape[1]), ("rank_detected", full["summary"]["rank_absolute"]),
                                ("rank_solver", full["summary"]["rank_svd"])):
        audit.scalar(result.get(key), expected_value, f"saved_engine:{key}")
    audit.require(record.get("branch") == result.get("branch"), "branch_consistency")
    audit.require(record.get("method") == result.get("method"), "method_consistency")
    audit.require(result.get("seed") == record.get("seed"), "seed_consistency")
    if result.get("branch") == "lp":
        audit.require(refit_expected["numerical_certificate"], "lp_branch_independent_certificate")
    y_scale = float(arrays["y_scale"].item())
    test_residual = Xt @ saved_coefficients - yt
    test_delta = Xt @ delta
    test_mse = float(np.mean(test_residual ** 2))
    full_test_mse = float(np.mean((Xt @ full["w"] - yt) ** 2))
    audit.scalar(result.get("test_mse"), test_mse, "saved_engine:test_mse")
    audit.scalar(result.get("test_ratio"), test_mse / full_test_mse if full_test_mse > 0 else None, "saved_engine:test_ratio")
    audit.scalar(result.get("test_prediction_error"), norm(test_delta), "saved_engine:test_prediction_error")
    metric_expected = {
        "train_rmse_original": math.sqrt(expected["loss"] / len(X)) * y_scale,
        "test_rmse_original": math.sqrt(test_mse) * y_scale,
        "train_prediction_difference_l2_original": norm(X @ delta) * y_scale,
        "test_prediction_difference_l2_original": norm(test_delta) * y_scale,
        "train_prediction_difference_rmse_original": norm(X @ delta) / math.sqrt(len(X)) * y_scale,
        "test_prediction_difference_rmse_original": norm(test_delta) / math.sqrt(len(Xt)) * y_scale,
    }
    if check_metrics:
        metrics = record.get("metrics", {})
        for key, value in metric_expected.items():
            audit.scalar(metrics.get(key), value, f"saved_metrics:{key}")
        audit.require(metrics.get("ratio_denominator_positive") is (full["full_loss"] > 0), "ratio_denominator_flag")
        audit.require(metrics.get("ratio_undefined") is (full["full_loss"] <= 0), "ratio_undefined_flag")
    scale = max(1.0, expected["loss"], expected["full_loss"], expected["loss_decomposition_quadratic"])
    audit.require(expected["loss_decomposition_roundoff"] <= 1e-10 * scale,
                  "raw_loss_decomposition", value=expected["loss_decomposition_roundoff"], scale=scale)
    audit.require(refit_expected["loss_decomposition_roundoff"] <= 1e-10 * scale,
                  "independent_refit_loss_decomposition", value=refit_expected["loss_decomposition_roundoff"], scale=scale)
    audit.scalar(expected["loss"], refit_expected["loss"], "saved_and_refit_training_loss")
    if expected["ratio"] is not None:
        audit.require(expected["ratio"] >= 1 - NEAR_ONE_TOL, "training_loss_not_below_full_minimum", ratio=expected["ratio"])
    claims_recovery = any(result.get(key) is True for key in ("near_one", "coefficient_recovery", "numerical_certificate"))
    qr_check = (record.get("method") == "SpanCancel" or record.get("protocol") == "common") and claims_recovery
    qr_summary = None
    if qr_check:
        qr_fit, _, qr_rank, _ = qr_lstsq(A, b, cond=FROZEN_CONFIG["solver_rcond"], lapack_driver="gelsy")
        qr_diagnostics, _, _ = recovery_diagnostics(X, y, S, a, qr_fit, full["qr_w"], full["qr_r"])
        qr_svd_relative = norm(qr_fit - fitted) / (1 + norm(fitted))
        audit.require(qr_svd_relative <= FROZEN_CONFIG["coefficient_tol"], "selected_qr_svd_coefficients", relative_error=qr_svd_relative)
        audit.require(int(fit_rank) == int(qr_rank), "selected_qr_svd_solver_rank", svd_rank=int(fit_rank), qr_rank=int(qr_rank))
        for key in ("coefficient_recovery", "numerical_certificate", "near_one"):
            if result.get(key) is True and qr_diagnostics["ratio"] is not None:
                audit.require(qr_diagnostics[key], f"qr_independent_recovery_claim:{key}",
                              coefficient_error_relative=qr_diagnostics["coefficient_error_relative"],
                              residual_cancellation_scaled=qr_diagnostics["residual_cancellation_scaled"],
                              ratio=qr_diagnostics["ratio"])
        qr_summary = {"rank": int(qr_rank), "svd_coefficient_error_relative": qr_svd_relative,
                      "coefficient_error_relative": qr_diagnostics["coefficient_error_relative"],
                      "residual_cancellation_scaled": qr_diagnostics["residual_cancellation_scaled"],
                      "numerical_certificate": qr_diagnostics["numerical_certificate"], "ratio": qr_diagnostics["ratio"]}
    return {"verified": not audit.issues, "recomputed": refit_expected,
            "saved_coefficient_arithmetic": expected,
            "independent_refit_comparison": {
                "coefficient_difference_l2": norm(fitted - saved_coefficients),
                "coefficient_difference_relative": norm(fitted - saved_coefficients) / (1 + norm(saved_coefficients)),
                "coefficients_bitwise_equal": bool(np.array_equal(fitted, saved_coefficients)),
                "training_loss_absolute_difference": abs(refit_expected["loss"] - expected["loss"]),
                "training_residual_difference_l2": norm(refit_residual - residual),
            },
            "metrics_original": metric_expected, "qr_cross_check": qr_summary,
            "ratio_scientifically_undefined": expected["ratio"] is None}


def verify_timing_metadata(record, audit):
    timing = record.get("timing", {})
    result = record.get("engine_result", {})
    prep = timing.get("preparation", {})
    keys = ("final_selected_fit_s", "final_evaluation_s", "shared_final_evaluation_without_fit_s",
            "native_method_s", "native_support_construction_s", "engine_call_wall_s")
    for key in keys:
        value = timing.get(key)
        audit.require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                      f"finite_nonnegative_timing:{key}", saved=value)
    if any(not isinstance(timing.get(key), (int, float)) for key in keys):
        return
    audit.scalar(timing["shared_final_evaluation_without_fit_s"],
                 max(0.0, timing["final_evaluation_s"] - timing["final_selected_fit_s"]),
                 "final_evaluation_cost_decomposition", rtol=1e-10, atol=1e-10)
    if record.get("protocol") == "common":
        support = record.get("support_construction", {}).get("engine_result", {})
        cost = support.get("candidate_s", float("nan"))
        expected = prep.get("native_precomputation_s", float("nan")) + cost + result.get("reweight_s", float("nan")) - timing["shared_final_evaluation_without_fit_s"]
        audit.scalar(timing["native_support_construction_s"], cost, "common_support_cost")
    else:
        expected = prep.get("native_precomputation_s", float("nan")) + result.get("selection_s", float("nan")) - timing["shared_final_evaluation_without_fit_s"]
    audit.scalar(timing["native_method_s"], expected, "native_timing_formula", rtol=1e-10, atol=1e-10)
    audit.scalar(timing.get("raw_engine_total_s"), result.get("total_s", float("nan")), "raw_engine_timing_preserved")


def expected_seed_fields(case):
    method = case["method"]
    group = (f"{method}-common" if case["protocol"] == "common"
             else "leverage-paired" if method in ("leverage", "leverage-w") else method)
    components = [SEED_NAMESPACE, case["task"], case["n"], group, case["rep"]]
    digest = hashlib.sha256("|".join(map(str, components)).encode("utf-8")).digest()
    return {"seed": int.from_bytes(digest[:4], byteorder="little"),
            "seed_namespace": SEED_NAMESPACE, "seed_group": group,
            "seed_components": components}


def verify_seed_metadata(record, audit):
    required = ("task", "n", "method", "protocol", "rep")
    audit.require(all(key in record for key in required), "seed_identity_fields_present")
    if any(key not in record for key in required):
        return
    expected = expected_seed_fields(record)
    audit.require(record.get("seed") == expected["seed"], "independent_sha256_seed",
                  saved=record.get("seed"), expected=expected["seed"])
    if record.get("status") == "complete" or "engine_result" in record:
        for key in ("seed_namespace", "seed_group", "seed_components"):
            audit.require(record.get(key) == expected[key], f"independent_seed_metadata:{key}",
                          saved=record.get(key), expected=expected[key])
    if "engine_result" in record:
        audit.require(record["engine_result"].get("seed") == expected["seed"],
                      "independent_engine_result_seed", saved=record["engine_result"].get("seed"), expected=expected["seed"])
    if record.get("kind") == "timing":
        audit.require(record.get("rep") == 0, "deterministic_timing_seed_rep_zero", saved=record.get("rep"))


def verify_paired_leverage_supports(expected, records, audit):
    """Compare saved ordered supports, preserving every incomplete pair."""
    observed = {path.stem: record for path, record in records}
    pairs = []
    for key, case in expected.items():
        if case["kind"] != "science" or case["protocol"] != "native" or case["method"] != "leverage":
            continue
        partner_case = {**case, "method": "leverage-w"}
        partner_key = case_key(partner_case)
        left, right = observed.get(key), observed.get(partner_key)
        pair = {"task": case["task"], "n": case["n"], "rep": case["rep"],
                "leverage_record": key, "leverage_weighted_record": partner_key,
                "both_completed": left is not None and right is not None
                                  and left.get("status") == right.get("status") == "complete"}
        if pair["both_completed"]:
            left_support = left.get("engine_result", {}).get("indices")
            right_support = right.get("engine_result", {}).get("indices")
            pair["ordered_supports_identical"] = (isinstance(left_support, list)
                                                  and isinstance(right_support, list)
                                                  and left_support == right_support)
            pair["seeds_identical"] = left.get("seed") == right.get("seed")
            audit.require(pair["ordered_supports_identical"], "paired_leverage_saved_supports_identical", **pair)
            audit.require(pair["seeds_identical"], "paired_leverage_seeds_identical", **pair)
        else:
            pair.update(ordered_supports_identical=None, seeds_identical=None,
                        leverage_status=None if left is None else left.get("status"),
                        leverage_weighted_status=None if right is None else right.get("status"))
        pairs.append(pair)
    return {"expected_pairs": len(pairs), "completed_pairs": sum(p["both_completed"] for p in pairs),
            "pairs_with_identical_supports": sum(p["ordered_supports_identical"] is True for p in pairs),
            "pairs_with_identical_seeds": sum(p["seeds_identical"] is True for p in pairs), "pairs": pairs}


def case_key(case):
    method = case["method"].lower().replace("-", "_")
    return (f'{case["task"]}__n{case["n"]}__{case["protocol"]}__{method}'
            f'__rep{case["rep"]:02d}__{case["kind"]}{case["trial"]:02d}')


def independent_expected_cases(protocol, shapes, records):
    observed = {path.stem: record for path, record in records}
    out = {}
    for task, shape in shapes.items():
        if "d" not in shape:
            continue
        d = shape["d"]
        primary = math.ceil(1.5 * d)
        for n in dict.fromkeys([d, math.ceil(1.25 * d), primary, 2 * d]):
            settings = [(method, "native") for method in protocol["methods"]]
            if n == primary:
                settings.append(("D-optimal", "common"))
            for method, weighting in settings:
                reps = range(10 if method in ("uniform", "leverage", "leverage-w") else 1)
                for rep in reps:
                    case = {"task": task, "n": n, "method": method, "protocol": weighting,
                            "rep": rep, "kind": "science", "trial": 0}
                    out[case_key(case)] = case
                first = {"task": task, "n": n, "method": method, "protocol": weighting,
                         "rep": 0, "kind": "science", "trial": 0}
                record = observed.get(case_key(first), {})
                if record.get("status") != "complete":
                    continue
                branch = record.get("branch")
                if (method == "SpanCancel" or weighting == "common") and branch == "heuristic":
                    for rep in range(1, 10):
                        case = {**first, "rep": rep}
                        out[case_key(case)] = case
                deterministic = ((method == "D-optimal" and weighting == "native")
                                 or branch in ("lp", "square_interpolation"))
                if n == primary and deterministic:
                    for trial in (1, 2):
                        case = {**first, "kind": "timing", "trial": trial}
                        out[case_key(case)] = case
    for case in out.values():
        case["seed"] = expected_seed_fields(case)["seed"]
    return out


def self_test():
    # Centered orthogonal residual has nonzero L, and this full support exactly
    # recovers the full fit. Tampering tests exercise scientific failure checks.
    X = np.array([[-2., 1.], [-1., 1.], [0., 1.], [1., 1.], [2., 1.]])
    r = np.array([1., -2., 2., -2., 1.])
    y = X @ np.array([0.4, 0.7]) - r
    w = np.linalg.lstsq(X, y, rcond=1e-12)[0]
    q = qr_lstsq(X, y, cond=1e-12, lapack_driver="gelsy")[0]
    diagnostic, _, _ = recovery_diagnostics(X, y, np.arange(5), np.ones(5) / 5,
                                           w, w, X @ w - y)
    assert diagnostic["numerical_certificate"] and diagnostic["near_one"]
    assert norm(w - q) / (1 + norm(w)) <= 1e-7
    bad_coef = w + np.array([0.01, 0])
    bad, _, _ = recovery_diagnostics(X, y, np.arange(5), np.ones(5) / 5,
                                    bad_coef, w, X @ w - y)
    assert not bad["coefficient_recovery"] and not bad["near_one"]
    assert bad["loss_decomposition_roundoff"] < 1e-12
    audit = Audit("intentional_tamper")
    audit.scalar(2.0, diagnostic["ratio"], "ratio_tamper")
    audit.array(w + 1., w, "coefficient_tamper")
    assert len(audit.issues) == 2
    zero, _, _ = recovery_diagnostics(X, X @ w, np.arange(5), np.ones(5) / 5,
                                     w, w, np.zeros(5))
    assert zero["ratio"] is None and not zero["near_one"]
    leverage_case = {"task": "gas_co", "n": 15, "method": "leverage", "protocol": "native",
                     "rep": 0, "kind": "science", "trial": 0}
    leverage_seed = expected_seed_fields(leverage_case)
    assert leverage_seed["seed"] == 3568334261
    assert expected_seed_fields({**leverage_case, "method": "leverage-w"}) == leverage_seed
    assert expected_seed_fields({**leverage_case, "task": "superconductivity", "n": 123, "method": "D-optimal",
                                 "protocol": "common", "rep": 9})["seed"] == 723920183
    seed_audit = Audit("seed_tamper")
    verify_seed_metadata({**leverage_case, **leverage_seed, "seed": leverage_seed["seed"] + 1,
                          "status": "complete", "engine_result": {"seed": leverage_seed["seed"]}}, seed_audit)
    assert len(seed_audit.issues) == 1 and seed_audit.issues[0]["check"] == "independent_sha256_seed"
    left = {**leverage_case, **leverage_seed, "status": "complete", "engine_result": {"indices": [2, 5, 7]}}
    right = {**left, "method": "leverage-w", "engine_result": {"indices": [2, 5, 7]}}
    expected_pair = {case_key(leverage_case): leverage_case}
    pair_audit = Audit("paired_support")
    paired = verify_paired_leverage_supports(expected_pair,
                [(Path(case_key(left) + ".json"), left), (Path(case_key(right) + ".json"), right)], pair_audit)
    assert not pair_audit.issues and paired["pairs_with_identical_supports"] == 1
    right["engine_result"]["indices"] = [5, 2, 7]
    verify_paired_leverage_supports(expected_pair,
                [(Path(case_key(left) + ".json"), left), (Path(case_key(right) + ".json"), right)], pair_audit)
    assert len(pair_audit.issues) == 1
    print(json.dumps({"self_test": "passed", "cases": ["SVD/QR exact recovery", "false recovery rejection",
                            "tampered saved values", "loss decomposition", "undefined ratio",
                            "frozen SHA256 seed vectors", "seed tamper detection", "paired support tamper detection"]}))


def recorded_source_hash(root, protocol, audit, source):
    path = root / "provenance.json"
    if not path.is_file():
        return None
    provenance = json.loads(path.read_text())
    original = provenance.get(f"recorded_{source}_sha256", "")
    source_path = root / {"protocol": "protocol.json", "runner": "source/run_study.py"}[source]
    valid = (provenance.get(f"package_{source}_sha256") == sha256(source_path)
             and provenance.get("tasks") == protocol.get("tasks")
             and len(original) == 64
             and all(c in "0123456789abcdef" for c in original))
    audit.require(valid, f"package_{source}_provenance")
    return original if valid else None


def run_audit(root):
    began = time.perf_counter()
    overall = Audit("study")
    protocol = json.loads((root / "protocol.json").read_text())
    recorded_protocol_sha256 = recorded_source_hash(root, protocol, overall, "protocol")
    recorded_runner_sha256 = recorded_source_hash(root, protocol, overall, "runner")
    overall.require(protocol.get("engine_sha256") == FROZEN_ENGINE_SHA256, "frozen_protocol_engine_hash")
    overall.require(sha256(root / "source/engine.py") == FROZEN_ENGINE_SHA256, "unchanged_engine_source")
    overall.require(protocol.get("config") == FROZEN_CONFIG, "unchanged_scientific_tolerances")
    pools = threadpool_info()
    overall.require(all(p.get("num_threads") == 1 for p in pools), "single_numerical_thread", pools=pools)
    report = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "independence": "No engine or runner imports; saved selected fits reconstructed from raw residuals; pivoted-QR gelsy cross-checks.",
              "comparison_tolerances": {"relative": COMPARE_RTOL, "absolute": COMPARE_ATOL},
              "scientific_thresholds": {**FROZEN_CONFIG, "near_one_tol": NEAR_ONE_TOL},
              "software": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "threadpools": pools},
              "tasks": {}, "runs": {}, "support_construction_fits": {},
              "noncompleted_records": [], "timing_records": [], "issues": overall.issues}
    records = []
    for path in sorted((root / "records/runs").glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (ValueError, OSError) as error:
            overall.require(False, "unreadable_run_record", file=path.name, error=str(error))
            continue
        records.append((path, rec))
    ids = collections.Counter((r.get("kind"), r.get("task"), r.get("n"), r.get("method"), r.get("protocol"),
                               r.get("rep"), r.get("trial")) for _, r in records)
    for ident, count in ids.items():
        overall.require(count == 1, "unique_run_identity", identity=ident, count=count)
    for path, record in records:
        seed_audit = Audit(path.name)
        verify_seed_metadata(record, seed_audit)
        overall.issues.extend(seed_audit.issues)
    record_hashes = {str(path): sha256(path) for path, _ in records}
    preflight_path = root / "analysis/preflight_verification.json"
    preflight = json.loads(preflight_path.read_text()) if preflight_path.is_file() else {}
    for task in protocol.get("tasks", []):
        task_audit = Audit(task)
        try:
            arrays = dataset_arrays(root, task, task_audit)
            hashes = source_hashes(root, task, task_audit)
            prepared_audit = preflight.get(task, {})
            if prepared_audit:
                task_audit.require(not prepared_audit.get("issues", ["missing preflight"]), "prior_independent_raw_preflight_passed")
                task_audit.require(prepared_audit.get("npz_sha256") == hashes.get("prepared_npz_sha256")
                                   and prepared_audit.get("metadata_sha256") == hashes.get("prepared_metadata_sha256"),
                                   "prior_independent_raw_preflight_inputs_unchanged")
            else:
                audit_original_scaling(root, task, arrays, task_audit)
            full = full_fit(root, task, arrays, hashes, task_audit, recorded_protocol_sha256, recorded_runner_sha256)
        except Exception as error:
            task_audit.require(False, "task_verification_exception", exception=type(error).__name__, message=str(error))
            report["tasks"][task] = {"verified": False}
            overall.issues.extend(task_audit.issues)
            continue
        report["tasks"][task] = {**full["summary"], "source_hashes": hashes, "verified": not task_audit.issues}
        overall.issues.extend(task_audit.issues)
        completed = 0
        for path, record in records:
            if record.get("task") != task:
                continue
            if record.get("status") != "complete":
                report["noncompleted_records"].append({"file": path.name, "task": task, "status": record.get("status"),
                                                       "kind": record.get("kind"), "n": record.get("n"),
                                                       "method": record.get("method"), "protocol": record.get("protocol"),
                                                       "rep": record.get("rep"), "failure": record.get("failure", record.get("error"))})
                if "engine_result" not in record or "model" not in record:
                    continue
            if record.get("kind") == "timing":
                # Duplicate timing fits are retained; only scientific fits enter inference.
                timing_audit = Audit(path.name)
                verify_provenance(record, hashes, timing_audit, recorded_protocol_sha256, recorded_runner_sha256)
                verify_timing_metadata(record, timing_audit)
                ref_case = {**record, "kind": "science", "trial": 0}
                reference_path = root / "records/runs" / f"{case_key(ref_case)}.json"
                timing_audit.require(reference_path.is_file(), "deterministic_scientific_reference_exists")
                if reference_path.is_file():
                    reference = json.loads(reference_path.read_text())
                    timing_audit.require(record["engine_result"]["indices"] == reference["engine_result"]["indices"], "deterministic_timing_identical_indices")
                    timing_audit.array(record["engine_result"]["weights"], reference["engine_result"]["weights"], "deterministic_timing_identical_weights", rtol=0, atol=0)
                    timing_audit.array(record["model"]["coefficients"], reference["model"]["coefficients"], "deterministic_timing_identical_coefficients")
                report["timing_records"].append({"file": path.name, "task": task, "issues": timing_audit.issues})
                overall.issues.extend(timing_audit.issues)
                continue
            run = Audit(path.name)
            try:
                result = verify_run(record, arrays, full, hashes, run, recorded_protocol_sha256=recorded_protocol_sha256, recorded_runner_sha256=recorded_runner_sha256)
                verify_timing_metadata(record, run)
                result["verified"] = result.get("verified", False) and not run.issues
            except Exception as error:
                run.require(False, "run_verification_exception", exception=type(error).__name__, message=str(error))
                result = {"verified": False}
            report["runs"][path.name] = {"task": task, "n": record.get("n"), "method": record.get("method"),
                                        "protocol": record.get("protocol"), "rep": record.get("rep"), **result,
                                        "issues": run.issues}
            overall.issues.extend(run.issues)
            if "support_construction" in record:
                sub = record["support_construction"]
                selection = sub["engine_result"]
                supporting = {**record, "engine_result": selection, "model": sub["model"], "protocol": "native",
                              "branch": selection["branch"], "rep": 0}
                supporting.update(expected_seed_fields(supporting))
                supporting.pop("support_construction", None)
                support_audit = Audit(path.name + "#support_construction")
                try:
                    verify_seed_metadata(supporting, support_audit)
                    support_result = verify_run(supporting, arrays, full, hashes, support_audit, check_metrics=False, recorded_protocol_sha256=recorded_protocol_sha256, recorded_runner_sha256=recorded_runner_sha256)
                except Exception as error:
                    support_audit.require(False, "support_fit_verification_exception", exception=type(error).__name__, message=str(error))
                    support_result = {"verified": False}
                report["support_construction_fits"][path.name] = {"task": task, **support_result, "issues": support_audit.issues}
                overall.issues.extend(support_audit.issues)
            completed += 1
            if completed % 20 == 0:
                print(json.dumps({"verification_progress": task, "scientific_fits_checked": completed}), flush=True)
        print(json.dumps({"verification_task_complete": task, "scientific_fits_checked": completed,
                          "task_issues": len(task_audit.issues)}), flush=True)
        for key, value in source_hashes(root, task, task_audit).items():
            overall.require(value == hashes.get(key), f"inputs_unchanged_during_audit:{key}", task=task)
    known_tasks = set(protocol.get("tasks", []))
    for path, record in records:
        overall.require(record.get("task") in known_tasks, "known_task_record", file=path.name, task=record.get("task"))
        overall.require(record_hashes[str(path)] == sha256(path), "run_record_unchanged_during_audit", file=path.name)
    expected = independent_expected_cases(protocol, report["tasks"], records)
    report["paired_leverage_supports"] = verify_paired_leverage_supports(expected, records, overall)
    observed_keys = {path.stem for path, _ in records}
    missing_keys = sorted(set(expected) - observed_keys)
    extra_keys = sorted(observed_keys - set(expected))
    overall.require(not missing_keys, "all_expected_grid_records_present", missing=missing_keys)
    overall.require(not extra_keys, "no_unplanned_grid_records", extra=extra_keys)
    grid_path = root / "records/expected_grid.json"
    if grid_path.is_file():
        grid = json.loads(grid_path.read_text())
        grid_keys = [case["run_key"] for case in grid.get("expected_cases", [])]
        overall.require(len(grid_keys) == len(set(grid_keys)), "manifest_grid_unique")
        overall.require(set(grid_keys) == set(expected), "manifest_matches_independent_grid",
                        missing_from_manifest=sorted(set(expected) - set(grid_keys)),
                        extra_in_manifest=sorted(set(grid_keys) - set(expected)))
    else:
        overall.require(False, "expected_grid_manifest_exists")
    for path, record in records:
        if path.stem in expected:
            for key, value in expected[path.stem].items():
                overall.require(record.get(key) == value, f"record_matches_frozen_grid:{key}", file=path.name)
    report["coverage"] = {"expected_records": len(expected), "observed_records": len(records),
                          "missing_keys": missing_keys, "unplanned_keys": extra_keys,
                          "all_expected_records_completed": not missing_keys and all(r.get("status") == "complete" for _, r in records)}
    report["summary"] = {"scientific_fits_reconstructed": len(report["runs"]),
                         "scientific_fits_verified": sum(r.get("verified") is True for r in report["runs"].values()),
                         "support_construction_fits_reconstructed": len(report["support_construction_fits"]),
                         "qr_selected_cross_checks": sum(r.get("qr_cross_check") is not None for r in report["runs"].values()),
                         "noncompleted_records": len(report["noncompleted_records"]),
                         "timing_duplicates_retained": len(report["timing_records"]),
                         "issue_count": len(overall.issues), "numerical_audit_passed": not overall.issues,
                         "study_complete_and_verified": not overall.issues and report["coverage"]["all_expected_records_completed"],
                         "elapsed_seconds": time.perf_counter() - began,
                         "inference": "Descriptive summaries by task; repetitions are stochastic draws within each task."}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.self_test:
            self_test()
            return
        report = run_audit(args.root.resolve())
    output = args.output or args.root / "analysis/verification_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verification_report": str(output), **report["summary"]}, allow_nan=False))
    if report["issues"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
