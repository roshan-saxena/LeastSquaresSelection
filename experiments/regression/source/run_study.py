#!/usr/bin/env python3
"""Serial, resumable large-regression campaign around the unmodified engine.

Default invocation prints a plan. ``--execute`` is the explicit launch switch.
Algorithm results are kept verbatim in ``engine_result``. Timing wrappers observe
the frozen engine without changing inputs, return values, settings, or branches.
"""
from __future__ import annotations

import os

# Set before importing any numerical library, including in launched workers.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import contextlib
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import scipy
import threadpoolctl
import engine

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1
SEED_NAMESPACE = "large_regression_v1"
DETERMINISTIC_BRANCHES = {"lp", "square_interpolation"}
STOCHASTIC_METHODS = {"uniform", "leverage", "leverage-w"}
EXPECTED_VERSIONS = {"numpy": "1.26.4", "scipy": "1.13.1", "threadpoolctl": "3.6.0"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_value(value: Any) -> Any:
    """Strict JSON: nonfinite engine numbers are explicit strings, never clipped."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else ("Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN")
    return value


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(json_value(obj), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def log_event(root: Path, event: str, **details: Any) -> None:
    path = root / "records/events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_value(dict(at_utc=utcnow(), event=event, **details)), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another study process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def protocol_for(root: Path) -> dict:
    protocol = read_json(root / "protocol.json")
    if asdict(engine.Config()) != protocol["config"]:
        raise RuntimeError("Engine defaults differ from protocol.json")
    if sha256(Path(engine.__file__)) != protocol["engine_sha256"]:
        raise RuntimeError("Engine checksum differs from protocol.json")
    limits = protocol["resource_limits"]
    if limits["workers"] != 1 or limits["numerical_threads"] != 1:
        raise RuntimeError("This runner requires one worker and one numerical thread")
    return protocol


def source_hashes(root: Path, task: str) -> dict:
    metadata = root / "data/prepared" / f"{task}.json"
    if not metadata.exists():
        raise FileNotFoundError(f"Prepared metadata is required: {metadata}")
    return dict(
        engine_sha256=sha256(Path(engine.__file__)),
        runner_sha256=sha256(Path(__file__)),
        prepared_npz_sha256=sha256(root / "data/prepared" / f"{task}.npz"),
        prepared_metadata_sha256=sha256(metadata),
        protocol_sha256=sha256(root / "protocol.json"),
    )


def seed_group(method: str, protocol: str = "native") -> str:
    if protocol == "common":
        return f"{method}-common"
    return "leverage-paired" if method in {"leverage", "leverage-w"} else method


def run_seed(task: str, n: int, method: str, rep: int, protocol: str = "native") -> int:
    return engine.seed_for(SEED_NAMESPACE, task, n, seed_group(method, protocol), rep)


def budgets(d: int) -> list[int]:
    # Preserve labels separately even if rounding makes two budgets identical.
    return [d, math.ceil(1.25 * d), math.ceil(1.5 * d), 2 * d]


def stable_key(case: dict) -> str:
    method = case["method"].lower().replace("-", "_")
    return f'{case["task"]}__n{case["n"]}__{case["protocol"]}__{method}__rep{case["rep"]:02d}__{case["kind"]}{case["trial"]:02d}'


def run_path(root: Path, case: dict) -> Path:
    return root / "records/runs" / f"{stable_key(case)}.json"


def make_case(task: str, n: int, method: str, *, protocol: str = "native", rep: int = 0, kind: str = "science", trial: int = 0, primary: bool = False) -> dict:
    return dict(task=task, n=int(n), method=method, protocol=protocol, rep=int(rep), kind=kind, trial=int(trial), primary=bool(primary), seed=run_seed(task, n, method, rep, protocol))


def scientific_base_cases(task: str, d: int, methods: list[str]) -> list[dict]:
    primary = math.ceil(1.5 * d)
    ordered = list(dict.fromkeys([primary] + budgets(d)))
    rows = []
    for n in ordered:
        for method in methods:
            for rep in range(10 if method in STOCHASTIC_METHODS else 1):
                rows.append(make_case(task, n, method, rep=rep, primary=n == primary))
        if n == primary:
            rows.append(make_case(task, n, "D-optimal", protocol="common", primary=True))
    return rows


def expansion_cases(case: dict, result: dict) -> list[dict]:
    """A first fallback commits to all ten fixed seeds, including seed zero."""
    if case["kind"] != "science" or case["rep"] != 0:
        return []
    if case["method"] == "SpanCancel" or case["protocol"] == "common":
        if result.get("status") == "complete" and result.get("branch") == "heuristic":
            return [make_case(case["task"], case["n"], case["method"], protocol=case["protocol"], rep=rep, primary=case["primary"]) for rep in range(1, 10)]
    return []


def deterministic_timing_cases(case: dict, result: dict) -> list[dict]:
    if case["kind"] != "science" or case["rep"] != 0 or not case["primary"] or result.get("status") != "complete":
        return []
    deterministic = case["method"] == "D-optimal" and case["protocol"] == "native" or result.get("branch") in DETERMINISTIC_BRANCHES
    if deterministic:
        return [make_case(case["task"], case["n"], case["method"], protocol=case["protocol"], kind="timing", trial=trial, primary=True) for trial in (1, 2)]
    return []


def load_arrays(root: Path, task: str, *, check: bool = True) -> dict:
    with np.load(root / "data/prepared" / f"{task}.npz", allow_pickle=False) as data:
        required = {"X", "y", "Xtest", "ytest", "train_indices", "test_indices", "x_mean", "x_scale", "y_mean", "y_scale"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Missing prepared-array keys for {task}: {sorted(missing)}")
        arrays = {k: data[k] for k in data.files}
    if check:
        X, y, Xt, yt = (arrays[k] for k in ("X", "y", "Xtest", "ytest"))
        if X.dtype != np.float64 or Xt.dtype != np.float64 or y.dtype != np.float64 or yt.dtype != np.float64:
            raise ValueError("Prepared matrices and targets must be float64")
        if X.ndim != 2 or Xt.ndim != 2 or X.shape[1] != Xt.shape[1] or y.shape != (len(X),) or yt.shape != (len(Xt),):
            raise ValueError("Prepared feature/target dimensions do not match")
        if not all(np.isfinite(arrays[k]).all() for k in ("X", "y", "Xtest", "ytest", "x_mean", "x_scale", "y_mean", "y_scale")):
            raise ValueError("Prepared values must all be finite")
        if arrays["y_scale"].size != 1 or arrays["y_mean"].size != 1:
            raise ValueError("Target mean and scale must each contain one value")
        if float(arrays["y_scale"].item()) <= 0 or np.any(arrays["x_scale"] <= 0):
            raise ValueError("Preparation scales must be positive")
        intercepts = np.flatnonzero(np.all(X == 1.0, axis=0) & np.all(Xt == 1.0, axis=0))
        if len(intercepts) != 1:
            raise ValueError("Exactly one explicit, shared intercept column is required")
        if len(arrays["train_indices"]) != len(X) or len(arrays["test_indices"]) != len(Xt):
            raise ValueError("Saved split indices do not match prepared rows")
        if abs(float(np.mean(y))) > 1e-10 or abs(float(np.std(y)) - 1) > 1e-10:
            raise ValueError("Target is not standardized from training moments")
    return arrays


class EngineTimings:
    """Observe nested engine calls; retain all inner algorithm evaluations.

    Only the final evaluation's fit is native selected-model work. Its remaining
    checks/diagnostics and the post-return rank SVD are shared evaluation costs.
    Internal cancellation/fallback evaluations remain charged to the algorithm,
    because the unchanged algorithm uses them for its checks and saved searches.
    """
    def __init__(self):
        self.events: list[dict] = []
        self.stack: list[str] = []
        self.final_coefficients: np.ndarray | None = None
        self.final_evaluation_complete = False
        self.originals: dict = {}

    def observe(self, name, function):
        def wrapped(*args, **kwargs):
            parents = tuple(self.stack)
            after_final = self.final_evaluation_complete
            self.stack.append(name)
            started = time.perf_counter()
            try:
                result = function(*args, **kwargs)
                if name == "fit" and "evaluate_final" in parents:
                    self.final_coefficients = np.asarray(result).copy()
                return result
            finally:
                elapsed = time.perf_counter() - started
                self.stack.pop()
                self.events.append(dict(name=name, parents=list(parents), seconds=elapsed, after_final_evaluation=after_final))
                if name == "evaluate_final":
                    self.final_evaluation_complete = True
        return wrapped

    def __enter__(self):
        for name in ("fit", "rank", "vol", "cancellation", "fallback", "greedy_cancel"):
            self.originals[name] = getattr(engine, name)
            setattr(engine, name, self.observe(name, self.originals[name]))
        self.originals["evaluate"] = engine.Problem.evaluate
        original_evaluate = engine.Problem.evaluate

        def evaluate(problem, *args, **kwargs):
            name = "evaluate_internal" if any(s in self.stack for s in ("cancellation", "fallback")) else "evaluate_final"
            return self.observe(name, original_evaluate)(problem, *args, **kwargs)

        engine.Problem.evaluate = evaluate
        self.originals["svd"] = np.linalg.svd
        np.linalg.svd = self.observe("svd", np.linalg.svd)
        return self

    def __exit__(self, *_exc):
        for name in ("fit", "rank", "vol", "cancellation", "fallback", "greedy_cancel"):
            setattr(engine, name, self.originals[name])
        engine.Problem.evaluate = self.originals["evaluate"]
        np.linalg.svd = self.originals["svd"]

    def seconds(self, name: str, *, parent: str | None = None, top_level: bool = False) -> float:
        return sum(e["seconds"] for e in self.events if e["name"] == name and (parent is None or parent in e["parents"]) and (not top_level or not e["parents"]))

    def components(self) -> dict:
        final_evaluation = self.seconds("evaluate_final")
        final_fit = self.seconds("fit", parent="evaluate_final")
        return dict(
            final_selected_fit_s=final_fit,
            final_evaluation_s=final_evaluation,
            shared_final_evaluation_without_fit_s=max(0.0, final_evaluation - final_fit),
            post_selection_rank_diagnostic_s=sum(e["seconds"] for e in self.events if e["name"] == "rank" and not e["parents"] and e["after_final_evaluation"]),
            internal_algorithm_evaluation_s=self.seconds("evaluate_internal"),
            svd_calls=[e for e in self.events if e["name"] == "svd"],
            observed_calls=self.events,
        )


def fresh_problem(arrays: dict, cfg: engine.Config) -> tuple[engine.Problem, dict]:
    with EngineTimings() as timer:
        start = time.perf_counter()
        problem = engine.Problem(arrays["X"], arrays["y"], cfg, arrays["Xtest"], arrays["ytest"])
        elapsed = time.perf_counter() - start
    # In Problem.__init__, its sole explicit SVD is the diagnostic condition SVD.
    condition_svd = timer.seconds("svd", top_level=True)
    return problem, dict(
        preparation_wall_s=elapsed,
        full_fit_s=problem.full_fit_s,
        residual_precompute_s=problem.residual_s,
        diagnostic_condition_svd_s=condition_svd,
        preparation_other_s=max(0.0, elapsed - problem.full_fit_s - problem.residual_s - condition_svd),
        native_precomputation_s=problem.full_fit_s + problem.residual_s,
        diagnostic_svd_observed_calls=timer.events,
    )


def cached_problem(root: Path, task: str, arrays: dict, cfg: engine.Config, hashes: dict) -> tuple[engine.Problem, dict]:
    full = read_json(root / "records/full_models" / f"{task}.json")
    if full.get("status") != "complete" or full["source_hashes"] != hashes:
        raise RuntimeError(f"Stale or incomplete full-model record: {task}")
    cache_path = root / "work/cache" / f"{task}.npz"
    if full["cache_sha256"] != sha256(cache_path):
        raise RuntimeError(f"Full-model cache hash mismatch: {task}")
    started = time.perf_counter()
    with np.load(cache_path, allow_pickle=False) as cache:
        values = {k: cache[k] for k in ("w", "r", "C", "singular")}
    P = engine.Problem.__new__(engine.Problem)
    P.X, P.y, P.Xtest, P.ytest = (arrays[k] for k in ("X", "y", "Xtest", "ytest"))
    P.cfg = cfg
    P.N, P.d = P.X.shape
    P.w, P.r, P.C, P.singular = (values[k] for k in ("w", "r", "C", "singular"))
    P.L = full["full_loss"]
    P.full_test_mse = full["full_test_mse"]
    P.condition = float(full["condition_X"])
    P.rank_solver = full["rank_solver"]
    P.full_fit_s = full["timings"]["full_fit_s"]
    P.residual_s = full["timings"]["residual_precompute_s"]
    P.Q = None
    return P, dict(
        cache_load_s=time.perf_counter() - started,
        full_model_record=f"records/full_models/{task}.json",
        inherited_full_fit_s=P.full_fit_s,
        inherited_residual_precompute_s=P.residual_s,
        native_precomputation_s=0.0,
        amortization="The full reference fit/residual cache is shared across baseline runs; inherited engine times are preserved but excluded from native baseline cost.",
    )


def basic_checks(P: engine.Problem, result: dict, coefficients: np.ndarray, n: int) -> dict:
    S, a = np.asarray(result["indices"], int), np.asarray(result["weights"], float)
    loss = float(np.sum((P.X @ coefficients - P.y) ** 2))
    ratio = loss / P.L if P.L > 0 else (1.0 if loss == 0 else float("inf"))
    values = dict(
        selected_count_within_budget=0 < len(S) <= n,
        indices_unique=len(np.unique(S)) == len(S),
        indices_in_range=bool(np.all((S >= 0) & (S < P.N))),
        finite_nonnegative_weights=bool(np.isfinite(a).all() and np.all(a >= 0)),
        normalized_weights=bool(abs(float(a.sum()) - 1) <= 1e-12),
        coefficient_finite=bool(np.isfinite(coefficients).all()),
        loss_matches_saved=bool(np.isclose(loss, result["loss"], rtol=1e-10, atol=1e-10)),
        ratio_matches_saved=bool(np.isclose(ratio, result["ratio"], rtol=1e-10, atol=1e-10)),
        config_unchanged=result["config"] == asdict(P.cfg),
    )
    return dict(passed=all(values.values()), checks=values, reconstructed_loss=loss, reconstructed_ratio=ratio)


def scientific_metrics(P: engine.Problem, coefficients: np.ndarray, y_scale: float) -> dict:
    train_residual = P.X @ coefficients - P.y
    test_residual = P.Xtest @ coefficients - P.ytest
    train_difference = P.X @ (coefficients - P.w)
    test_difference = P.Xtest @ (coefficients - P.w)
    return dict(
        train_rmse_original=float(np.sqrt(np.mean(train_residual ** 2)) * y_scale),
        test_rmse_original=float(np.sqrt(np.mean(test_residual ** 2)) * y_scale),
        train_prediction_difference_l2_original=float(np.linalg.norm(train_difference) * y_scale),
        test_prediction_difference_l2_original=float(np.linalg.norm(test_difference) * y_scale),
        train_prediction_difference_rmse_original=float(np.sqrt(np.mean(train_difference ** 2)) * y_scale),
        test_prediction_difference_rmse_original=float(np.sqrt(np.mean(test_difference ** 2)) * y_scale),
        ratio_denominator_positive=bool(P.L > 0),
        ratio_undefined=bool(P.L <= 0),
    )


def environment_record() -> dict:
    versions = dict(numpy=np.__version__, scipy=scipy.__version__, threadpoolctl=threadpoolctl.__version__)
    if versions != EXPECTED_VERSIONS or sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Wrong numerical environment: Python {platform.python_version()}, {versions}")
    pools = threadpoolctl.threadpool_info()
    if any(p.get("num_threads") != 1 for p in pools):
        raise RuntimeError(f"Numerical runtime exceeded one thread: {pools}")
    return dict(python=platform.python_version(), executable=sys.executable, platform=platform.platform(), versions=versions, threadpools=pools)


def prepare_full_model(root: Path, task: str) -> dict:
    protocol_for(root)
    hashes = source_hashes(root, task)
    arrays = load_arrays(root, task)
    P, timing = fresh_problem(arrays, engine.Config())
    cache_path = root / "work/cache" / f"{task}.npz"
    atomic_npz(cache_path, w=P.w, r=P.r, C=P.C, singular=P.singular)
    record = dict(
        schema_version=SCHEMA_VERSION, status="complete", kind="full_model", task=task,
        saved_at_utc=utcnow(), N=P.N, d=P.d, coefficients=P.w.tolist(),
        full_loss=P.L, full_test_mse=P.full_test_mse,
        train_rmse_original=float(np.sqrt(P.L / P.N) * float(arrays["y_scale"].item())),
        test_rmse_original=float(np.sqrt(P.full_test_mse) * float(arrays["y_scale"].item())),
        full_stationarity=float(np.linalg.norm(P.X.T @ P.r)),
        condition_X=P.condition, rank_solver=P.rank_solver,
        source_hashes=hashes, config=asdict(P.cfg), timings=timing,
        cache_sha256=sha256(cache_path), cache_path=f"work/cache/{task}.npz",
        environment=environment_record(), metadata=read_json(root / "data/prepared" / f"{task}.json"),
        ratio_denominator_positive=bool(P.L > 0), ratio_undefined=bool(P.L <= 0),
    )
    if not np.isfinite(P.w).all() or P.rank_solver != P.d or P.L <= 0:
        record["status"] = "invalid"
        record["error"] = "Full model must have finite coefficients, full solver rank and strictly positive full residual loss; a zero denominator is preserved but blocks ratio/fallback experiments"
    atomic_json(root / "records/full_models" / f"{task}.json", record)
    return record


def execute_case(root: Path, case: dict) -> dict:
    protocol_for(root)
    hashes = source_hashes(root, case["task"])
    arrays = load_arrays(root, case["task"])
    cfg = engine.Config()
    requires_reference = case["method"] == "SpanCancel" or case["protocol"] == "common"
    # Every SC/common scientific seed and timing trial is a full fresh engine run.
    P, prep = fresh_problem(arrays, cfg) if requires_reference else cached_problem(root, case["task"], arrays, cfg, hashes)
    with EngineTimings() as timer:
        start = time.perf_counter()
        if case["protocol"] == "common":
            support_seed = run_seed(case["task"], case["n"], "D-optimal", 0)
            selection = engine.select(P, case["n"], "D-optimal", cfg, seed=support_seed)
            support_timing = timer.components()
            support_fit = timer.final_coefficients.copy()
            support_events = list(timer.events)
            timer.events.clear()
            timer.final_coefficients = None
            timer.final_evaluation_complete = False
            result = engine.reweight(P, selection, cfg, seed=case["seed"])
        else:
            selection = None
            support_timing = None
            support_fit = None
            support_events = []
            result = engine.select(P, case["n"], case["method"], cfg, seed=case["seed"])
        engine_wall_s = time.perf_counter() - start
    coefficients = timer.final_coefficients
    if coefficients is None:
        raise RuntimeError("Final selected fit was not captured")
    components = timer.components()
    if case["protocol"] == "common":
        # D support is native construction; its unweighted fit is only a reference
        # diagnostic before common reweighting. LP/fallback evaluations retained.
        support_cost = selection["candidate_s"]
        reweight_native = result["reweight_s"] - components["shared_final_evaluation_without_fit_s"]
        native = prep["native_precomputation_s"] + support_cost + reweight_native
        native_formula = "fresh full_fit + residual_precompute + D-optimal candidate_s + reweight_s - final_evaluation_without_fit"
    else:
        support_cost = 0.0
        native = prep["native_precomputation_s"] + result["selection_s"] - components["shared_final_evaluation_without_fit_s"]
        native_formula = "native_precomputation + selection_s - final_evaluation_without_fit"
    checked_at = time.perf_counter()
    checks = basic_checks(P, result, coefficients, case["n"])
    metrics = scientific_metrics(P, coefficients, float(arrays["y_scale"].item()))
    extra_check_s = time.perf_counter() - checked_at
    record = dict(
        schema_version=SCHEMA_VERSION, **case, run_key=stable_key(case),
        status="complete" if checks["passed"] else "invalid", saved_at_utc=utcnow(),
        branch=result["branch"], scientific_observation=case["kind"] == "science",
        stochastic_observation=case["kind"] == "science" and (case["method"] in STOCHASTIC_METHODS or result["branch"] == "heuristic"),
        seed_namespace=SEED_NAMESPACE, seed_group=seed_group(case["method"], case["protocol"]),
        seed_components=[SEED_NAMESPACE, case["task"], case["n"], seed_group(case["method"], case["protocol"]), case["rep"]],
        source_hashes=hashes, config=asdict(cfg),
        full_model_record=f'records/full_models/{case["task"]}.json',
        metadata=read_json(root / "data/prepared" / f'{case["task"]}.json'),
        engine_result=result, model=dict(coefficients=coefficients.tolist()),
        metrics=metrics, checks=checks, environment=environment_record(),
        timing=dict(
            **components, preparation=prep, engine_call_wall_s=engine_wall_s,
            native_method_s=float(native), native_formula=native_formula,
            native_support_construction_s=float(support_cost),
            extra_saved_model_and_basic_checks_s=extra_check_s,
            raw_engine_total_s=result["total_s"],
            raw_engine_total_note="Preserved engine field: always includes its full_fit_s/residual_s; not used as fair native method timing.",
            timing_role="One of ten stochastic seed timings" if case["method"] in STOCHASTIC_METHODS or result["branch"] == "heuristic" else "Fresh deterministic timing trial; duplicates are not independent scientific observations",
        ),
    )
    if selection is not None:
        record["support_construction"] = dict(engine_result=selection, model=dict(coefficients=support_fit.tolist()), timing=support_timing, observed_calls=support_events)
    if case["kind"] == "timing":
        reference_case = make_case(case["task"], case["n"], case["method"], protocol=case["protocol"], primary=case["primary"])
        reference = read_json(run_path(root, reference_case))
        same = result["indices"] == reference["engine_result"]["indices"] and np.array_equal(result["weights"], reference["engine_result"]["weights"])
        record["checks"]["deterministic_reference_identical"] = bool(same)
        if not same:
            record["status"] = "invalid"
    if native < 0:
        record["status"] = "invalid"
        record["checks"]["native_timing_nonnegative"] = False
    atomic_json(run_path(root, case), record)
    return record


def current_record(path: Path, hashes: dict) -> dict | None:
    if not path.exists():
        return None
    record = read_json(path)
    if record.get("source_hashes") != hashes:
        raise RuntimeError(f"Existing record has different input/source hashes: {path}")
    return record


def expected_manifest(root: Path, protocol: dict) -> dict:
    rows = []
    shapes = {}
    unresolved = []
    for task in protocol["tasks"]:
        path = root / "data/prepared" / f"{task}.json"
        if not path.exists() or not path.with_suffix(".npz").exists():
            unresolved.append(task)
            continue
        metadata = read_json(path)
        N, d, Nt = (int(metadata[k]) for k in ("n_train", "design_dimension", "n_test"))
        if max(budgets(d)) > N:
            raise ValueError(f"Budget exceeds rows: {task}")
        shapes[task] = dict(N=N, d=d, Ntest=Nt, budgets=budgets(d), primary=math.ceil(1.5 * d))
        base = scientific_base_cases(task, d, protocol["methods"])
        rows.extend(base)
        for case in base:
            path = run_path(root, case)
            if path.exists():
                record = read_json(path)
                rows.extend(expansion_cases(case, record))
                rows.extend(deterministic_timing_cases(case, record))
    return dict(
        schema_version=SCHEMA_VERSION, saved_at_utc=utcnow(), source_hashes=dict(engine_sha256=sha256(Path(engine.__file__)), runner_sha256=sha256(Path(__file__)), protocol_sha256=sha256(root / "protocol.json")),
        tasks=shapes, unprepared_tasks=unresolved,
        expected_cases=[dict(**r, run_key=stable_key(r), path=f"records/runs/{stable_key(r)}.json") for r in rows],
        conditional_rules=dict(heuristic="First scientific rep0 heuristic branch commits to ten full seeds 0..9", deterministic_primary="One scientific trial0 plus timing-only fresh trials1 and2; exactly3 timings", stochastic="Ten fixed scientific seed IDs0..9, each with one measured timing"),
    )


def archive_attempt(root: Path, path: Path) -> None:
    if not path.exists():
        return
    previous = read_json(path)
    target = root / "records/attempts" / f"{path.stem}__{time.time_ns()}.json"
    atomic_json(target, previous)


def worker_main(root: Path, spec: dict) -> int:
    with exclusive_lock(root / "work/heavy_worker.lock"), threadpoolctl.threadpool_limits(limits=1):
        if spec["kind"] == "full_model":
            result = prepare_full_model(root, spec["task"])
        else:
            result = execute_case(root, spec)
    return 0 if result["status"] == "complete" else 2


def launch_attempt(root: Path, spec: dict, protocol: dict, state: dict) -> bool:
    cap = protocol["resource_limits"]["cumulative_execution_seconds"]
    remaining = cap - state["charged_execution_s"]
    if remaining <= 0:
        state["status"] = "paused_resource_limit"
        atomic_json(root / "records/campaign_state.json", state)
        log_event(root, "campaign_paused", reason="cumulative_execution_limit", **state)
        return False
    timeout = min(protocol["resource_limits"]["attempt_seconds"], remaining)
    key = f'full_model__{spec["task"]}' if spec["kind"] == "full_model" else stable_key(spec)
    hashes = source_hashes(root, spec["task"])
    record_path = root / "records/full_models" / f'{spec["task"]}.json' if spec["kind"] == "full_model" else run_path(root, spec)
    archive_attempt(root, record_path)
    record_path.unlink(missing_ok=True)
    spec_path = root / "work/worker_specs" / f"{key}.json"
    atomic_json(spec_path, spec)
    stdout_path = root / "work/worker_logs" / f"{key}__{time.time_ns()}.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    launch_utc = utcnow()
    log_event(root, "attempt_started", run_key=key, spec=spec, timeout_s=timeout, charged_execution_before_s=state["charged_execution_s"])
    with stdout_path.open("w", encoding="utf-8") as stdout:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--worker-spec", str(spec_path)], stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
        state["active_attempt"] = dict(run_key=key, pid=proc.pid, started_epoch_s=time.time(), started_at_utc=launch_utc, timeout_s=timeout, spec=spec, source_hashes=hashes)
        atomic_json(root / "records/campaign_state.json", state)
        interrupted = None
        try:
            returncode = proc.wait(timeout=timeout)
            outcome = "complete" if returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            returncode, outcome = None, "timeout"
        except BaseException as exc:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            returncode, outcome, interrupted = None, "interrupted", exc
        finally:
            elapsed = time.perf_counter() - started
            state["charged_execution_s"] += elapsed
            state["attempts"] += 1
            state["active_attempt"] = None
            state["updated_at_utc"] = utcnow()
            atomic_json(root / "records/campaign_state.json", state)
    result = read_json(record_path) if record_path.exists() else None
    if outcome != "complete" and (result is None or result.get("status") == "complete"):
        result = dict(schema_version=SCHEMA_VERSION, **spec, status=outcome, run_key=key, source_hashes=hashes, config=protocol["config"], saved_at_utc=utcnow(), error=dict(returncode=returncode, stdout_path=str(stdout_path.relative_to(root)), timeout_s=timeout))
        atomic_json(record_path, result)
    elif result is None:
        raise RuntimeError("Successful worker did not save its record")
    result["attempt"] = dict(started_at_utc=launch_utc, execution_wall_s=elapsed, timeout_s=timeout, returncode=returncode, stdout_path=str(stdout_path.relative_to(root)))
    atomic_json(record_path, result)
    log_event(root, "attempt_finished", run_key=key, outcome=result["status"], execution_wall_s=elapsed, returncode=returncode, charged_execution_total_s=state["charged_execution_s"])
    print(f'{key}: {result["status"]}; charged {state["charged_execution_s"]:.1f}/{cap}s', flush=True)
    if interrupted is not None:
        raise interrupted
    return True


def recover_abandoned_attempt(root: Path, state: dict) -> None:
    active = state.get("active_attempt")
    if active is None:
        return
    try:
        os.kill(active["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError(f'Prior heavy worker PID {active["pid"]} is still alive; campaign cannot launch another')
    # A hard-killed controller cannot measure its last wall interval. Charge the
    # full allowed timeout conservatively; preserve that distinction explicitly.
    charge = active["timeout_s"]
    state["charged_execution_s"] += charge
    state["attempts"] += 1
    state["active_attempt"] = None
    atomic_json(root / "records/campaign_state.json", state)
    log_event(root, "abandoned_attempt_recovered", active_attempt=active, conservative_charge_s=charge, reason="Controller stopped before elapsed wall interval was persisted")
    spec = active["spec"]
    path = root / "records/full_models" / f'{spec["task"]}.json' if spec["kind"] == "full_model" else run_path(root, spec)
    if not path.exists():
        atomic_json(path, dict(schema_version=SCHEMA_VERSION, **spec, run_key=active["run_key"], status="interrupted", source_hashes=active["source_hashes"], saved_at_utc=utcnow(), attempt=dict(execution_wall_s=None, conservative_resource_charge_s=charge), error="Prior controller ended before the attempt completed"))


def campaign(root: Path, *, retry_failures: bool = False, tasks: list[str] | None = None) -> dict:
    protocol = protocol_for(root)
    with exclusive_lock(root / "work/campaign.lock"):
        manifest = expected_manifest(root, protocol)
        selected_tasks = list(protocol["tasks"] if tasks is None else tasks)
        if len(selected_tasks) != len(set(selected_tasks)) or set(selected_tasks) - set(protocol["tasks"]):
            raise ValueError("Selected tasks must be distinct names from protocol.json")
        unprepared = set(manifest["unprepared_tasks"]) & set(selected_tasks)
        if unprepared:
            raise RuntimeError(f'Prepared inputs are missing: {sorted(unprepared)}')
        atomic_json(root / "records/expected_grid.json", manifest)
        state_path = root / "records/campaign_state.json"
        state = read_json(state_path) if state_path.exists() else dict(schema_version=SCHEMA_VERSION, started_at_utc=utcnow(), charged_execution_s=0.0, attempts=0, active_attempt=None, status="running", resource_charge_scope="Wall time of every full-model preparation, scientific run, timing-only repetition and unsuccessful worker attempt, including worker startup/input IO; data download/preprocessing and controller plan IO are outside this execution ledger.")
        recover_abandoned_attempt(root, state)
        state["status"] = "running"
        atomic_json(state_path, state)
        ready_tasks = []
        for task in selected_tasks:
            hashes = source_hashes(root, task)
            full_path = root / "records/full_models" / f"{task}.json"
            full = current_record(full_path, hashes)
            if full is None or (full.get("status") != "complete" and retry_failures):
                if not launch_attempt(root, dict(kind="full_model", task=task), protocol, state):
                    return state
                full = read_json(full_path)
            if full.get("status") != "complete":
                log_event(root, "task_blocked", task=task, reason="full_model_preparation_failed", full_model_status=full.get("status"))
                continue
            ready_tasks.append(task)
        queue = []
        # Finish every selected task's primary budget before secondary budgets.
        for primary_phase in (True, False):
            for task in ready_tasks:
                d = manifest["tasks"][task]["d"]
                queue.extend(r for r in scientific_base_cases(task, d, protocol["methods"]) if r["primary"] == primary_phase)
        seen = set()
        cursor = 0
        while cursor < len(queue):
            case = queue[cursor]
            cursor += 1
            key = stable_key(case)
            if key in seen:
                continue
            seen.add(key)
            path = run_path(root, case)
            hashes = source_hashes(root, case["task"])
            result = current_record(path, hashes)
            if result is None or (result.get("status") != "complete" and retry_failures):
                if not launch_attempt(root, case, protocol, state):
                    atomic_json(root / "records/expected_grid.json", expected_manifest(root, protocol))
                    return state
                result = read_json(path)
            # Insert expansions immediately to keep the primary budget first.
            additions = expansion_cases(case, result) + deterministic_timing_cases(case, result)
            queue[cursor:cursor] = additions
            atomic_json(root / "records/expected_grid.json", expected_manifest(root, protocol))
            if result.get("status") != "complete":
                log_event(root, "case_terminal_failure", run_key=key, outcome=result.get("status"), retry_requires_explicit_flag=True)
        expected = expected_manifest(root, protocol)
        atomic_json(root / "records/expected_grid.json", expected)
        outcomes = {}
        for row in expected["expected_cases"]:
            path = root / row["path"]
            status = read_json(path).get("status", "unknown") if path.exists() else "missing"
            outcomes[status] = outcomes.get(status, 0) + 1
        all_tasks_selected = set(selected_tasks) == set(protocol["tasks"])
        state["status"] = "execution_complete_pending_independent_verification" if set(outcomes) == {"complete"} and all_tasks_selected else "selected_tasks_execution_ended" if not all_tasks_selected else "execution_ended_with_failures"
        state["selected_tasks_last_invocation"] = selected_tasks
        state["outcomes"] = outcomes
        state["updated_at_utc"] = utcnow()
        atomic_json(state_path, state)
        log_event(root, "campaign_execution_ended", **state)
        return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--execute", action="store_true", help="Run the case studies")
    parser.add_argument("--retry-failures", action="store_true", help="Explicitly retry saved terminal errors/timeouts; old attempts are archived")
    parser.add_argument("--tasks", nargs="+", help="Run the selected tasks from protocol.json")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker_spec:
        spec = read_json(args.worker_spec)
        try:
            return worker_main(root, spec)
        except BaseException as exc:
            record = dict(schema_version=SCHEMA_VERSION, **spec, status="error", saved_at_utc=utcnow(), config=asdict(engine.Config()), error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
            try:
                record["source_hashes"] = source_hashes(root, spec["task"])
                path = root / "records/full_models" / f'{spec["task"]}.json' if spec["kind"] == "full_model" else run_path(root, spec)
                atomic_json(path, record)
            finally:
                traceback.print_exc()
            return 1
    protocol = protocol_for(root)
    if args.execute:
        print(json.dumps(campaign(root, retry_failures=args.retry_failures, tasks=args.tasks), indent=2))
    else:
        plan = expected_manifest(root, protocol)
        print(json.dumps(dict(mode="plan_only", real_workers_started=0, tasks=plan["tasks"], unprepared_tasks=plan["unprepared_tasks"], base_expected_cases=len(plan["expected_cases"]), conditional_rules=plan["conditional_rules"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
