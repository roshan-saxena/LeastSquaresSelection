#!/usr/bin/env python3
"""Download original UCI data, then prepare the regression case studies.

Downloads use only the Python standard library and HTTP GET. Preparation uses
the study's pinned NumPy/SciPy/scikit-learn environment. Existing raw files and
prepared arrays are checked and reused; conflicting files are never replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
import urllib.request
import zipfile

# Set before importing a numerical package. Preparation verifies the effective
# limits as well, so an inherited larger thread setting cannot go unnoticed.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
FEATURES_GAS = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
TASKS = ("gas_co", "superconductivity")
SOURCES = {
    "gas": {
        "uci_id": 551,
        "title": "Gas Turbine CO and NOx Emission Data Set",
        "page_url": "https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set",
        "download_url": "https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip",
        "doi": "10.24432/C5WC95", "license": "CC BY 4.0",
        "attribution": "Gas Turbine CO and NOx Emission Data Set [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5WC95.",
        "files": [f"gt_{year}.csv" for year in range(2011, 2016)],
        "expected_rows": 36733,
    },
    "superconductivity": {
        "uci_id": 464, "title": "Superconductivty Data",
        "page_url": "https://archive.ics.uci.edu/dataset/464/superconductivty+data",
        "download_url": "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
        "doi": "10.24432/C53P47", "license": "CC BY 4.0",
        "attribution": "Hamidieh, K. (2018). Superconductivty Data [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53P47.",
        "files": ["train.csv", "unique_m.csv"], "expected_rows": 21263,
    },

}


class DataIssue(RuntimeError):
    """An unexpected data property or conflicting input file."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def create_or_verify(path, payload):
    """Write once, or verify byte identity. Never replace a conflicting file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DataIssue(f"Conflicting existing file: {path}")
        return False
    with path.open("xb") as f:
        f.write(payload)
    return True


def publish_temp(temp, destination):
    """Atomically publish a complete temporary file without overwriting."""
    destination = Path(destination)
    try:
        os.link(temp, destination)
    except FileExistsError:
        if sha256(temp) != sha256(destination):
            raise DataIssue(f"Conflicting existing file: {destination}")
    finally:
        Path(temp).unlink(missing_ok=True)


def download_source(name):
    source = SOURCES[name]
    raw_dir = ROOT / "data" / "raw" / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "original.zip"
    provenance = raw_dir / "download.json"
    if archive.exists():
        if not provenance.exists():
            raise DataIssue(f"Existing archive has no download provenance: {archive}")
        meta = json.loads(provenance.read_text())
        if meta["requested_url"] != source["download_url"] or meta["sha256"] != sha256(archive):
            raise DataIssue(f"Cached original archive/provenance mismatch: {archive}")
        print(f"Reusing verified original archive: {name}", flush=True)
    else:
        with tempfile.NamedTemporaryFile(prefix="download-", suffix=".part", dir=raw_dir, delete=False) as f:
            temp = Path(f.name)
        try:
            request = urllib.request.Request(source["download_url"], method="GET",
                                             headers={"User-Agent": "MLWA-regression-reproducibility/1.0"})
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as out:
                if response.status != 200:
                    raise DataIssue(f"Download returned HTTP {response.status}: {name}")
                size = 0
                next_notice = 50 * 1024 * 1024
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    size += len(chunk)
                    if size >= next_notice:
                        print(f"Downloading {name}: {size / 1024**2:.0f} MiB", flush=True)
                        next_notice += 50 * 1024 * 1024
                expected_size = response.headers.get("Content-Length")
                if expected_size is not None and size != int(expected_size):
                    raise DataIssue(f"Incomplete original download for {name}: {size} != {expected_size}")
                meta = {"requested_url": source["download_url"], "resolved_url": response.geturl(),
                        "http_method": "GET", "http_status": response.status,
                        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                        "http_content_length": expected_size, "bytes": size,
                        "sha256": sha256(temp), "download_seconds": time.monotonic() - started}
            if not zipfile.is_zipfile(temp):
                raise DataIssue(f"UCI download is not a valid ZIP archive: {name}")
            publish_temp(temp, archive)
            create_or_verify(provenance, json_bytes(meta))
        finally:
            temp.unlink(missing_ok=True)
    # Preserve the unmodified ZIP and the unmodified bytes of each original file.
    extracted = []
    with zipfile.ZipFile(archive) as z:
        for filename in source["files"]:
            members = [info for info in z.infolist() if Path(info.filename).name == filename]
            if len(members) != 1:
                raise DataIssue(f"Expected one archive member for {filename}; found {len(members)}")
            info = members[0]
            destination = raw_dir / filename
            with tempfile.NamedTemporaryFile(prefix="extract-", suffix=".part", dir=raw_dir, delete=False) as out:
                temp = Path(out.name)
                try:
                    with z.open(info) as stream:
                        shutil.copyfileobj(stream, out, 1024 * 1024)
                except BaseException:
                    temp.unlink(missing_ok=True)
                    raise
            try:
                if temp.stat().st_size != info.file_size:
                    raise DataIssue(f"Extracted length mismatch for {filename}")
                publish_temp(temp, destination)
            finally:
                temp.unlink(missing_ok=True)
            extracted.append({"path": str(destination.relative_to(ROOT)), "archive_member": info.filename,
                              "bytes": destination.stat().st_size, "sha256": sha256(destination)})
    return {**source, "archive": {"path": str(archive.relative_to(ROOT)), **meta}, "original_files": extracted}


def require_numerical_environment():
    global np, StandardScaler, GroupShuffleSplit
    import numpy as np
    import scipy
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupShuffleSplit
    from threadpoolctl import threadpool_info, threadpool_limits
    threadpool_limits(limits=1)
    libraries = threadpool_info()
    if any(item["num_threads"] != 1 for item in libraries):
        raise DataIssue("A numerical library is using more than one BLAS thread")
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "blas_threads": 1,
            "threadpools": [{key: item.get(key) for key in ("user_api", "internal_api", "version", "num_threads", "architecture")}
                            for item in libraries]}


def load_csv(path):
    with Path(path).open(newline="") as f:
        header = next(csv.reader(f))
    a = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if a.shape[1] != len(header):
        raise DataIssue(f"CSV header/column mismatch: {path}")
    return header, a


def check_numeric(name, a):
    bad = int(np.count_nonzero(~np.isfinite(a)))
    if bad:
        raise DataIssue(f"{name} contains {bad} missing/non-finite values")
    return {"missing_or_nonfinite_count": bad, "dtype": str(a.dtype)}


def load_gas():
    rows = []
    per_year = []
    header = None
    offset = 0
    for year in range(2011, 2016):
        path = ROOT / "data" / "raw" / "gas" / f"gt_{year}.csv"
        names, a = load_csv(path)
        if names != FEATURES_GAS + ["CO", "NOX"]:
            raise DataIssue(f"Unexpected gas columns in {path}: {names}")
        if header is not None and header != names:
            raise DataIssue("Gas yearly headers differ")
        header = names
        rows.append(a)
        per_year.append({"year": year, "rows": len(a), "original_row_start_zero_based": offset,
                         "original_row_stop_exclusive": offset + len(a)})
        offset += len(a)
    a = np.concatenate(rows, axis=0)
    if len(a) != SOURCES["gas"]["expected_rows"]:
        raise DataIssue(f"Unexpected gas row count: {len(a)}")
    check_numeric("gas originals", a)
    ntrain = sum(year["rows"] for year in per_year[:3])
    return a[:, :9], a[:, 9], np.arange(ntrain), np.arange(ntrain, len(a)), {
        "rule": "Concatenate original yearly files in ascending year; 2011-2013 training, 2014-2015 test.",
        "train_years": [2011, 2012, 2013], "test_years": [2014, 2015],
        "year_is_predictor": False, "other_emission_target_is_predictor": False,
        "row_index_basis": "Zero-based row in ascending-year concatenation, excluding CSV headers.",
        "per_year": per_year}


def load_superconductivity():
    raw_dir = ROOT / "data" / "raw" / "superconductivity"
    header, a = load_csv(raw_dir / "train.csv")
    if len(header) != 82 or header[-1] != "critical_temp":
        raise DataIssue(f"Unexpected superconductivity predictor/target columns: {header}")
    with (raw_dir / "unique_m.csv").open(newline="") as f:
        composition_header = next(csv.reader(f))
    if composition_header[-2:] != ["critical_temp", "material"]:
        raise DataIssue(f"Unexpected unique_m.csv target/material columns: {composition_header[-2:]}")
    numeric = np.loadtxt(raw_dir / "unique_m.csv", delimiter=",", skiprows=1,
                         usecols=range(len(composition_header) - 1), dtype=np.float64, ndmin=2)
    if len(a) != SOURCES["superconductivity"]["expected_rows"] or len(numeric) != len(a):
        raise DataIssue("Superconductivity train.csv/unique_m.csv row counts do not match the expected 21263")
    check_numeric("superconductivity train.csv", a)
    check_numeric("superconductivity elemental composition/target", numeric)
    if not np.array_equal(a[:, -1], numeric[:, -1]):
        unequal = int(np.count_nonzero(a[:, -1] != numeric[:, -1]))
        raise DataIssue(f"Superconductivity target correspondence fails on {unequal} rows")
    composition = numeric[:, :-1]
    element_count = np.count_nonzero(composition, axis=1)
    if not np.array_equal(a[:, 0], element_count):
        mismatches = np.flatnonzero(a[:, 0] != element_count)
        raise DataIssue(f"Superconductivity number_of_elements/composition correspondence fails on {len(mismatches)} rows")
    # Group by the transitive closure of exact
    # supplied-composition equality OR exact supplied-predictor equality. Labels
    # are the minimum original row index in each component, not an incidental
    # dictionary insertion or lexicographic order of the feature vectors.
    _, composition_groups = np.unique(composition, axis=0, return_inverse=True)
    _, feature_groups = np.unique(a[:, :-1], axis=0, return_inverse=True)
    parent = np.arange(len(a), dtype=np.int64)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i
    for equality_labels in (composition_groups, feature_groups):
        first = {}
        for i, label in enumerate(equality_labels):
            label = int(label)
            if label not in first:
                first[label] = i
            else:
                left, right = find(first[label]), find(i)
                if left != right:
                    parent[max(left, right)] = min(left, right)
    group_ids = np.asarray([find(i) for i in range(len(a))], dtype=np.int64)
    _, group_counts = np.unique(group_ids, return_counts=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train, test = next(splitter.split(a[:, :-1], a[:, -1], groups=group_ids))
    overlap = np.intersect1d(group_ids[train], group_ids[test])
    if len(overlap):
        raise DataIssue("Identical superconductivity elemental compositions cross the split")
    split = {"rule": "GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0) on connected components linked by identical supplied elemental composition vectors OR identical supplied 81-predictor vectors.",
             "row_index_basis": "Zero-based original train.csv row, excluding header; unique_m.csv row correspondence verified by exact critical_temp equality.",
             "target_correspondence_exact": True, "target_correspondence_mismatches": 0,
             "number_of_elements_correspondence_exact": True, "number_of_elements_correspondence_mismatches": 0,
             "composition_feature_names": composition_header[:-2],
             "composition_dimensions": composition.shape[1], "composition_is_normalized": False,
             "grouping_algorithm": "Union-find transitive closure of equality edges for each of the two complete float64 vectors; no normalization, tolerance, formula parsing, target grouping, or row removal.",
             "component_label": "Minimum zero-based original row index in each connected component; sklearn GroupShuffleSplit orders distinct component labels ascending.",
             "component_labels_sha256": hashlib.sha256(group_ids.astype("<i8").tobytes()).hexdigest(),
             "component_labels_hash_encoding": "Full original-row component label array, little-endian signed int64, SHA-256.",
             "n_composition_only_groups": int(len(np.unique(composition_groups))),
             "n_predictor_only_groups": int(len(np.unique(feature_groups))),
             "n_groups": int(len(group_counts)), "n_train_groups": int(len(np.unique(group_ids[train]))),
             "n_test_groups": int(len(np.unique(group_ids[test]))), "cross_split_groups": int(len(overlap)),
             "groups_with_multiple_original_rows": int(np.count_nonzero(group_counts > 1)),
             "max_original_rows_per_group": int(group_counts.max()),
             "test_size_is_group_fraction": True, "random_state": 0}
    return a[:, :-1], a[:, -1], train, test, header[:-1], split




def duplicate_checks(x, y, train, test):
    """Exact numerical duplicates, using SHA-256 for bounded-memory indexing.

    Matching hashes are independently checked against the float64 rows. Signed
    zero is canonicalized only in the temporary duplicate-check representation.
    The prepared scientific values are never changed by this check.
    """
    feature_hashes = np.empty(len(x), dtype="S32")
    joint_hashes = np.empty(len(x), dtype="S32")
    for i, row in enumerate(x):
        canonical = np.where(row == 0.0, 0.0, row).astype("<f8", copy=False).tobytes()
        target = np.asarray(0.0 if y[i] == 0.0 else y[i], dtype="<f8").tobytes()
        feature_hashes[i] = hashlib.sha256(canonical).digest()
        joint_hashes[i] = hashlib.sha256(canonical + target).digest()
    def check_hashes(hashes, include_target=False):
        _, inverse, counts = np.unique(hashes, return_inverse=True, return_counts=True)
        first = {}
        for i in np.flatnonzero(counts[inverse] > 1):
            group = int(inverse[i])
            if group in first:
                previous = first[group]
                if not np.array_equal(x[i], x[previous]) or (include_target and y[i] != y[previous]):
                    raise DataIssue("SHA-256 row collision encountered during duplicate checks")
            else:
                first[group] = i
        return {"unique_rows": int(len(counts)), "duplicate_rows_after_first": int(np.sum(counts - 1)),
                "duplicate_groups": int(np.count_nonzero(counts > 1)),
                "max_group_size": int(counts.max()),
                "train_duplicate_rows_after_first": int(len(train) - len(np.unique(hashes[train]))),
                "test_duplicate_rows_after_first": int(len(test) - len(np.unique(hashes[test]))),
                "cross_split_distinct_duplicate_groups": int(len(np.intersect1d(hashes[train], hashes[test])))}
    return {"method": "Float64 numerical equality, signed-zero-canonical SHA-256 index with direct equality confirmation for every duplicate group.",
            "features_only": check_hashes(feature_hashes),
            "features_and_target": check_hashes(joint_hashes, True)}


def spectrum(a):
    # Work directly on the design: forming X.T @ X squares the condition number.
    s = np.linalg.svd(a, full_matrices=False, compute_uv=False)
    return {"singular_values": s.tolist(), "rank_absolute_1e-8": int(np.count_nonzero(s > 1e-8)),
            "rank_relative_1e-12": int(np.count_nonzero(s > 1e-12 * s[0])),
            "rank_absolute_threshold": 1e-8, "rank_relative_threshold": 1e-12,
            "largest_singular_value": float(s[0]), "smallest_singular_value": float(s[-1]),
            "condition_number_2": float(s[0] / s[-1]) if s[-1] > 0 else None,
            "condition_number_is_infinite": bool(s[-1] == 0)}


def prepare_task(task, x, y, train, test, feature_names, split, raw_source):
    print(f"Preparing {task}: train={len(train)}, test={len(test)}, predictors={x.shape[1]}", flush=True)
    if x.shape != (len(y), len(feature_names)):
        raise DataIssue(f"{task} inconsistent predictors, labels, and row counts")
    if not np.array_equal(np.sort(np.concatenate((train, test))), np.arange(len(y))):
        raise DataIssue(f"{task} split does not partition every original row exactly once")
    finite_x, finite_y = check_numeric(f"{task} X", x), check_numeric(f"{task} y", y)
    constants = {"all_original_rows": np.flatnonzero(np.ptp(x, axis=0) == 0).tolist(),
                 "training_rows": np.flatnonzero(np.ptp(x[train], axis=0) == 0).tolist(),
                 "test_rows": np.flatnonzero(np.ptp(x[test], axis=0) == 0).tolist()}
    if any(constants.values()) or np.ptp(y[train]) == 0 or np.ptp(y[test]) == 0:
        raise DataIssue(f"{task} has an unexpected constant predictor or target: {constants}")
    duplicates = duplicate_checks(x, y, train, test)
    scaler_x = StandardScaler().fit(x[train])
    scaler_y = StandardScaler().fit(y[train, None])
    train_design = np.ones((len(train), x.shape[1] + 1), dtype=np.float64)
    test_design = np.ones((len(test), x.shape[1] + 1), dtype=np.float64)
    train_design[:, 1:] = scaler_x.transform(x[train])
    test_design[:, 1:] = scaler_x.transform(x[test])
    train_y = scaler_y.transform(y[train, None]).ravel()
    test_y = scaler_y.transform(y[test, None]).ravel()
    check_numeric(f"{task} scaled design", train_design)
    print(f"Computing {task} full-training design singular values", flush=True)
    train_spectrum = spectrum(train_design)
    test_spectrum = spectrum(test_design)
    expected_rank = train_design.shape[1]
    if (train_spectrum["rank_absolute_1e-8"] != expected_rank
            or train_spectrum["rank_relative_1e-12"] != expected_rank):
        raise DataIssue(f"{task} training rank is {train_spectrum['rank_absolute_1e-8']}/{train_spectrum['rank_relative_1e-12']} rather than {expected_rank}")
    arrays = {"X": train_design, "y": train_y, "Xtest": test_design, "ytest": test_y,
              "train_indices": np.asarray(train, dtype=np.int64), "test_indices": np.asarray(test, dtype=np.int64),
              "x_mean": scaler_x.mean_, "x_scale": scaler_x.scale_,
              "y_mean": scaler_y.mean_, "y_scale": scaler_y.scale_}
    destination = ROOT / "data" / "prepared" / f"{task}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with np.load(destination, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise DataIssue(f"Conflicting existing prepared arrays: {destination}")
        print(f"Reusing byte-value-identical prepared arrays: {task}", flush=True)
    else:
        with tempfile.NamedTemporaryFile(prefix=f"{task}-", suffix=".npz", dir=destination.parent, delete=False) as f:
            temp = Path(f.name)
        try:
            # Uncompressed NPZ keeps loading predictable and preserves float64.
            np.savez(temp, **arrays)
            publish_temp(temp, destination)
        finally:
            temp.unlink(missing_ok=True)
    # Read back actual delivery rather than trusting only in-memory values.
    with np.load(destination, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise DataIssue(f"Prepared readback mismatch: {destination}")
    target = {"gas_co": "CO", "superconductivity": "critical_temp"}[task]
    split = dict(split)
    split["train_indices_sha256"] = hashlib.sha256(np.asarray(train, dtype="<i8").tobytes()).hexdigest()
    split["test_indices_sha256"] = hashlib.sha256(np.asarray(test, dtype="<i8").tobytes()).hexdigest()
    split["index_hash_encoding"] = "Contiguous original-row index array, little-endian signed int64, SHA-256."
    metadata = {"source_dataset": raw_source, "path": str(destination.relative_to(ROOT)),
            "sha256": sha256(destination), "bytes": destination.stat().st_size,
            "n_original": len(y), "n_train": len(train), "n_test": len(test),
            "predictors_excluding_intercept": x.shape[1], "design_dimension": expected_rank,
            "feature_names": ["intercept"] + feature_names, "target_name": target,
            "split": split, "scaling": {"method": "sklearn.preprocessing.StandardScaler, population variance (ddof=0), fitted on training rows only for both predictors and target.",
                                          "intercept_added_after_scaling": True, "intercept_column_index": 0,
                                          "x_mean": scaler_x.mean_.tolist(), "x_scale": scaler_x.scale_.tolist(),
                                          "y_mean": scaler_y.mean_.tolist(), "y_scale": scaler_y.scale_.tolist(),
                                          "no_other_transformations_or_row_removals": True},
            "checks": {"features": finite_x, "target": finite_y, "constant_predictor_indices": constants,
                       "split_index_overlap_count": int(len(np.intersect1d(train, test))),
                       "original_rows_preserved": True, "duplicates": duplicates,
                       "training_design": train_spectrum, "test_design": test_spectrum,
                       "prepared_arrays_readback_exact": True}}
    provenance = json.loads((ROOT / "data" / "raw" / raw_source / "download.json").read_text())
    metadata["raw_archive_sha256"] = provenance["sha256"]
    metadata["raw_original_file_sha256"] = {filename: sha256(ROOT / "data" / "raw" / raw_source / filename)
                                           for filename in SOURCES[raw_source]["files"]}
    metadata["metadata_path"] = str(destination.with_suffix(".json").relative_to(ROOT))
    create_or_verify(destination.with_suffix(".json"), json_bytes(metadata))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("download", "prepare", "all"), default="all")
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=list(TASKS))
    args = parser.parse_args()
    requested = set(args.tasks)
    datasets = [name for name in SOURCES if (name == "gas" and "gas_co" in requested) or name in requested]
    raw_meta = {}
    for name in datasets:
        if args.stage == "prepare" and not (ROOT / "data" / "raw" / name / "original.zip").exists():
            raise DataIssue(f"Missing original {name} archive; run --stage download first")
        raw_meta[name] = download_source(name)
    if args.stage == "download":
        print("Original downloads complete; no numerical preparation was performed.", flush=True)
        return
    environment = require_numerical_environment()
    manifest_path = ROOT / "records" / "data_manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    manifest = {"schema_version": 1, "status": "partially_prepared",
                "created_utc": existing["created_utc"] if existing else datetime.now(timezone.utc).isoformat(),
                "preparation_source_path": "source/prepare_data.py", "preparation_source_sha256": sha256(__file__),
                "environment": environment, "datasets": dict(existing.get("datasets", {})) if existing else {},
                "tasks": dict(existing.get("tasks", {})) if existing else {}}
    for name, meta in raw_meta.items():
        if name in manifest["datasets"] and manifest["datasets"][name] != meta:
            raise DataIssue(f"Existing dataset provenance differs for {name}")
        manifest["datasets"][name] = meta
    completed = {}
    if "gas_co" in requested:
        x, y, train, test, split = load_gas()
        completed["gas_co"] = prepare_task("gas_co", x, y, train, test, FEATURES_GAS, split, "gas")
        del x, y, train, test
    if "superconductivity" in requested:
        x, y, train, test, names, split = load_superconductivity()
        completed["superconductivity"] = prepare_task("superconductivity", x, y, train, test, names, split, "superconductivity")
        del x, y, train, test
    for task, meta in completed.items():
        if task in manifest["tasks"] and manifest["tasks"][task] != meta:
            raise DataIssue(f"Existing prepared task metadata differs for {task}")
        manifest["tasks"][task] = meta
    manifest["status"] = "validated" if set(TASKS).issubset(manifest["tasks"]) else "partially_prepared"
    payload = json_bytes(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if existing is not None:
        if json_bytes(existing) == payload:
            print("Existing data manifest verified; no overwrite.", flush=True)
        elif (existing["preparation_source_sha256"] == manifest["preparation_source_sha256"]
              and existing["environment"] == environment
              and set(existing["tasks"]).issubset(manifest["tasks"])):
            # The only permitted update adds previously absent dataset/task
            # entries. All existing scientific metadata above was compared.
            print("Adding newly prepared tasks to the existing data manifest.", flush=True)
            with tempfile.NamedTemporaryFile(prefix="data-manifest-", suffix=".json", dir=manifest_path.parent, delete=False) as f:
                temp = Path(f.name)
                f.write(payload)
            os.replace(temp, manifest_path)
        else:
            raise DataIssue("Existing data manifest conflicts with this preparation; preserve it and review")
    else:
        create_or_verify(manifest_path, payload)
    print("Validated and saved tasks: " + ", ".join(completed), flush=True)


if __name__ == "__main__":
    try:
        main()
    except DataIssue as exc:
        print(f"Data preparation error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
