from experiments.paths import RESULTS, OUTPUT
from pathlib import Path
import json, sys, hashlib, time
import numpy as np
from threadpoolctl import threadpool_limits
from spancancel.datasets_local import holdout, load_input, input_metadata


def main():
    sys.dont_write_bytecode = True

    ROOT = Path(__file__).resolve().parents[1]
    cache = {}
    checks = []
    issues = []
    count = 0
    maxrel = 0.0
    maxabs = 0.0
    restart_count = 0
    with threadpool_limits(1):
        for p in (RESULTS).glob("*.jsonl"):
            seen = set()
            for line in p.read_text().splitlines():
                r = json.loads(line)
                assert r["key"] not in seen, (p, r["key"])
                seen.add(r["key"])
                if r.get("status") != "ok" or "weights" not in r or "data_id" not in r:
                    continue
                dataid = r["data_id"]
                cfg = r.get("config", {})
                rc = cfg.get("solver_rcond", 1e-12)
                key = (dataid, rc)
                if key not in cache:
                    z = load_input(dataid)
                    X = z["X"]
                    y = z["y"]
                    w = np.linalg.lstsq(X, y, rcond=rc)[0]
                    L = float(np.linalg.norm(X @ w - y) ** 2)
                    cache[key] = (X, y, w, L)
                X, y, w, L = cache[key]
                S = np.asarray(r["indices"], int)
                a = np.asarray(r["weights"], float)
                assert len(S) <= r["n"] and len(S) == len(a) and len(set(S)) == len(S)
                assert np.isfinite(a).all() and np.all(a >= 0) and a.sum() > 0
                aa = np.sqrt(a / a.max())
                wf = np.linalg.lstsq(X[S] * aa[:, None], y[S] * aa, rcond=rc)[0]
                ratio = float(np.linalg.norm(X @ wf - y) ** 2 / L)
                delta = abs(ratio - r["ratio"])
                rel = delta / max(1.0, abs(r["ratio"]))
                maxrel = max(maxrel, rel)
                maxabs = max(maxabs, delta)
                count += 1
                if rel > 1e-9:
                    issues.append(
                        dict(
                            key=r["key"],
                            stored=r["ratio"],
                            recomputed=ratio,
                            relative_discrepancy=rel,
                        )
                    )
                for z in r.get("restarts", []):
                    if "result_weights" in z:
                        az = np.asarray(z["result_weights"])
                        assert (
                            np.isfinite(az).all()
                            and np.all(az >= 0)
                            and abs(az.sum() - 1) < 1e-10
                        )
                        restart_count += 1
        # All saved held-out transformations are reproduced from raw source rows.
        transformation = []
        for data_id, m in input_metadata().items():
            if m.get("fit_preprocessing_on") != "training split only":
                continue
            name = m["dataset"]
            split = m["split_seed"]
            M = m.get("rff_count")
            cap = m["retained_N"] if M else None
            X, y, Xe, ye, mm = holdout(name, split, cap, M)
            z = load_input(data_id)
            errs = {
                k: float(np.max(np.abs(v - z[k])))
                for k, v in [("X", X), ("y", y), ("Xtest", Xe), ("ytest", ye)]
            }
            assert set(m["train_indices"]).isdisjoint(m["test_indices"])
            assert all(v <= 1e-9 for v in errs.values()), (data_id, errs)
            transformation.append(
                dict(
                    data_id=data_id,
                    max_abs_errors=errs,
                    train_test_disjoint=True,
                    whitened_train_gram_error=(
                        None
                        if not M
                        else float(np.linalg.norm(X.T @ X - np.eye(X.shape[1])))
                    ),
                )
            )
        (OUTPUT / "preprocessing_validation.json").write_text(
            json.dumps(transformation, indent=2)
        )

    out = dict(
        checked_final_weighted_results=count,
        checked_restart_weight_vectors=restart_count,
        unique_problem_solver_settings=len(cache),
        max_ratio_absolute_discrepancy=maxabs,
        max_ratio_relative_discrepancy=maxrel,
        ratio_reproduction_issues=issues,
        Fourier_transformations_reproduced=len(transformation),
        qualification="Binary64 truncated-SVD fits evaluated on the stored supports and weights.",
    )
    (OUTPUT / "numerical_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    assert not issues


if __name__ == "__main__":
    main()
