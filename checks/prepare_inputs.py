"""Prepare dataset inputs using the paper’s preprocessing protocols."""

from experiments.paths import OUTPUT
from pathlib import Path
import argparse, json, sys, hashlib
import shutil
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from spancancel.datasets_local import controlled, holdout, _sm, load_input


def main():
    sys.dont_write_bytecode = True

    ROOT = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument(
        "--download-california",
        action="store_true",
        help="Fetch California Housing through scikit-learn into results/generated/download_cache",
    )
    args = p.parse_args()
    inputs = OUTPUT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / "california.npz").exists():
        if not args.download_california:
            raise SystemExit(
                "California data missing. Explicitly allow its upstream download with --download-california."
            )
        a = datasets.fetch_california_housing(
            data_home=str(OUTPUT / "download_cache"),
            download_if_missing=True,
        )
        np.savez_compressed(raw / "california.npz", X=a.data, y=a.target)
    with threadpool_limits(1):
        for nm, X, y, meta in controlled():
            np.savez_compressed(inputs / f"controlled_{nm}.npz", X=X, y=y)
        for M, reps in [(50, 10), (100, 10), (200, 5)]:
            for seed in range(reps):
                X, y, Xtest, ytest, meta = holdout(
                    "california", seed, 12000 if M == 200 else 6000, M
                )
                np.savez_compressed(
                    inputs / f"rff_{M}_{seed}.npz",
                    X=X,
                    y=y,
                    Xtest=Xtest,
                    ytest=ytest,
                )
        shutil.copy2(inputs / "rff_50_0.npz", inputs / "large_common_rff50.npz")
        # The preserved Figure 2 protocol intentionally standardizes before splitting.
        for nm in ["diabetes", "california", "wine", "engel", "stackloss"]:
            if nm == "diabetes":
                a = datasets.load_diabetes()
                X0, y0 = a.data, a.target
            elif nm == "wine":
                a = datasets.load_wine()
                X0, y0 = a.data, a.target
            elif nm == "california":
                a = np.load(raw / "california.npz")
                X0, y0 = a["X"], a["y"]
            else:
                X0, y0 = _sm(nm)
            m = np.isfinite(X0).all(1) & np.isfinite(y0)
            X0, y0 = X0[m], y0[m]
            X0 = np.column_stack([np.ones(len(y0)), StandardScaler().fit_transform(X0)])
            y0 = (y0 - y0.mean()) / (y0.std() + 1e-12)
            for seed in range(5):
                Xtr, Xte, ytr, yte = train_test_split(
                    X0, y0, test_size=0.4, random_state=seed
                )
                np.savez_compressed(
                    ROOT / "experiments/figure2" / f"input_{nm}_{seed}.npz",
                    Xtr=Xtr,
                    Xte=Xte,
                    ytr=ytr,
                    yte=yte,
                )

    def digest(a):
        a = np.ascontiguousarray(a)
        return hashlib.sha256(
            str(a.shape).encode() + a.dtype.str.encode() + a.tobytes()
        ).hexdigest()

    checks = []
    for record in json.loads((ROOT / "data/input_manifest.json").read_text()):
        z = (
            load_input(record["input_id"])
            if record["collection"] == "experiments"
            else np.load(
                ROOT / "experiments/figure2" / f'{record["input_id"]}.npz',
                allow_pickle=False,
            )
        )
        checks.append(
            dict(
                input_id=record["input_id"],
                array_hashes_match=all(
                    digest(z[k]) == h for k, h in record["arrays"].items()
                ),
            )
        )
    (OUTPUT / "input_validation.json").write_text(json.dumps(checks, indent=2))
    bad = [r["input_id"] for r in checks if not r["array_hashes_match"]]
    print("Inputs checked:", len(checks), "bitwise array mismatches:", len(bad))
    if bad:
        raise SystemExit(
            "Investigate platform or source differences before trusting reproduction: "
            + str(bad)
        )


if __name__ == "__main__":
    main()
