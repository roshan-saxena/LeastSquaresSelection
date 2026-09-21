from pathlib import Path
import json, os
from functools import lru_cache
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.linalg import block_diag, null_space
import statsmodels.api as sm


def _sm(nm):
    t = (
        getattr(sm.datasets, nm)
        .load_pandas()
        .data.select_dtypes(include=[np.number])
        .dropna()
    )
    return t.iloc[:, 1:].to_numpy(float), t.iloc[:, 0].to_numpy(float)


def reals():
    """Controlled low-dimensional regression tasks used by the training tables."""
    out = []
    for nm, ld, k in [
        ("diabetes", datasets.load_diabetes, 5),
        ("wine", datasets.load_wine, 5),
        ("breast", datasets.load_breast_cancer, 6),
    ]:
        d = ld()
        out.append(
            (nm, np.asarray(d.data, float)[:, :k], np.asarray(d.target, float).ravel())
        )
    for nm in [
        "longley",
        "stackloss",
        "ccard",
        "copper",
        "statecrime",
        "scotland",
        "grunfeld",
    ]:
        out.append((nm, *_sm(nm)))
    return out


def prep(X, y, cap=80):
    """Standardize an intercept model after deterministic evenly spaced capping."""
    m = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[m], y[m]
    if len(y) > cap:
        idx = np.linspace(0, len(y) - 1, cap).round().astype(int)
        X, y = X[idx], y[idx]
    X = np.column_stack([np.ones(len(y)), StandardScaler().fit_transform(X)])
    return X, (y - y.mean()) / (y.std() + 1e-12)


def controlled():
    for nm, X0, y0 in reals():
        X, y = prep(X0, y0)
        yield nm, X, y, dict(
            original_N=len(y0),
            original_features=X0.shape[1],
            retained_N=len(y),
            features_before_intercept=X.shape[1] - 1,
            cap=80,
            row_selection="deterministic evenly spaced",
            target=(
                "scalar class label"
                if nm in ("wine", "breast")
                else "regression response"
            ),
            feature_preprocessing="population mean and standard deviation on retained training-only dataset",
            target_preprocessing="(y-mean)/(population SD+1e-12)",
            intercept=True,
        )


def whiten_svd(X):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if s[-1] <= 1e-12 * s[0]:
        raise ValueError("Whitening requires full column rank")
    W = (Vt.T / s) @ Vt
    return X @ W, W


def raw_dataset(name):
    if name == "diabetes":
        a = datasets.load_diabetes(scaled=False)
        return np.asarray(a.data, float), np.asarray(a.target, float)
    if name == "california":

        p = Path(__file__).resolve().parents[1] / "data/raw/california.npz"
        if not p.exists():
            raise FileNotFoundError(
                "Run python -m checks.prepare_inputs --download-california first."
            )
        z = np.load(p)
        return z["X"], z["y"]
    return _sm(name)


def holdout(name, seed, Ncap=None, rff=None):
    X, y = raw_dataset(name)
    origN = len(y)
    mask = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    retained_original = np.flatnonzero(mask)
    if Ncap and len(y) > Ncap:
        pool = 12000 if name == "california" and rff else Ncap
        ix = np.random.default_rng(0).choice(len(y), pool, replace=False)[:Ncap]
        X, y = X[ix], y[ix]
        retained_original = retained_original[ix]
    inds = np.arange(len(y))
    train, test = train_test_split(inds, test_size=0.4, random_state=seed)
    scaler = StandardScaler().fit(X[train])
    Xt = scaler.transform(X[train])
    Xe = scaler.transform(X[test])
    ym = float(y[train].mean())
    ys = float(y[train].std())
    yt = (y[train] - ym) / (ys + 1e-12)
    ye = (y[test] - ym) / (ys + 1e-12)
    meta = dict(
        dataset=name,
        original_N=origN,
        original_features=X.shape[1],
        retained_features=X.shape[1],
        removed_nonfinite=int(origN - mask.sum()),
        retained_original_indices=retained_original.tolist(),
        retained_N=len(y),
        train_indices=train.tolist(),
        test_indices=test.tolist(),
        feature_mean=scaler.mean_.tolist(),
        feature_scale=scaler.scale_.tolist(),
        target_mean=ym,
        target_sd=ys,
        intercept=True,
        fit_preprocessing_on="training split only",
        split_seed=seed,
        test_fraction=0.4,
    )
    if rff:
        rng = np.random.default_rng(7)
        omega = rng.normal(size=(Xt.shape[1], rff)) * np.sqrt(0.2)
        phase = rng.uniform(0, 2 * np.pi, rff)
        Xt = np.sqrt(2 / rff) * np.cos(Xt @ omega + phase)
        Xe = np.sqrt(2 / rff) * np.cos(Xe @ omega + phase)
        meta.update(
            rff_count=rff,
            rff_seed=7,
            kernel="exp(-0.1*||x-z||^2)",
            gamma=0.1,
            omega_distribution="N(0,0.2 I)",
            phase_distribution="Uniform(0,2pi)",
            normalization="sqrt(2/M)",
        )
    Xt = np.c_[np.ones(len(yt)), Xt]
    Xe = np.c_[np.ones(len(ye)), Xe]
    if rff:
        Xt, W = whiten_svd(Xt)
        Xe = Xe @ W
        meta["whitening"] = "training SVD; same invertible W on test"
        meta["whitening_matrix"] = W.tolist()
    return Xt, yt, Xe, ye, meta


def simplex(d):
    return null_space(np.ones((1, d + 1)))


def blocks(dims, seed=0, duplicate=1, noise=0.0):
    X = block_diag(*[simplex(s) for s in dims])
    r = np.concatenate([np.ones(s + 1) * np.sqrt(1 / (s * (s + 1))) for s in dims])
    if duplicate > 1:
        X = np.repeat(X, duplicate, axis=0) / np.sqrt(duplicate)
        r = np.repeat(r, duplicate) / np.sqrt(duplicate)
    if noise:
        X = X + np.random.default_rng(seed).normal(scale=noise, size=X.shape)
        X, _ = whiten_svd(X)
        r = r - X @ np.linalg.lstsq(X, r, rcond=1e-12)[0]
    # Generic shift avoids rank-deficient supports accidentally containing w*.
    w = np.random.default_rng(seed + 981).normal(size=X.shape[1])
    w = 100 * w / np.linalg.norm(w)
    return (
        X,
        X @ w - r,
        dict(
            family="noisy_blocks" if noise else "blocks",
            dims=list(dims),
            duplicate=duplicate,
            noise=noise,
            shift=w.tolist(),
        ),
    )


def synthetic(family, d, N, seed, kappa=1.0):
    rng = np.random.default_rng(seed)
    if family == "overlap":
        # Two triangles share e2; append stationary duplicate/split rays to reach N.
        V = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1], [-1, -1, 0], [0, -1, -1]], float)
        if d > 3:
            V = block_diag(V, *[np.array([[1.0], [-1.0]]) for _ in range(d - 3)])
        while len(V) < N:
            j = int(rng.integers(len(V)))
            half = V[j] / 2
            V[j] = half
            V = np.vstack([V, half])
        rr = rng.uniform(0.3, 2, len(V)) * rng.choice([-1, 1], len(V))
        X = V / rr[:, None]
        X, _ = whiten_svd(X)
        r = rr
    else:
        X = rng.normal(size=(N, d))
        if family == "correlated":
            X = 0.95 * rng.normal(size=(N, 1)) + np.sqrt(1 - 0.95**2) * X
        X, _ = whiten_svd(X)
        base = rng.normal(size=N)
        if family == "structured_residual":
            base = np.where(np.arange(N) % 3 == 0, 3.0, -0.3)
        r = base - X @ (X.T @ base)
    r = r / np.linalg.norm(r)
    if family == "ill_conditioned" or kappa != 1.0:
        Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
        X = X @ Q @ np.diag(np.geomspace(1, 1 / kappa, d))
    w = rng.normal(size=d) * 20
    return (
        X,
        X @ w - r,
        dict(family=family, seed=seed, requested_N=N, kappa_requested=kappa),
    )


def frontier(d, n):
    if n < d:
        return float("inf")
    if n == d:
        return float(d + 1)
    if n >= 2 * d:
        return 1.0
    k = n - d
    v = []
    for g in range(k + 1, d + 1):
        q, r = divmod(d, g)
        den = r / (q + 1) + (g - r) / q
        v.append(1 + (g - k) / den)
    return max(v)


@lru_cache(maxsize=1)
def input_metadata():
    """Generation and preprocessing records, keyed by input identifier."""
    path = Path(__file__).resolve().parents[1] / "data/metadata.json"
    return json.loads(path.read_text())


def load_input(data_id):
    """Load a saved synthetic input or a locally reconstructed public dataset."""
    root = Path(__file__).resolve().parents[1]
    if os.environ.get("MLWA_RESULTS_DIR"):
        path = (
            Path(os.environ["MLWA_RESULTS_DIR"]).resolve().parent
            / "data"
            / f"{data_id}.npz"
        )
        if path.exists():
            with np.load(path, allow_pickle=False) as arrays:
                return {key: arrays[key] for key in arrays.files}
    generated = (
        Path(os.environ.get("MLWA_OUTPUT_DIR", root / "results/generated"))
        / "inputs"
        / f"{data_id}.npz"
    )
    if generated.exists():
        with np.load(generated, allow_pickle=False) as arrays:
            return {key: arrays[key] for key in arrays.files}
    prefix = data_id + "/"
    with np.load(root / "data/inputs.npz", allow_pickle=False) as arrays:
        result = {
            key[len(prefix) :]: arrays[key]
            for key in arrays.files
            if key.startswith(prefix)
        }
    if not result:
        raise FileNotFoundError(
            f"Input {data_id!r} is missing; run the README verification command to prepare public datasets."
        )
    return result
