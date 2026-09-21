from experiments.paths import OUTPUT
from pathlib import Path
import sys, json, time, hashlib

sys.dont_write_bytecode = True
import numpy as np
from spancancel.engine import seed_for
from .run_experiments import Log, enumerated_search
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
from . import constructions as adv


def main():
    log = Log("known_constructions")
    cases = []
    for label, dims, n, value in [
        ("block11", [1, 1], 3, 1.5),
        ("block111", [1, 1, 1], 4, 5 / 3),
        ("block21", [2, 1], 4, 5 / 3),
        ("block111n5", [1, 1, 1], 5, 4 / 3),
        ("block1111", [1, 1, 1, 1], 6, 1.5),
        ("block22", [2, 2], 5, 2.0),
    ]:
        X, r = adv.blocks(dims)
        cases.append((label, X, r, n, value))
    for seed in (0, 1, 2):
        X, r = adv.chain(seed)
        cases.append((f"chain{seed}", X, r, 4, None))
    for omega in (0.05, 0.25):
        X, r = adv.merge(2, omega)
        cases.append((f"overlap{omega}", X, r, 3, 3 / (1 - omega)))
    rng = np.random.default_rng(7)
    for sigma in (0.02, 0.1):
        X, r = adv.blocks([2, 1])
        X += sigma * rng.standard_normal(X.shape)
        X, _ = adv.whiten(X)
        r -= X @ np.linalg.lstsq(X, r, rcond=None)[0]
        cases.append((f"noise{sigma}", X, r, 4, None))
    for label, dims, n, value in [
        ("minimal_d8n10", [3, 3, 2], 10, 13 / 7),
        ("global_block_d8n10", [2, 2, 2, 2], 10, 2.0),
    ]:
        X, r = adv.blocks(dims)
        cases.append((label, X, r, n, value))
    for label, X, r, n, value in cases:
        key = "known_" + label
        y = X @ (30 * np.ones(X.shape[1])) - r
        P = log.problem(
            key,
            X,
            y,
            dict(
                family=label,
                source="structured generators in experiments/constructions.py",
                shift="30 times all-ones vector, matching original source",
                analytical_reference=value,
            ),
        )
        out = log.run(
            key + ":SpanCancel",
            P,
            n,
            "SpanCancel",
            seed_for(key),
            dict(
                dataset=label,
                data_id=key,
                protocol="known_construction",
                analytical_reference=value,
                replicate=0,
            ),
        )
        en = enumerated_search(
            log, P, n, key, seed_for(key, "enum"), full_enumeration=True
        )
        print(
            label,
            "SC",
            out["ratio"],
            "enumerated best",
            en["best_ratio"],
            "reference",
            value,
            "supports",
            en["supports_visited"],
            flush=True,
        )


if __name__ == "__main__":
    cfg = dict(
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        reason="Evaluate the original structured constructions and d=8, n=10 block cases; enumerate every qualifying support size.",
        cases="13 source Table4 instances plus two d8,n10 block constructions",
        continuous_search="uniform plus8 random NM restarts800iterations; exact recovery LP when possible; discrete enumeration complete but continuous optima are local",
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (OUTPUT / "known_constructions_protocol.json").write_text(json.dumps(cfg, indent=2))
    with threadpool_limits(1):
        main()
