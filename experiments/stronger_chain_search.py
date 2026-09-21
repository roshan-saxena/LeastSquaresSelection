from experiments.paths import OUTPUT
from pathlib import Path
import sys, json, time, itertools, hashlib
import numpy as np
from scipy.optimize import differential_evolution
from spancancel.engine import Problem, fit, seed_for
from spancancel.datasets_local import load_input
from .run_experiments import Log
from threadpoolctl import threadpool_limits


def main():
    sys.dont_write_bytecode = True

    ROOT = Path(__file__).resolve().parents[1]
    protocol = dict(
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        reason="Run differential evolution on every size-four support of known_chain1.",
        instance="known_chain1",
        supports="all five supports of size4",
        reps=2,
        bounds=[-14, 14],
        logits="three free logits; final logit fixed zero",
        initial_population=24,
        iterations=200,
        tol=1e-10,
        atol=1e-12,
        polish=True,
        qualification="Bounded differential evolution over the enumerated supports",
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (OUTPUT / "chain_followup_protocol.json").write_text(json.dumps(protocol, indent=2))
    log = Log("chain_followup")
    z = load_input("known_chain1")
    P = Problem(z["X"], z["y"])
    R = np.linalg.qr(P.X, mode="r")

    def weights(v):
        vv = np.r_[v, 0]
        a = np.exp(vv - vv.max())
        return a / a.sum()

    with threadpool_limits(1):
        for S in itertools.combinations(range(P.N), 4):
            for rep in range(2):
                key = f"chain_DE_{S}_{rep}"
                seed = seed_for(key)
                rng = np.random.default_rng(seed)
                pop = rng.uniform(-14, 14, (24, 3))
                t = time.perf_counter()

                def fun(v):
                    return (
                        1
                        + float(
                            np.linalg.norm(R @ (fit(P.X, P.y, S, weights(v)) - P.w))
                            ** 2
                        )
                        / P.L
                    )

                rr = differential_evolution(
                    fun,
                    [(-14, 14)] * 3,
                    seed=rng,
                    init=pop,
                    maxiter=200,
                    tol=1e-10,
                    atol=1e-12,
                    polish=True,
                    workers=1,
                    updating="immediate",
                )
                out = P.evaluate(S, weights(rr.x))
                out.update(
                    data_id="known_chain1",
                    dataset="chain1",
                    d=P.d,
                    N=P.N,
                    n=4,
                    method="differential-evolution",
                    protocol="stronger_search",
                    replicate=rep,
                    seed=seed,
                    status="ok",
                    optimizer_success=bool(rr.success),
                    message=str(rr.message),
                    nfev=int(rr.nfev),
                    nit=int(rr.nit),
                    seconds=time.perf_counter() - t,
                    initial_population=pop.tolist(),
                    final_population=rr.population.tolist(),
                    bounds=[-14, 14],
                )
                log.save(key, out)
        print(
            "Stronger chain search best",
            min(r["ratio"] for r in log.done.values()),
            flush=True,
        )


if __name__ == "__main__":
    main()
