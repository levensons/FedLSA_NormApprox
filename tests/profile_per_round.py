"""
Per-round breakdown of fedlsa_train + N-scaling sweep.

A. At production shape, time each component of one round in isolation:
     1. _sample_round_Ab           (numpy sampling + transpose + transfer to JAX)
     2. _sample_bootstrap_weights  (numpy Beta + transfer to JAX)
     3. _fedlsa_one_round (main, K=1)
     4. _fedlsa_one_round (bootstrap, chunked over K)
     5. _fedlsa_one_round (bootstrap, single K=B call) — comparison

B. Sweep N = nenvs to see how each component scales with the number of
   federated agents (R, B, H, D fixed at production values).

Run on one process (no ProcessPoolExecutor) so numbers reflect the
pure single-worker cost of each component.
"""
import time
import statistics

import numpy as np
import jax
import jax.numpy as jnp

from garnet import Garnet
from fedlsa import (
    _fedlsa_one_round,
    _sample_round_Ab,
    _sample_bootstrap_weights,
)


BASE_GARNET_CFG = dict(
    ns=30, na=2, b=2, p=5, gamma=0.95,
    nenvs=5,
    heteregoneity_kern=0.02,
    heteregoneity_reward=0.001,
    gen_seed=10,
)
R          = 16    # per-worker n_traj (128 traj / 8 workers)
H          = 20    # local steps
B          = 256   # bootstrap replicates
BOOT_CHUNK = 32
N_ITERS    = 30


def block(x):
    if isinstance(x, (list, tuple)):
        for xx in x:
            block(xx)
        return
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()


def time_fn(fn, n=N_ITERS):
    """Warm up twice, then return median wall time over n calls."""
    block(fn()); block(fn())
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        block(fn())
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def make_callables(garnet, R_, B_, boot_chunk):
    D = garnet.p
    N = garnet.nenvs
    garnet.set_sample_rng(np.random.default_rng(33))
    boot_rng = np.random.default_rng(0)
    alpha_t_arr = jnp.asarray(0.5)

    A_round, b_round = _sample_round_Ab(garnet, H_t=H, R=R_)
    boot_w = _sample_bootstrap_weights(boot_rng, (H, R_, B_, N))
    block((A_round, b_round, boot_w))

    theta_main = jnp.zeros((R_, 1, D))
    theta_boot = jnp.zeros((R_, B_, D))
    main_w     = jnp.ones((H, R_, 1, N))

    def sample_Ab():
        return _sample_round_Ab(garnet, H_t=H, R=R_)

    def sample_weights():
        return _sample_bootstrap_weights(boot_rng, (H, R_, B_, N))

    def main_update():
        return _fedlsa_one_round(
            theta_main, A_round, b_round, alpha_t_arr, main_w
        )

    def boot_chunked():
        chunks = []
        for k0 in range(0, B_, boot_chunk):
            k1 = min(k0 + boot_chunk, B_)
            chunks.append(_fedlsa_one_round(
                theta_boot[:, k0:k1, :],
                A_round, b_round, alpha_t_arr,
                boot_w[:, :, k0:k1, :],
            ))
        return jnp.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]

    def boot_full():
        return _fedlsa_one_round(
            theta_boot, A_round, b_round, alpha_t_arr, boot_w
        )

    return {
        "sample_Ab":       sample_Ab,
        "sample_weights":  sample_weights,
        "main_update":     main_update,
        "boot_chunked":    boot_chunked,
        "boot_full":       boot_full,
    }


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    section("A. Per-round breakdown at production shape")
    g = Garnet(**BASE_GARNET_CFG)
    print(f"  N={g.nenvs}  R={R}  B={B}  H={H}  D={g.p}  boot_chunk={BOOT_CHUNK}")
    fns = make_callables(g, R, B, BOOT_CHUNK)
    timings = {name: time_fn(fn) for name, fn in fns.items()}

    round_total = sum(timings[k] for k in
                      ("sample_Ab", "sample_weights", "main_update", "boot_chunked"))
    print(f"\n  {'component':>20s}  {'ms':>9s}  {'% of round':>11s}")
    print("  " + "-" * 44)
    for k in ("sample_Ab", "sample_weights", "main_update",
              "boot_chunked", "boot_full"):
        ms = timings[k] * 1e3
        pct = (timings[k] / round_total * 100.0
               if k != "boot_full" else float('nan'))
        pct_s = f"{pct:>10.1f}%" if k != "boot_full" else "  (n/a)"
        print(f"  {k:>20s}  {ms:>9.3f}  {pct_s:>11s}")
    print(f"  {'round total':>20s}  {round_total * 1e3:>9.3f}")
    print(f"\n  speedup of chunked vs full bootstrap: "
          f"{timings['boot_full'] / timings['boot_chunked']:.2f}x")

    section("B. Scaling with number of agents N (others fixed)")
    print(f"  R={R}  B={B}  H={H}  D={BASE_GARNET_CFG['p']}  "
          f"boot_chunk={BOOT_CHUNK}")
    print(f"\n  {'N':>4s}  {'sample_Ab':>10s}  {'sample_w':>10s}  "
          f"{'main':>8s}  {'boot_chnk':>10s}  {'boot_full':>10s}  "
          f"{'round':>8s}")
    print("  " + "-" * 70)
    for N in (1, 5, 10, 20, 50, 100):
        g = Garnet(**{**BASE_GARNET_CFG, "nenvs": N})
        fns = make_callables(g, R, B, BOOT_CHUNK)
        t = {name: time_fn(fn, n=15) * 1e3 for name, fn in fns.items()}
        round_ms = (t["sample_Ab"] + t["sample_weights"]
                    + t["main_update"] + t["boot_chunked"])
        print(f"  {N:>4d}  {t['sample_Ab']:>10.2f}  "
              f"{t['sample_weights']:>10.2f}  {t['main_update']:>8.2f}  "
              f"{t['boot_chunked']:>10.2f}  {t['boot_full']:>10.2f}  "
              f"{round_ms:>8.2f}")


if __name__ == "__main__":
    main()
