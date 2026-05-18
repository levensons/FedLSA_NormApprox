"""
For each parallelism level, time the *internal* components of one
fedlsa_train round, inside each worker, and report. This pinpoints
which component (numpy beta sampling, JIT compute, sampling, ...) is
the one that scales badly under contention.

Output per worker:
  - sample_round_Ab    ms/round
  - bootstrap weights  ms/round
  - JIT _fedlsa_one_round ms/round
  - combined           ms/round
"""
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np


GARNET_CFG = dict(
    ns=30, na=2, b=2, p=5, gamma=0.95,
    nenvs=5,
    heteregoneity_kern=0.02,
    heteregoneity_reward=0.001,
    gen_seed=10,
)
LOCAL_STEPS   = 20
NUM_BOOTSTRAP = 256
N_TRAJ_TOTAL  = 128
NREP          = 50


def _worker(args):
    n_traj_batch, sample_seed, key_seed = args

    import jax
    import jax.numpy as jnp
    from garnet import Garnet
    from fedlsa import (
        _fedlsa_one_round,
        _sample_round_Ab,
        _sample_bootstrap_weights,
    )

    g = Garnet(**GARNET_CFG)
    g.set_sample_rng(np.random.default_rng(sample_seed))

    R       = n_traj_batch
    B       = NUM_BOOTSTRAP
    K       = 1 + B
    D       = g.p
    H_t     = LOCAL_STEPS
    N       = g.nenvs
    boot_rng = np.random.default_rng(key_seed)

    theta = jnp.zeros((R, K, D))

    # Warm up JIT
    A_round, b_round = _sample_round_Ab(g, H_t=H_t, R=R)
    main_w = jnp.ones((H_t, R, 1, N))
    boot_w = _sample_bootstrap_weights(boot_rng, (H_t, R, B, N))
    weights_round = jnp.concatenate([main_w, boot_w], axis=2)
    alpha_t0 = jnp.asarray(1.0)
    theta = _fedlsa_one_round(theta, A_round, b_round, alpha_t0, weights_round)
    jax.block_until_ready(theta)

    # ----- timed components -----
    # (a) sample
    t0 = time.time()
    for _ in range(NREP):
        A_r, b_r = _sample_round_Ab(g, H_t=H_t, R=R)
    jax.block_until_ready(A_r)
    sample_t = (time.time() - t0) / NREP

    # (b) bootstrap weights
    t0 = time.time()
    for _ in range(NREP):
        bw = _sample_bootstrap_weights(boot_rng, (H_t, R, B, N))
    jax.block_until_ready(bw)
    weights_t = (time.time() - t0) / NREP

    # (c) JIT call alone (reuse pre-built tensors)
    t0 = time.time()
    for _ in range(NREP):
        theta = _fedlsa_one_round(
            theta, A_round, b_round, alpha_t0, weights_round
        )
    jax.block_until_ready(theta)
    jit_t = (time.time() - t0) / NREP

    # (d) realistic combined
    t0 = time.time()
    for t_idx in range(NREP):
        A_r, b_r = _sample_round_Ab(g, H_t=H_t, R=R)
        bw_l = _sample_bootstrap_weights(boot_rng, (H_t, R, B, N))
        wr = jnp.concatenate([main_w, bw_l], axis=2)
        alpha_t = jnp.asarray(1.0 * (t_idx + 1.0) ** (-0.6))
        theta = _fedlsa_one_round(theta, A_r, b_r, alpha_t, wr)
    jax.block_until_ready(theta)
    combined_t = (time.time() - t0) / NREP

    return {
        "n_traj_batch": n_traj_batch,
        "sample_ms":    1000 * sample_t,
        "weights_ms":   1000 * weights_t,
        "jit_ms":       1000 * jit_t,
        "combined_ms":  1000 * combined_t,
    }


def _split_evenly(total, n):
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)
            if base + (1 if i < rem else 0) > 0]


def run(num_workers):
    print("\n" + "=" * 64)
    print(f"num_workers = {num_workers}, n_traj_batch = {N_TRAJ_TOTAL // num_workers}")
    print("=" * 64)

    batch_sizes = _split_evenly(N_TRAJ_TOTAL, num_workers)
    args_list = [(b, 33 + i, 1007 + i) for i, b in enumerate(batch_sizes)]

    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=spawn_ctx) as ex:
        results = list(ex.map(_worker, args_list))

    # Average across workers
    fields = ("sample_ms", "weights_ms", "jit_ms", "combined_ms")
    print(f"  worker  {'sample':>10}  {'weights':>10}  {'jit':>10}  "
          f"{'combined':>10}")
    for i, r in enumerate(results):
        print(f"  {i:>6d}  {r['sample_ms']:>9.2f}  {r['weights_ms']:>9.2f}  "
              f"{r['jit_ms']:>9.2f}  {r['combined_ms']:>9.2f}")
    means = {f: np.mean([r[f] for r in results]) for f in fields}
    print(f"  {'mean':>6s}  {means['sample_ms']:>9.2f}  "
          f"{means['weights_ms']:>9.2f}  {means['jit_ms']:>9.2f}  "
          f"{means['combined_ms']:>9.2f}")
    return means


if __name__ == "__main__":
    grand_t0 = time.time()
    one    = run(num_workers=1)
    eight  = run(num_workers=8)

    print("\n" + "=" * 64)
    print("Slowdown (8 workers / 1 worker), per component")
    print("=" * 64)
    print(f"  {'component':>14}  {'1w':>10}  {'8w':>10}  {'8w/1w':>10}")
    for f in ("sample_ms", "weights_ms", "jit_ms", "combined_ms"):
        ratio = eight[f] / one[f] if one[f] > 0 else float("nan")
        print(f"  {f:>14s}  {one[f]:>9.2f}  {eight[f]:>9.2f}  {ratio:>9.2f}x")

    print(f"\nTotal investigation time: {time.time() - grand_t0:.1f}s")
