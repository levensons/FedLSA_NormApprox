"""
Diagnose parallelization bottlenecks in _run_trajectory_batch.

KEY FIX vs the previous version: thread-limit env vars are set in the
PARENT process before spawning workers. Spawned children inherit the
parent's env, so when each child re-imports numpy/JAX at top of the
module, OpenBLAS / MKL / Eigen pick up the limited thread count at
LIBRARY LOAD time. Setting env vars inside _worker is too late — by
then numpy is already imported (top-level) and BLAS is already
initialised with default (12-thread) pools.

Two investigations:
  B. Parallel scaling with default thread settings
  C. Parallel scaling with 1 BLAS / Eigen thread per worker
"""
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np


# ===================================================================
# Common config — production-shape per worker
# ===================================================================
GARNET_CFG = dict(
    ns=30, na=2, b=2, p=5, gamma=0.95,
    nenvs=5,
    heteregoneity_kern=0.02,
    heteregoneity_reward=0.001,
    gen_seed=10,
)
T              = 300       # smaller than before for faster turnaround
LOCAL_STEPS    = 20
NUM_BOOTSTRAP  = 256
N_TRAJ_TOTAL   = 128
WORKER_COUNTS  = (1, 4, 8)


def make_kwargs(n_traj_batch, sample_seed, key_seed):
    return dict(
        garnet_cfg=GARNET_CFG,
        sample_seed=sample_seed,
        u=None,                 # filled in inside the worker
        num_rounds=T,
        local_steps=LOCAL_STEPS,
        alpha=1.0,
        n_traj_batch=n_traj_batch,
        t0=1.0,
        gamma_eta=0.6,
        gamma_H=0.0,
        num_bootstrap=NUM_BOOTSTRAP,
        ci_alpha=0.05,
        key_seed=key_seed,
    )


# ===================================================================
# Worker — reports thread counts so we can verify pinning took effect
# ===================================================================
def _worker(args):
    import os, time
    n_traj_batch, sample_seed, key_seed = args

    # Probe what BLAS / OMP / XLA actually got configured with.
    probes = {
        "OMP_NUM_THREADS":      os.environ.get("OMP_NUM_THREADS",      "<unset>"),
        "MKL_NUM_THREADS":      os.environ.get("MKL_NUM_THREADS",      "<unset>"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "<unset>"),
        "XLA_FLAGS":            os.environ.get("XLA_FLAGS",            "<unset>"),
    }

    # numpy was imported at module top — already constrained or not.
    import jax
    import jax.numpy as jnp
    from inference import sample_unit_vector
    from run_na_fedlsa import _run_trajectory_batch

    u = np.asarray(sample_unit_vector(jax.random.key(99), GARNET_CFG["p"]))
    kw = make_kwargs(n_traj_batch, sample_seed, key_seed)
    kw["u"] = u

    t0 = time.time()
    out = _run_trajectory_batch(kw)
    elapsed = time.time() - t0
    return elapsed, n_traj_batch, probes


def _split_evenly(total, n):
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)
            if base + (1 if i < rem else 0) > 0]


# ===================================================================
# Parent-side env var control
# ===================================================================
THREAD_PIN_VARS = {
    "OMP_NUM_THREADS":      "1",
    "MKL_NUM_THREADS":      "1",
    "OPENBLAS_NUM_THREADS": "1",
    "XLA_FLAGS":            "--xla_cpu_multi_thread_eigen=false",
}


def _set_env(pin: bool, saved: dict):
    """Set or restore env vars in the parent process."""
    if pin:
        for k, v in THREAD_PIN_VARS.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v
    else:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        saved.clear()


def run_scaling(label: str, pin: bool):
    saved = {}
    _set_env(pin, saved)
    print("\n" + "=" * 64)
    print(label)
    print(f"   total trajectories = {N_TRAJ_TOTAL}, T={T}, B={NUM_BOOTSTRAP}")
    print("=" * 64)
    print(f"  {'workers':>8}  {'per-worker':>11}  {'wall':>8}  "
          f"{'cpu-busy':>10}  {'speedup':>9}")

    spawn_ctx = mp.get_context("spawn")
    base_wall = None
    last_probes = None

    try:
        for nw in WORKER_COUNTS:
            batch_sizes = _split_evenly(N_TRAJ_TOTAL, nw)
            args_list = [
                (b, 33 + i, 1007 + i)
                for i, b in enumerate(batch_sizes)
            ]
            t0 = time.time()
            with ProcessPoolExecutor(
                max_workers=nw, mp_context=spawn_ctx
            ) as ex:
                results = list(ex.map(_worker, args_list))
            wall = time.time() - t0

            per_batch = [r[0] for r in results]
            max_pb = max(per_batch)
            sum_pb = sum(per_batch)
            cpu_busy = sum_pb / wall
            if base_wall is None:
                base_wall = wall
            speedup = base_wall / wall

            print(f"  {nw:>8d}  {max_pb:>10.2f}s  {wall:>8.2f}s  "
                  f"{cpu_busy:>10.2f}x  {speedup:>9.2f}x")
            last_probes = results[0][2]
    finally:
        # Restore env so the next call has a clean slate
        _set_env(False, saved)

    if last_probes:
        print("  worker env probes:")
        for k, v in last_probes.items():
            print(f"    {k:22s} = {v}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    grand_t0 = time.time()
    run_scaling("B. Default thread settings", pin=False)
    run_scaling("C. With thread-pin (1 thread/worker, set in parent)", pin=True)
    print(f"\nTotal investigation time: {time.time() - grand_t0:.1f}s")
