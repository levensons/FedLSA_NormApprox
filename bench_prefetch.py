"""
Benchmark prefetch_weights on/off for fedlsa_train at production shape.

Runs two warmups per setting then 3 timed runs. Compares median wall time
and verifies bitwise equivalence of outputs.
"""
# import os
# os.environ["JAX_PLATFORMS"] = "cpu"

import time
import statistics

import numpy as np
import jax

from garnet import Garnet
from fedlsa import FedLSAConfig, fedlsa_train

GARNET_CFG = dict(
    ns=30, na=2, b=2, p=5, gamma=0.95,
    nenvs=5,
    heteregoneity_kern=0.02,
    heteregoneity_reward=0.001,
    gen_seed=10,
)
T       = 300
R       = 16
B_BOOT  = 256
N_RUNS  = 3


def run(prefetch: bool):
    g = Garnet(**GARNET_CFG)
    g.set_sample_rng(np.random.default_rng(33))
    cfg = FedLSAConfig(
        num_rounds=T, local_steps=20, alpha=1.0,
        n_traj=R, t0=1.0, gamma_eta=0.6, gamma_H=0.0,
    )
    t0 = time.perf_counter()
    out = fedlsa_train(
        g, cfg,
        num_bootstrap=B_BOOT, boot_seed=1007,
        progress=False,
        prefetch_weights=prefetch,
    )
    out["theta_final"].block_until_ready()
    out["theta_boot_final"].block_until_ready()
    return time.perf_counter() - t0, out


def bench(prefetch: bool):
    run(prefetch); run(prefetch)  # warmup (JIT compile + caches)
    times = [run(prefetch)[0] for _ in range(N_RUNS)]
    return statistics.median(times), times


print(f"Config: T={T}  R={R}  B={B_BOOT}  N={GARNET_CFG['nenvs']}  runs={N_RUNS}")
print()

t_off, all_off = bench(prefetch=False)
t_on,  all_on  = bench(prefetch=True)

print(f"{'mode':>12s}  {'median (s)':>11s}  {'all runs (s)':>30s}")
print("-" * 58)
print(f"{'no prefetch':>12s}  {t_off:>11.2f}  {str([f'{x:.2f}' for x in all_off]):>30s}")
print(f"{'prefetch':>12s}  {t_on:>11.2f}  {str([f'{x:.2f}' for x in all_on]):>30s}")
print()
print(f"Speedup: {t_off / t_on:.3f}x  (saved {t_off - t_on:.2f}s per run, "
      f"{(1 - t_on/t_off)*100:.1f}%)")

# Bitwise equivalence
_, out_off = run(prefetch=False)
_, out_on  = run(prefetch=True)
for k in ("theta_final", "theta_boot_final"):
    d = float(np.max(np.abs(np.asarray(out_off[k]) - np.asarray(out_on[k]))))
    print(f"  max|off - on|[{k}] = {d:.2e}")
