"""
Profile `_run_trajectory_batch` to find the bottleneck.

We separately time:
  1. Garnet construction
  2. fedlsa_train (whole)
  3. fedlsa_train internals: sample_A_and_b vs JIT call vs Python loop overhead
  4. multiplier_bootstrap_ci_jit (warm)
  5. Whole _run_trajectory_batch (warm — JIT compiled)
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import time
import math
import numpy as np
import jax
import jax.numpy as jnp

from garnet import Garnet
from fedlsa import (
    FedLSAConfig,
    fedlsa_train,
    _fedlsa_one_round,
    _sample_round_Ab,
    _sample_bootstrap_weights,
)
from inference import sample_unit_vector, multiplier_bootstrap_ci_jit
from run_na_fedlsa import _run_trajectory_batch


GARNET_CFG = dict(
    ns=30, na=2, b=2, p=5, gamma=0.95,
    nenvs=5,
    heteregoneity_kern=0.02,
    heteregoneity_reward=0.001,
    gen_seed=10,
)
T            = 1000
LOCAL_STEPS  = 20
N_TRAJ       = 20
NUM_BOOT     = 8
ALPHA        = 1.0
GAMMA_ETA    = 0.6


def banner(s):
    print(f"\n=== {s} ===", flush=True)


# ----------------------------------------------------
# 1. Garnet construction
# ----------------------------------------------------
banner("1. Garnet construction")
t0 = time.time()
g = Garnet(**GARNET_CFG)
print(f"  {time.time() - t0:.3f}s")

theta_star = np.asarray(g.thetalim)
u_np = np.asarray(sample_unit_vector(jax.random.key(99), g.p))
u = jnp.asarray(u_np)


# ----------------------------------------------------
# 2. Whole _run_trajectory_batch — cold and warm
# ----------------------------------------------------
def make_kw(T_):
    return dict(
        garnet_cfg=GARNET_CFG,
        sample_seed=33,
        u=u_np,
        snapshot_rounds=[T_],
        local_steps=LOCAL_STEPS,
        alpha=ALPHA,
        n_traj_batch=N_TRAJ,
        t0=1.0,
        gamma_eta=GAMMA_ETA,
        gamma_H=0.0,
        num_bootstrap=NUM_BOOT,
        key_seed=1007,
    )

banner("2a. _run_trajectory_batch — cold (T=20, JIT compile)")
t0 = time.time()
_ = _run_trajectory_batch(make_kw(20))
print(f"  {time.time() - t0:.3f}s")

banner("2b. _run_trajectory_batch — warm (T=20)")
t0 = time.time()
_ = _run_trajectory_batch(make_kw(20))
warm_T20 = time.time() - t0
print(f"  {warm_T20:.3f}s")

banner(f"2c. _run_trajectory_batch — warm (T={T})")
t0 = time.time()
_ = _run_trajectory_batch(make_kw(T))
warm_TBIG = time.time() - t0
print(f"  {warm_TBIG:.3f}s  ({1000*warm_TBIG/T:.2f} ms/round)")


# ----------------------------------------------------
# 3. Inside fedlsa_train — break the per-round cost down
# ----------------------------------------------------
banner("3. fedlsa_train per-round breakdown")

# Set up state once exactly like fedlsa_train does
g.set_sample_rng(np.random.default_rng(33))
R = N_TRAJ
B = NUM_BOOT
K = 1 + B
D = g.p
theta = jnp.zeros((R, K, D))
boot_rng = np.random.default_rng(1007)

# Pre-compute one round of data so JIT compiles before timing
H_t0 = LOCAL_STEPS
A_round, b_round = _sample_round_Ab(g, H_t=H_t0, R=R)
N_agents = A_round.shape[2]
main_w = jnp.ones((H_t0, R, 1, N_agents))
boot_w = _sample_bootstrap_weights(boot_rng, (H_t0, R, B, N_agents))
weights_round = jnp.concatenate([main_w, boot_w], axis=2)
alpha_t0 = jnp.asarray(ALPHA * (0 + 1.0) ** (-GAMMA_ETA))
theta = _fedlsa_one_round(theta, A_round, b_round, alpha_t0, weights_round)
jax.block_until_ready(theta)

NREP = 200

# (a) sample_A_and_b — pure NumPy on Garnet
t0 = time.time()
for _ in range(NREP):
    A_np_, b_np_ = g.sample_A_and_b(LOCAL_STEPS * R)
sample_only = (time.time() - t0) / NREP
print(f"  3a. garnet.sample_A_and_b only:        {1000*sample_only:8.3f} ms/round")

# (b) full _sample_round_Ab (sample + reshape + jnp.asarray transfer)
t0 = time.time()
for _ in range(NREP):
    A_r, b_r = _sample_round_Ab(g, H_t=LOCAL_STEPS, R=R)
jax.block_until_ready(A_r)
sample_reshape_transfer = (time.time() - t0) / NREP
print(f"  3b. _sample_round_Ab (incl. transfer): {1000*sample_reshape_transfer:8.3f} ms/round")

# (c) bootstrap weights generation
t0 = time.time()
for _ in range(NREP):
    bw = _sample_bootstrap_weights(boot_rng, (LOCAL_STEPS, R, B, N_agents))
jax.block_until_ready(bw)
weights_t = (time.time() - t0) / NREP
print(f"  3c. _sample_bootstrap_weights:         {1000*weights_t:8.3f} ms/round")

# (d) the JIT call alone
t0 = time.time()
for _ in range(NREP):
    theta = _fedlsa_one_round(theta, A_round, b_round, alpha_t0, weights_round)
jax.block_until_ready(theta)
jit_only = (time.time() - t0) / NREP
print(f"  3d. _fedlsa_one_round JIT call:        {1000*jit_only:8.3f} ms/round")

# (e) all three together (sample + weights + JIT) — the realistic per-round cost
t0 = time.time()
for t_idx in range(NREP):
    A_r, b_r = _sample_round_Ab(g, H_t=LOCAL_STEPS, R=R)
    main_w_local = jnp.ones((LOCAL_STEPS, R, 1, N_agents))
    bw = _sample_bootstrap_weights(boot_rng, (LOCAL_STEPS, R, B, N_agents))
    wr = jnp.concatenate([main_w_local, bw], axis=2)
    alpha_local = jnp.asarray(ALPHA * (t_idx + 1.0) ** (-GAMMA_ETA))
    theta = _fedlsa_one_round(theta, A_r, b_r, alpha_local, wr)
jax.block_until_ready(theta)
combined = (time.time() - t0) / NREP
print(f"  3e. combined per-round (realistic):    {1000*combined:8.3f} ms/round")


# ----------------------------------------------------
# 4. multiplier_bootstrap_ci_jit (post-processing)
# ----------------------------------------------------
banner("4. multiplier_bootstrap_ci_jit")
# Build dummy inputs of the right shape: 1-D after projection
theta_proj_dummy      = jnp.zeros((R, 1))
theta_boot_proj_dummy = jnp.zeros((R, B, 1))
eta_dummy             = jnp.asarray(0.01)
alpha_dummy           = jnp.asarray(0.05)
# warm
ci = multiplier_bootstrap_ci_jit(
    theta_boot=theta_boot_proj_dummy,
    theta_final=theta_proj_dummy,
    eta_final=eta_dummy,
    alpha=alpha_dummy,
)
jax.block_until_ready(ci["ci_normal_lo"])

t0 = time.time()
for _ in range(NREP):
    ci = multiplier_bootstrap_ci_jit(
        theta_boot=theta_boot_proj_dummy,
        theta_final=theta_proj_dummy,
        eta_final=eta_dummy,
        alpha=alpha_dummy,
    )
jax.block_until_ready(ci["ci_normal_lo"])
ci_t = (time.time() - t0) / NREP
print(f"  multiplier_bootstrap_ci_jit (warm):    {1000*ci_t:8.3f} ms/call")


# ----------------------------------------------------
# 5. Summary
# ----------------------------------------------------
banner("5. Summary")
print(f"  whole _run_trajectory_batch (T={T}):  {warm_TBIG:.2f}s "
      f"({1000*warm_TBIG/T:.2f} ms/round)")
print(f"  predicted from per-round breakdown:    {1000*combined:.2f} ms/round * {T} = "
      f"{combined*T:.2f}s")
print(f"  multiplier_bootstrap_ci_jit (1 call):  {1000*ci_t:.2f} ms  (one-time)")
