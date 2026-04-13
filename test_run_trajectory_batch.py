"""
Correctness tests for `run_na_fedlsa._run_trajectory_batch`.

Runs in-process (no ProcessPoolExecutor). Kept tiny so JIT compiles only
once per shape and the whole suite finishes in seconds.
"""
import time
import numpy as np
import jax
from scipy.stats import norm

from garnet import Garnet
from fedlsa import FedLSAConfig, fedlsa_train
from inference import sample_unit_vector
from run_na_fedlsa import _run_trajectory_batch


# ---------------------------------------------------------------
# Common (small) setup
# ---------------------------------------------------------------
GARNET_CFG = dict(
    ns=10, na=2, b=2, p=4, gamma=0.95,
    nenvs=3,
    heteregoneity_kern=0.02,
    heteregoneity_reward=0.001,
    gen_seed=10,
)
T_ROUNDS = 30
N_TRAJ   = 4
B_BOOT   = 8

_g = Garnet(**GARNET_CFG)
THETA_STAR = np.asarray(_g.thetalim)
U = np.asarray(sample_unit_vector(jax.random.key(99), _g.p))
THETA_STAR_PROJ = float(THETA_STAR @ U)


def make_kwargs(sample_seed=33, key_seed=1007, ci_alpha=0.05):
    return dict(
        garnet_cfg=GARNET_CFG,
        sample_seed=sample_seed,
        u=U,
        num_rounds=T_ROUNDS,
        local_steps=5,
        alpha=1.0,
        n_traj_batch=N_TRAJ,
        t0=1.0,
        gamma_eta=0.6,
        gamma_H=0.0,
        num_bootstrap=B_BOOT,
        ci_alpha=ci_alpha,
        key_seed=key_seed,
    )


def section(title):
    print(f"\n=== {title} ===", flush=True)


total_t0 = time.time()


# ---------------------------------------------------------------
# Test 1: shapes
# ---------------------------------------------------------------
section("Test 1: output shapes")
out = _run_trajectory_batch(make_kwargs())
expected = {
    "theta_proj":     (N_TRAJ, 1),
    "ci_quantile_lo": (N_TRAJ, 1),
    "ci_quantile_hi": (N_TRAJ, 1),
    "ci_normal_lo":   (N_TRAJ, 1),
    "ci_normal_hi":   (N_TRAJ, 1),
}
for k, exp in expected.items():
    print(f"  {k:18s} {out[k].shape}  expected {exp}")
    assert out[k].shape == exp
print("  PASS")


# ---------------------------------------------------------------
# Test 2: determinism
# ---------------------------------------------------------------
section("Test 2: determinism")
out_b = _run_trajectory_batch(make_kwargs())
for k in out:
    d = float(np.max(np.abs(out[k] - out_b[k])))
    assert d == 0.0, f"non-deterministic: {k} (max|diff|={d})"
    print(f"  {k:18s} max|diff| = {d:.2e}")
print("  PASS")


# ---------------------------------------------------------------
# Test 3: seed effects
# ---------------------------------------------------------------
section("Test 3: sample_seed and key_seed have correct effect")
out_alt_sample = _run_trajectory_batch(make_kwargs(sample_seed=12345))
out_alt_boot   = _run_trajectory_batch(make_kwargs(key_seed=99999))

d_sample_theta = float(np.max(np.abs(out["theta_proj"] - out_alt_sample["theta_proj"])))
d_boot_theta   = float(np.max(np.abs(out["theta_proj"] - out_alt_boot["theta_proj"])))
d_boot_ci      = float(np.max(np.abs(out["ci_quantile_hi"] - out_alt_boot["ci_quantile_hi"])))

print(f"  sample_seed -> max|theta_proj diff|     = {d_sample_theta:.4f}  (>0)")
print(f"  key_seed    -> max|theta_proj diff|     = {d_boot_theta:.2e} (==0: main is w≡1)")
print(f"  key_seed    -> max|ci_quantile_hi diff| = {d_boot_ci:.4f}  (>0)")
assert d_sample_theta > 0
assert d_boot_theta == 0.0, "key_seed leaked into the main trajectory!"
assert d_boot_ci > 0
print("  PASS")


# ---------------------------------------------------------------
# Test 4: hand-rolled reference cross-check
# ---------------------------------------------------------------
section("Test 4: matches hand-rolled fedlsa_train + numpy reference")
g_ref = Garnet(**GARNET_CFG)
g_ref.set_sample_rng(np.random.default_rng(33))
cfg = FedLSAConfig(
    num_rounds=T_ROUNDS, local_steps=5, alpha=1.0,
    n_traj=N_TRAJ, t0=1.0, gamma_eta=0.6, gamma_H=0.0,
)
res = fedlsa_train(g_ref, cfg, num_bootstrap=B_BOOT, boot_seed=1007, progress=False)
theta_final     = np.asarray(res["theta_final"])           # [R, D]
theta_boot_full = np.asarray(res["theta_boot_final"])      # [R, B, D]
eta_final       = float(res["step_sizes"][-1])

theta_proj_ref      = theta_final @ U                      # [R]
theta_boot_proj_ref = theta_boot_full @ U                  # [R, B]

z = norm.ppf(1.0 - 0.05 / 2.0)
sigma_ref = (theta_boot_proj_ref - theta_proj_ref[:, None]).std(axis=1) / np.sqrt(eta_final)
half_ref = z * np.sqrt(eta_final) * sigma_ref
n_lo_ref = theta_proj_ref - half_ref
n_hi_ref = theta_proj_ref + half_ref
q_lo_ref = np.quantile(theta_boot_proj_ref, 0.025, axis=1)
q_hi_ref = np.quantile(theta_boot_proj_ref, 0.975, axis=1)

diffs = {
    "theta_proj":     np.max(np.abs(out["theta_proj"][:, 0]     - theta_proj_ref)),
    "ci_normal_lo":   np.max(np.abs(out["ci_normal_lo"][:, 0]   - n_lo_ref)),
    "ci_normal_hi":   np.max(np.abs(out["ci_normal_hi"][:, 0]   - n_hi_ref)),
    "ci_quantile_lo": np.max(np.abs(out["ci_quantile_lo"][:, 0] - q_lo_ref)),
    "ci_quantile_hi": np.max(np.abs(out["ci_quantile_hi"][:, 0] - q_hi_ref)),
}
for k, d in diffs.items():
    print(f"  {k:18s} max|worker - ref| = {d:.2e}")
assert max(diffs.values()) < 1e-4
print("  PASS")


# ---------------------------------------------------------------
# Test 5: normal CI symmetric around theta_proj
# ---------------------------------------------------------------
section("Test 5: normal CI symmetric around theta_proj")
mid = 0.5 * (out["ci_normal_lo"] + out["ci_normal_hi"])
d = float(np.max(np.abs(mid - out["theta_proj"])))
print(f"  max|midpoint - theta_proj| = {d:.2e}")
assert d < 1e-5
print("  PASS")


# ---------------------------------------------------------------
# Test 6: alpha controls width
# ---------------------------------------------------------------
section("Test 6: ci_alpha controls width")
out_05 = _run_trajectory_batch(make_kwargs(ci_alpha=0.05))
out_20 = _run_trajectory_batch(make_kwargs(ci_alpha=0.20))
w05 = float((out_05["ci_normal_hi"]   - out_05["ci_normal_lo"]).mean())
w20 = float((out_20["ci_normal_hi"]   - out_20["ci_normal_lo"]).mean())
q05 = float((out_05["ci_quantile_hi"] - out_05["ci_quantile_lo"]).mean())
q20 = float((out_20["ci_quantile_hi"] - out_20["ci_quantile_lo"]).mean())
print(f"  normal   w(0.05)={w05:.4f}  w(0.20)={w20:.4f}")
print(f"  quantile w(0.05)={q05:.4f}  w(0.20)={q20:.4f}")
assert w05 > w20
assert q05 > q20
print("  PASS")


# ---------------------------------------------------------------
# Test 7: convergence sanity at sufficiently large T
# ---------------------------------------------------------------
section("Test 7: convergence — bias shrinks as T grows (realistic config)")
# Use a realistic (production-like) config: bigger problem, more local steps
# per round, more trajectories. JIT recompiles once for this shape.
GARNET_CFG_CONV = dict(
    ns=30, na=2, b=2, p=5, gamma=0.95,
    nenvs=5,
    heteregoneity_kern=0.02,
    heteregoneity_reward=0.001,
    gen_seed=10,
)
g_conv = Garnet(**GARNET_CFG_CONV)
THETA_STAR_CONV      = np.asarray(g_conv.thetalim)
U_CONV               = np.asarray(sample_unit_vector(jax.random.key(99), g_conv.p))
THETA_STAR_PROJ_CONV = float(THETA_STAR_CONV @ U_CONV)
print(f"  theta*_proj = {THETA_STAR_PROJ_CONV:.4f}  |theta*| = {np.linalg.norm(THETA_STAR_CONV):.4f}")

def conv_kwargs(num_rounds):
    return dict(
        garnet_cfg=GARNET_CFG_CONV,
        sample_seed=33,
        u=U_CONV,
        num_rounds=num_rounds,
        local_steps=20,
        alpha=1.0,
        n_traj_batch=20,
        t0=1.0,
        gamma_eta=0.6,
        gamma_H=0.0,
        num_bootstrap=8,
        ci_alpha=0.05,
        key_seed=1007,
    )

stats = {}
for T in (50, 1000):
    t0 = time.time()
    o = _run_trajectory_batch(conv_kwargs(num_rounds=T))
    proj = o["theta_proj"][:, 0]
    bias = float(abs(proj.mean() - THETA_STAR_PROJ_CONV))
    rmse = float(np.sqrt(((proj - THETA_STAR_PROJ_CONV) ** 2).mean()))
    stats[T] = (bias, rmse)
    print(f"  T={T:5d}  bias={bias:.4f}  rmse={rmse:.4f}  "
          f"rel_bias={bias/abs(THETA_STAR_PROJ_CONV):.1%}  ({time.time()-t0:.1f}s)")

# Convergence checks:
# 1. RMSE shrinks substantially (at least 3x) between T=50 and T=1000
assert stats[1000][1] < stats[50][1] / 3.0, (
    f"RMSE did not shrink enough: {stats[50][1]:.4f} -> {stats[1000][1]:.4f}"
)
# 2. Bias at T=1000 is small in absolute terms — at most a few percent of |theta*|
rel_bias = stats[1000][0] / abs(THETA_STAR_PROJ_CONV)
assert rel_bias < 0.05, f"relative bias at T=1000 too large: {rel_bias:.1%}"
print("  PASS")


print(f"\nAll tests PASSED in {time.time() - total_t0:.1f}s.")
