from typing import Dict

import jax
import jax.numpy as jnp

Array = jax.Array


# ----------------------------
# Multiplier bootstrap
# ----------------------------

def multiplier_bootstrap_ci(
    theta_boot: Array,    # [R, B, D]   paired bootstrap endpoints
    theta_final: Array,   # [R, D]      main trajectory endpoints
    eta_final: Array,     # scalar:     step size η_t at the final round
    alpha: Array,         # traced scalar
) -> Dict[str, Array]:
    """
    Build confidence intervals from pre-computed multiplier-bootstrap samples.

    The bootstrap replicates θ_r^{(b)} are produced inside `fedlsa_train`:
    every replicate follows its own FedLSA trajectory on the same (A, b)
    noise as main trajectory r, with a per-step multiplier w ~ Beta-standardized
    (mean 1, variance 1). This function is purely the post-processing step.

    Two per-trajectory CIs are returned:
      - Quantile:              [q_{α/2}(θ^b_r), q_{1−α/2}(θ^b_r)]
      - Normal (η-normalized): θ_r ± z_{1−α/2} · √η_t · σ̂_r,  where

            σ̂_r = std_b( η_t^{−1/2} · (θ_r^b − θ_r) )
                = std_b(θ_r^b) / √η_t

      The √η_t inside σ̂_r makes σ̂_r a stable estimator of √Σ_∞ in the
      FedLSA CLT  (θ_T − θ*) / √η_T → N(0, Σ_∞).  The √η_t outside the
      half-width converts that limit-scale back into the actual fluctuation
      scale of θ_T, giving the asymptotically valid CI.
    """
    z = jax.scipy.special.ndtri(1.0 - alpha / 2.0)
    sqrt_eta = jnp.sqrt(eta_final)

    # Empirical-quantile CI (per trajectory)
    q_lo = jnp.quantile(theta_boot, alpha / 2.0, axis=1)          # [R, D]
    q_hi = jnp.quantile(theta_boot, 1.0 - alpha / 2.0, axis=1)    # [R, D]

    # η-normalized, centered std (per trajectory)
    centered = (theta_boot - theta_final[:, None, :]) / sqrt_eta   # [R, B, D]
    sigma = jnp.std(centered, axis=1)                              # [R, D]
    n_lo = theta_final - z * sqrt_eta * sigma
    n_hi = theta_final + z * sqrt_eta * sigma

    return {
        "ci_quantile_lo": q_lo,
        "ci_quantile_hi": q_hi,
        "ci_normal_lo": n_lo,
        "ci_normal_hi": n_hi,
    }


multiplier_bootstrap_ci_jit = jax.jit(multiplier_bootstrap_ci)


# ----------------------------
# Coverage
# ----------------------------

def compute_coverage(
    ci_lo: Array,       # [R, D]
    ci_hi: Array,       # [R, D]
    theta_star: Array,  # [D]
) -> Dict[str, Array]:
    """
    Returns:
      coord_coverage: [D]   – per-coordinate coverage across runs
      joint_coverage: float  – fraction of runs where ALL coordinates covered
    """
    covered = (ci_lo <= theta_star[None, :]) & (theta_star[None, :] <= ci_hi)
    coord_coverage = jnp.mean(covered.astype(jnp.float32), axis=0)
    joint_coverage = jnp.mean(jnp.all(covered, axis=1).astype(jnp.float32))
    return {"coord_coverage": coord_coverage, "joint_coverage": joint_coverage}


# ----------------------------
# Random projection
# ----------------------------

def sample_unit_vector(key: Array, dim: int) -> Array:
    """Sample u uniformly from S^{d-1}."""
    v = jax.random.normal(key, (dim,))
    return v / jnp.linalg.norm(v)
