"""
Plug-in Lyapunov baseline for last-iterate inference in FedLSA.

Single loop 0…T_max−1 over sorted trajectory lengths [T_1, …, T_k]:

  Each round t:
    1. Sample (A, b); update running averages of A, b.
    2. Run global FedLSA trajectory θ_t (used as θ̂*).
    3. Compute centered noise ε = (A − Â)θ_t − (b − b̂);
       update running average of ε⊗ε.
    4. At t+1 = T_i/2: solve Lyapunov equation → Σ̂(T_i).
    5. For T_i/2 ≤ t < T_i: run per-T_i trajectory on the same samples.

CI:  u^T θ_T ± z_{α/2} √η_T √(u^T Σ̂ u).
"""
import math
from typing import Dict, Any, Sequence

import numpy as np
import scipy.linalg
from scipy.stats import norm


def _solve_lyapunov_batched(A_bar, Sigma_eps):
    """Solve  A Σ + Σ Aᵀ = Σ_ε  per trajectory r."""
    R = A_bar.shape[0]
    Sigma = np.empty_like(A_bar)
    for r in range(R):
        Sigma[r] = scipy.linalg.solve_continuous_lyapunov(A_bar[r], Sigma_eps[r])
    return Sigma


def lyapunov_ci_batch(theta_T, Sigma, u, eta_T, alpha):
    """Gaussian CI  u^T θ_T ± z_{α/2} √η_T √(u^T Σ u),  per trajectory."""
    z = norm.ppf(1.0 - alpha / 2.0)
    var_u = np.einsum('i,rij,j->r', u, Sigma, u)
    half = z * np.sqrt(eta_T) * np.sqrt(np.clip(var_u, 0.0, None))
    proj = theta_T @ u
    return proj - half, proj + half


def run_lyapunov_baseline(
    *,
    garnet_cfg: Dict[str, Any],
    trajectory_lengths: Sequence[int],
    local_steps: int,
    alpha_lr: float,
    t0: float,
    gamma_eta: float,
    gamma_H: float,
    n_traj: int,
    sample_seed: int,
    sigma_burn_in: int = 0,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Multi-L single-loop Lyapunov baseline.

    `sigma_burn_in`: skip first `sigma_burn_in` rounds when accumulating
    Σ̂_ε. Early θ_t are far from θ* and inflate ε = (A − Â)θ_t − (b − b̂);
    discarding them removes the transient bias. Â, b̂ and the global
    trajectory θ still update from t=0.

    Returns {T_i: {"theta_T_lyap": [R, D], "Sigma": [R, D, D], "eta_T_lyap": float}}
    """
    import jax.numpy as jnp
    from garnet import Garnet
    from fedlsa import _fedlsa_one_round, _sample_round_Ab

    Ts = sorted(int(T) for T in trajectory_lengths)
    T_max = Ts[-1]

    garnet = Garnet(**garnet_cfg)
    N, D, R = garnet.nenvs, garnet.p, n_traj

    garnet.set_sample_rng(np.random.default_rng(sample_seed))

    def _H_at(t):
        return max(1, int(math.ceil(local_steps * (t + 1) ** gamma_H)))

    # Running averages in [R, N, ...] layout
    avg_A     = np.zeros((R, N, D, D))
    avg_b     = np.zeros((R, N, D))
    avg_sigma = np.zeros((R, N, D, D))
    n_samples     = 0      # for Â, b̂  (counts all rounds)
    n_eps_samples = 0      # for Σ̂_ε  (excludes burn-in)

    theta       = jnp.zeros((R, 1, D))          # global trajectory
    results: Dict[int, Dict[str, Any]] = {}

    for t in range(T_max):
        H_t = _H_at(t)
        A_jax, b_jax = _sample_round_Ab(garnet, H_t, R)  # [H, R, N, D, D], [H, R, N, D]
        As = np.asarray(A_jax)
        bs = np.asarray(b_jax)

        # Running averages of A, b (always accumulate)
        avg_A = (n_samples * avg_A + As.sum(axis=0)) / (n_samples + H_t)
        avg_b = (n_samples * avg_b + bs.sum(axis=0)) / (n_samples + H_t)
        n_samples += H_t

        # Centered noise ε = (A − Â)θ − (b − b̂), running average of ε⊗ε.
        # Skip while θ_t is in transient regime to avoid inflating Σ̂_ε.
        if t >= sigma_burn_in:
            th = np.asarray(theta[:, 0, :])                            # [R, D]
            eps = (np.einsum('hrnij,rj->hrni', As - avg_A[None], th)
                   - (bs - avg_b[None]))                               # [H, R, N, D]
            avg_sigma = (n_eps_samples * avg_sigma
                         + np.einsum('hrni,hrnj->rnij', eps, eps)) / (n_eps_samples + H_t)
            n_eps_samples += H_t

        # Global trajectory update
        weights = jnp.ones((H_t, R, 1, N))
        alpha_t = alpha_lr * (t + t0) ** (-gamma_eta)
        theta = _fedlsa_one_round(theta, A_jax, b_jax, jnp.asarray(alpha_t), weights)

        # Solve Lyapunov at T_i/2
        if (t + 1) in Ts:
            A_bar     = avg_A.mean(axis=1)            # [R, D, D]
            Sigma_eps = avg_sigma.mean(axis=1) / N    # [R, D, D]
            Sigma = _solve_lyapunov_batched(A_bar, Sigma_eps)
            results[t + 1] = {"Sigma": Sigma,
                              "theta_T_lyap": np.asarray(theta[:, 0, :]),
                              "eta_T_lyap": alpha_t}


    return results
