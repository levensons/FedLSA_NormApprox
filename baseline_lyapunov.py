"""
Plug-in Lyapunov baseline for last-iterate inference in FedLSA.

Sample-splitting (total T Garnet-sampling rounds per trajectory):

  Phase 1 (T/2 rounds, no θ update) — estimate, per trajectory:
      Â_n, b̂_n        = sample means of per-agent (A_c, b_c) over phase 1
      Â = (1/N) Σ_n Â_n,   b̂ = (1/N) Σ_n b̂_n
      θ̂* = Â⁻¹ b̂                           (plug-in fixed point)
      ε_{t,h,n} = A_{c,n,t,h} θ̂* − b_{c,n,t,h}
      Σ̂_ε = (1/N²) Σ_n Cov(ε_n)            (per-agent residual covariance)
      Σ̂  solves the continuous Lyapunov equation  Â Σ̂ + Σ̂ Âᵀ = Σ̂_ε

  Phase 2 (T/2 rounds, run FedLSA from θ=0 with step sizes
    η_t = α (t + 1 + t₀)^{−γ_η} for t = 0..T/2−1, so effectively
    t' = 1..T/2 with the usual α (t' + t₀)^{−γ_η} schedule) — gets θ_T.

  CI:  u^T θ_T ± z_{α/2} √η_T √(u^T Σ̂ u),  η_T = α (T/2 + t₀)^{−γ_η}.
"""
from typing import Dict, Any

import numpy as np
import scipy.linalg
from scipy.stats import norm


def _phase1_moments(garnet, T_half: int, H: int, R: int, seed: int):
    """
    Two-pass sampling over T_half rounds of H×R (A_c, b_c) draws per agent.
    Pass 1 computes per-agent means; pass 2 (with the same seed) computes
    per-agent residual covariances at the plug-in θ̂*.
    """
    N = garnet.nenvs
    D = garnet.p

    # Pass 1: per-agent means.
    garnet.set_sample_rng(np.random.default_rng(seed))
    sum_A = np.zeros((R, N, D, D))
    sum_b = np.zeros((R, N, D))
    for _ in range(T_half):
        As, bs = garnet.sample_A_and_b(H * R)                 # (N, H*R, D, D), (N, H*R, D)
        As = As.reshape(N, R, H, D, D).sum(axis=2)            # (N, R, D, D)
        bs = bs.reshape(N, R, H, D).sum(axis=2)               # (N, R, D)
        sum_A += np.transpose(As, (1, 0, 2, 3))
        sum_b += np.transpose(bs, (1, 0, 2))
    M = T_half * H
    A_mean = sum_A / M                                        # [R, N, D, D]
    b_mean = sum_b / M                                        # [R, N, D]
    A_bar  = A_mean.mean(axis=1)                              # [R, D, D]
    b_bar  = b_mean.mean(axis=1)                              # [R, D]
    # NumPy 2.0: solve() broadcasts (R,D) as a stack of matrices, not vectors.
    # Force vector semantics with an explicit trailing axis.
    theta_hat_star = np.linalg.solve(A_bar, b_bar[..., None])[..., 0]  # [R, D]

    # Pass 2: per-agent residual second moments at θ̂*.
    garnet.set_sample_rng(np.random.default_rng(seed))
    sum_eps  = np.zeros((R, N, D))
    sum_eps2 = np.zeros((R, N, D, D))
    for _ in range(T_half):
        As, bs = garnet.sample_A_and_b(H * R)
        As = As.reshape(N, R, H, D, D)
        bs = bs.reshape(N, R, H, D)
        A_theta = np.einsum('nrhij,rj->nrhi', As, theta_hat_star)   # (N, R, H, D)
        eps = A_theta - bs                                          # (N, R, H, D)
        sum_eps  += np.transpose(eps.sum(axis=2), (1, 0, 2))        # (R, N, D)
        eps_outer = np.einsum('nrhi,nrhj->nrij', eps, eps)          # (N, R, D, D)
        sum_eps2 += np.transpose(eps_outer, (1, 0, 2, 3))           # (R, N, D, D)
    eps_mean   = sum_eps  / M                                       # [R, N, D]
    eps_second = sum_eps2 / M                                       # [R, N, D, D]
    cov_eps_n  = eps_second - np.einsum('rni,rnj->rnij', eps_mean, eps_mean)
    Sigma_eps  = cov_eps_n.sum(axis=1) / (N * N)                    # [R, D, D]

    return A_bar, theta_hat_star, Sigma_eps


def _solve_lyapunov_batched(A_bar: np.ndarray, Sigma_eps: np.ndarray) -> np.ndarray:
    """For each r, solve  A Σ + Σ Aᵀ = Σ_ε  via scipy's continuous Lyapunov."""
    R = A_bar.shape[0]
    Sigma = np.empty_like(A_bar)
    for r in range(R):
        Sigma[r] = scipy.linalg.solve_continuous_lyapunov(A_bar[r], Sigma_eps[r])
    return Sigma


def lyapunov_ci_batch(
    theta_T: np.ndarray,   # [R, D]
    Sigma:   np.ndarray,   # [R, D, D]
    u:       np.ndarray,   # [D]
    eta_T:   float,
    alpha:   float,
):
    """Gaussian CI  u^T θ_T ± z_{α/2} √η_T √(u^T Σ u),  per trajectory."""
    z = norm.ppf(1.0 - alpha / 2.0)
    theta_proj = theta_T @ u                                            # [R]
    var_u = np.einsum('i,rij,j->r', u, Sigma, u)                        # [R]
    half = z * np.sqrt(eta_T) * np.sqrt(np.clip(var_u, 0.0, None))
    return theta_proj - half, theta_proj + half


def run_lyapunov_baseline(
    *,
    garnet_cfg: Dict[str, Any],
    num_rounds: int,
    local_steps: int,
    alpha_lr: float,
    t0: float,
    gamma_eta: float,
    gamma_H: float,
    n_traj: int,
    sample_seed: int,
    u: np.ndarray,
    ci_alpha: float,
):
    """
    Returns a dict with per-trajectory Lyapunov-CI arrays.
    Uses two independent Garnet instances (different seeds) for
    phases 1 and 2.
    """
    from garnet import Garnet
    from fedlsa import FedLSAConfig, fedlsa_train

    T_half = num_rounds // 2
    # Offsets are arbitrary large primes chosen to make the seeds distinct
    # from the bootstrap path's sample_seed.
    PHASE1_OFFSET = 100003
    PHASE2_OFFSET = 200003

    g1 = Garnet(**garnet_cfg)
    A_bar, theta_hat_star, Sigma_eps = _phase1_moments(
        g1, T_half, local_steps, n_traj, sample_seed + PHASE1_OFFSET
    )
    Sigma = _solve_lyapunov_batched(A_bar, Sigma_eps)

    g2 = Garnet(**garnet_cfg)
    g2.set_sample_rng(np.random.default_rng(sample_seed + PHASE2_OFFSET))
    cfg2 = FedLSAConfig(
        num_rounds=T_half,
        local_steps=local_steps,
        alpha=alpha_lr,
        n_traj=n_traj,
        t0=t0 + 1.0,          # shift schedule so step sizes run 1+t₀ … T/2+t₀
        gamma_eta=gamma_eta,
        gamma_H=gamma_H,
    )
    out = fedlsa_train(g2, cfg2, num_bootstrap=0, progress=False)
    theta_T_lyap = np.asarray(out["theta_final"])                  # [R, D]
    eta_T = float(out["step_sizes"][-1])

    lo, hi = lyapunov_ci_batch(
        theta_T_lyap, Sigma, np.asarray(u), eta_T, ci_alpha
    )
    return {
        "ci_lyap_lo":     lo[:, None],                              # [R, 1]
        "ci_lyap_hi":     hi[:, None],
        "theta_T_lyap":   theta_T_lyap,
        "theta_hat_star": theta_hat_star,
        "eta_T_lyap":     eta_T,
    }
