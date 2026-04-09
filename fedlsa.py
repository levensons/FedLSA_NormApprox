import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from garnet import Garnet

Array = jax.Array


@dataclass(frozen=True)
class FedLSAConfig:
    num_rounds: int
    local_steps: int        # base local steps H₀
    alpha: float            # base learning rate η
    n_traj: int
    t0: float = 1.0         # step-size offset
    gamma_eta: float = 0.0  # step-size decay:  η_t = η (t+t₀)^{−γ_η}
    gamma_H: float = 0.0    # local-step growth: H_t = ⌈H₀ (t+1)^{γ_H}⌉


# ------------------------------------------------------------
# Inner loop
# ------------------------------------------------------------
#
# We carry a tensor  θ : [R, K, D]  where
#     K = 1 + num_bootstrap
#     index 0       → the main trajectory   (weight w ≡ 1)
#     indices 1..B  → multiplier-bootstrap replicates (random w, mean 1 var 1)
#
# The main update and all bootstrap updates run in lockstep on the same
# (A, b) noise sampled from Garnet, which is the whole point of the
# multiplier bootstrap: the bootstrap trajectories track the main one
# because they share the noise, and their dispersion around the main
# trajectory approximates the sampling distribution of θ_T.


@jax.jit
def _fedlsa_one_round(
    theta: Array,           # [R, K, D]   main (k=0) + bootstrap replicates
    A_round: Array,         # [H, R, N, D, D]
    b_round: Array,         # [H, R, N, D]
    alpha_t: Array,         # 0-d scalar
    weights_round: Array,   # [H, R, K, N]  with weights[..., 0, :] ≡ 1
):
    """
    One FedLSA round: H local TD(0) steps, run in parallel across
      R  main trajectories,
      K = 1 + B replicates each (main + bootstrap, sharing the same Z_t),
      N agents,
    then averaging across agents.
    """
    R, K, D = theta.shape
    N = A_round.shape[2]

    theta_local = jnp.broadcast_to(theta[:, :, None, :], (R, K, N, D))

    def local_body(theta_rkn, step_data):
        A_h, b_h, w_h = step_data   # [R, N, D, D], [R, N, D], [R, K, N]
        # A_h θ − b_h,  broadcasted across the K replicates
        A_theta = jnp.einsum('rnij,rknj->rkni', A_h, theta_rkn)
        update = A_theta - b_h[:, None, :, :]             # [R, K, N, D]
        theta_next = theta_rkn - alpha_t * w_h[..., None] * update
        return theta_next, None

    theta_after, _ = jax.lax.scan(
        local_body,
        theta_local,
        (A_round, b_round, weights_round),
    )

    # Aggregate agents for both main and every bootstrap replicate
    return jnp.mean(theta_after, axis=2)   # [R, K, D]


def _sample_round_Ab(garnet, H_t: int, R: int):
    """
    Pull a batch of H_t × R i.i.d. (A_c, b_c) samples per agent from Garnet
    and reshape into JAX arrays of shape [H, R, N, D, D] and [H, R, N, D].
    """
    num_agents = garnet.nenvs
    D = garnet.p

    As_np, bs_np = garnet.sample_A_and_b(H_t * R)
    # (N, H_t*R, D, D), (N, H_t*R, D)

    As_np = As_np.reshape(num_agents, R, H_t, D, D)
    As_np = np.transpose(As_np, (2, 1, 0, 3, 4))   # [H, R, N, D, D]
    bs_np = bs_np.reshape(num_agents, R, H_t, D)
    bs_np = np.transpose(bs_np, (2, 1, 0, 3))       # [H, R, N, D]

    return jnp.asarray(As_np), jnp.asarray(bs_np)


def _sample_bootstrap_weights(rng: np.random.Generator, shape) -> Array:
    """
    Standardized Beta(0.5, 2) multipliers
        e = 1 + (w − E[w]) / √Var[w],   w ~ Beta(0.5, 2)
    Mean 1, variance 1.

    Sampled with NumPy's C-implemented Beta sampler (much faster than
    jax.random.beta, which falls back to rejection sampling in a
    lax.while_loop for α<1) and transferred to a JAX array once.
    """
    a_beta, b_beta = 0.5, 2.0
    mu_w = a_beta / (a_beta + b_beta)
    var_w = (a_beta * b_beta) / (
        (a_beta + b_beta) ** 2 * (a_beta + b_beta + 1)
    )
    w = rng.beta(a_beta, b_beta, size=shape)
    e = 1.0 + (w - mu_w) / np.sqrt(var_w)
    return jnp.asarray(e, dtype=jnp.float32)


# ------------------------------------------------------------
# Trainer (main + optional paired multiplier bootstrap)
# ------------------------------------------------------------

def fedlsa_train(
    garnet: "Garnet",
    config: FedLSAConfig,
    num_bootstrap: int = 0,
    boot_seed: int = 0,
    theta0: Optional[Array] = None,
    progress: bool = True,
    progress_desc: Optional[str] = None,
) -> Dict[str, Any]:
    """
    FedLSA with i.i.d. sampling from Garnet and, when `num_bootstrap > 0`,
    a paired multiplier bootstrap that shares the same (A, b) noise.

    At every round:
      1. Compute H_t and η_t.
      2. Sample H_t × R observations per agent from Garnet.
      3. Draw per-local-step weights  w_{t,n}^b  (mean 1, var 1) for every
         bootstrap replicate; the main trajectory uses w ≡ 1.
      4. Run H_t local TD(0) steps for main + all B bootstrap replicates
         together, on the same (A, b).
      5. Average across agents.

    Returns:
      theta_final:      [R, D]              main trajectory endpoints
      theta_boot_final: [R, B, D]            bootstrap endpoints (B = num_bootstrap)
      step_sizes:       [T]                  η_t per round
      local_steps:      [T]                  H_t per round (numpy int array)
    """
    R = config.n_traj
    D = garnet.p
    B = num_bootstrap
    K = 1 + B

    if theta0 is None:
        theta_main0 = jnp.zeros((R, D))
    else:
        theta_main0 = theta0

    # theta [R, K, D]:  k=0 main, k=1..B bootstrap replicates (all start at θ₀)
    theta = jnp.broadcast_to(theta_main0[:, None, :], (R, K, D))

    step_sizes_hist: List[float] = []
    H_hist: List[int] = []

    boot_rng = np.random.default_rng(boot_seed)

    round_iter = range(config.num_rounds)
    if progress:
        round_iter = tqdm(
            round_iter,
            total=config.num_rounds,
            desc=progress_desc or "FedLSA",
            leave=False,
        )

    for t in round_iter:
        H_t = max(
            1,
            int(math.ceil(config.local_steps * (t + 1) ** config.gamma_H)),
        )
        alpha_t = float(
            config.alpha * (t + config.t0) ** (-config.gamma_eta)
        )

        A_round, b_round = _sample_round_Ab(garnet, H_t=H_t, R=R)
        N = A_round.shape[2]

        # Per-local-step weights: [H_t, R, K, N]
        #   k=0           → main, w ≡ 1
        #   k=1..B        → bootstrap, standardized Beta multipliers
        main_w = jnp.ones((H_t, R, 1, N))
        if B > 0:
            boot_w = _sample_bootstrap_weights(boot_rng, (H_t, R, B, N))
            weights_round = jnp.concatenate([main_w, boot_w], axis=2)
        else:
            weights_round = main_w

        theta = _fedlsa_one_round(
            theta, A_round, b_round, jnp.asarray(alpha_t), weights_round
        )

        step_sizes_hist.append(alpha_t)
        H_hist.append(H_t)

    return {
        "theta_final": theta[:, 0, :],              # [R, D]
        "theta_boot_final": theta[:, 1:, :],        # [R, B, D]
        "step_sizes": jnp.asarray(step_sizes_hist), # [T]
        "local_steps": np.asarray(H_hist),          # [T]
    }
