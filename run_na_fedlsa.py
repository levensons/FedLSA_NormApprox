# fedlsa_linear_td_jax.py
from dataclasses import dataclass
from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp

Array = jax.Array



# ----------------------------
# Problem / environment setup
# ----------------------------

@jax.tree_util.register_dataclass
@dataclass
class HeteroMRP:
    """
    Heterogeneous Markov reward process for multiple agents.

    Shapes:
      transition_logits: [A, S, S]
      reward_mean:       [A, S]
      reward_std:        [A, S]
      start_probs:       [A, S]
      phi:               [S, D]
    """
    transition_logits: Array
    reward_mean: Array
    reward_std: Array
    start_probs: Array
    phi: Array
    gamma: float


@dataclass(frozen=True)
class FedLSAConfig:
    num_rounds: int
    local_steps: int
    alpha: float
    num_runs: int
    seed: int = 0


def normalize_probs(x: Array, axis: int = -1, eps: float = 1e-8) -> Array:
    x = jnp.clip(x, a_min=0.0)
    return x / (jnp.sum(x, axis=axis, keepdims=True) + eps)


def transition_probs(env: HeteroMRP) -> Array:
    return jax.nn.softmax(env.transition_logits, axis=-1)


def sample_categorical_batched(key: Array, probs: Array) -> Array:
    """
    Batched categorical sampling with one key, using Gumbel-max.

    probs shape: [..., K]
    returns:     [...]
    """
    g = jax.random.gumbel(key, probs.shape)
    return jnp.argmax(jnp.log(probs) + g, axis=-1)


def make_random_hetero_mrp(
    key: Array,
    num_agents: int,
    num_states: int,
    feature_dim: int,
    gamma: float = 0.95,
    reward_noise_scale: float = 0.1,
    heterogeneity_scale: float = 0.5,
) -> HeteroMRP:
    """
    Creates a synthetic heterogeneous multi-agent MRP.
    Good for smoke tests and benchmarking.
    """
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # Agent-specific transitions around a shared base.
    base_logits = jax.random.normal(k1, (num_states, num_states))
    agent_shift = heterogeneity_scale * jax.random.normal(
        k2, (num_agents, num_states, num_states)
    )
    transition_logits = base_logits[None, :, :] + agent_shift

    # Agent-specific rewards.
    reward_mean = jax.random.normal(k3, (num_agents, num_states))
    reward_std = reward_noise_scale * jnp.ones((num_agents, num_states))

    # Agent-specific initial-state distributions.
    start_raw = jax.random.uniform(k4, (num_agents, num_states))
    start_probs = normalize_probs(start_raw, axis=-1)

    # Shared linear features.
    phi = jax.random.normal(k5, (num_states, feature_dim)) / jnp.sqrt(feature_dim)

    return HeteroMRP(
        transition_logits=transition_logits,
        reward_mean=reward_mean,
        reward_std=reward_std,
        start_probs=start_probs,
        phi=phi,
        gamma=gamma,
    )


# ----------------------------
# TD / FedLSA core
# ----------------------------

def reset_states(key: Array, env: HeteroMRP, num_runs: int) -> Array:
    """
    Sample initial states for every run and every agent.

    Returns shape: [R, A]
    """
    probs = jnp.broadcast_to(
        env.start_probs[None, :, :],
        (num_runs,) + env.start_probs.shape
    )  # [R, A, S]
    return sample_categorical_batched(key, probs)


def td_local_step(
    theta_ra: Array,   # [R, A, D]
    states_ra: Array,  # [R, A]
    env: HeteroMRP,
    alpha: float,
    step_key: Array,
):
    """
    One synchronous local TD(0) step for all runs and all agents.
    """
    num_runs, num_agents, _ = theta_ra.shape
    P = transition_probs(env)  # [A, S, S]

    next_state_key, reward_key = jax.random.split(step_key)

    # Select P_a[s, :]
    batch_P = jnp.broadcast_to(P[None, :, :, :], (num_runs,) + P.shape)   # [R, A, S, S]
    cur_probs = jnp.take_along_axis(
        batch_P,
        states_ra[..., None, None],
        axis=2
    ).squeeze(2)  # [R, A, S]

    next_states = sample_categorical_batched(next_state_key, cur_probs)  # [R, A]

    phi_s = env.phi[states_ra]       # [R, A, D]
    phi_sp = env.phi[next_states]    # [R, A, D]

    r_mean = env.reward_mean[None, :, :]  # [1, A, S]
    r_std = env.reward_std[None, :, :]    # [1, A, S]
    mean_s = jnp.take_along_axis(r_mean, states_ra[..., None], axis=-1).squeeze(-1)  # [R, A]
    std_s = jnp.take_along_axis(r_std, states_ra[..., None], axis=-1).squeeze(-1)    # [R, A]

    rewards = mean_s + std_s * jax.random.normal(reward_key, mean_s.shape)  # [R, A]

    v_s = jnp.sum(phi_s * theta_ra, axis=-1)     # [R, A]
    v_sp = jnp.sum(phi_sp * theta_ra, axis=-1)   # [R, A]
    td_error = rewards + env.gamma * v_sp - v_s  # [R, A]

    theta_next = theta_ra + alpha * td_error[..., None] * phi_s  # [R, A, D]

    metrics = {
        "mean_td_error_sq": jnp.mean(td_error ** 2),
        "mean_reward": jnp.mean(rewards),
    }
    return theta_next, next_states, metrics


def fedlsa_train(
    env: HeteroMRP,
    config: FedLSAConfig,
    theta0: Optional[Array] = None,   # [R, D] or None
) -> Dict[str, Any]:
    """
    FedLSA-style training:
      - broadcast global theta to all agents
      - K local TD steps per agent
      - server averages local thetas

    Returns:
      theta_final: [R, D]
      final_states: [R, A]
      history:
        theta_mean_norm: [T]
        local_td_error_sq: [T]
        local_reward: [T]
    """
    num_agents = env.reward_mean.shape[0]
    feature_dim = env.phi.shape[1]

    key = jax.random.key(config.seed)

    if theta0 is None:
        theta_global0 = jnp.zeros((config.num_runs, feature_dim))
    else:
        theta_global0 = theta0

    init_states = reset_states(
        key=jax.random.fold_in(key, 0),
        env=env,
        num_runs=config.num_runs,
    )  # [R, A]

    def round_body(carry, round_idx):
        theta_global, states = carry  # [R, D], [R, A]

        # Broadcast current global model to all agents.
        theta_local = jnp.broadcast_to(
            theta_global[:, None, :],
            (config.num_runs, num_agents, feature_dim)
        )  # [R, A, D]

        round_key = jax.random.fold_in(key, round_idx + 1)
        step_keys = jax.random.split(round_key, config.local_steps)

        def local_body(local_carry, step_key):
            theta_ra, s_ra = local_carry
            theta_next, s_next, metrics = td_local_step(
                theta_ra=theta_ra,
                states_ra=s_ra,
                env=env,
                alpha=config.alpha,
                step_key=step_key,
            )
            return (theta_next, s_next), metrics

        (theta_after, states_after), local_metrics = jax.lax.scan(
            local_body,
            (theta_local, states),
            step_keys,
        )

        # theta_after is the post-local-updates parameter for each agent.
        # shape: [R, A, D]
        theta_global_next = jnp.mean(theta_after, axis=1)  # [R, D]

        round_metrics = {
            "theta_mean_norm": jnp.mean(jnp.linalg.norm(theta_global_next, axis=-1)),
            "local_td_error_sq": jnp.mean(local_metrics["mean_td_error_sq"]),
            "local_reward": jnp.mean(local_metrics["mean_reward"]),
        }
        return (theta_global_next, states_after), round_metrics

    (theta_final, final_states), history = jax.lax.scan(
        round_body,
        (theta_global0, init_states),
        jnp.arange(config.num_rounds),
    )

    return {
        "theta_final": theta_final,
        "final_states": final_states,
        "history": history,
    }


# Mark config static so num_rounds/local_steps/num_runs are compile-time constants.
fedlsa_train_jit = jax.jit(fedlsa_train, static_argnames=("config",))


# ----------------------------
# Optional utilities
# ----------------------------

def estimate_value(theta_rd: Array, phi_sd: Array) -> Array:
    """
    theta_rd: [R, D]
    phi_sd:   [S, D]
    returns:  [R, S]
    """
    return theta_rd @ phi_sd.T


def per_run_l2_to_reference(theta_rd: Array, theta_ref_d: Array) -> Array:
    return jnp.linalg.norm(theta_rd - theta_ref_d[None, :], axis=-1)


# ----------------------------
# Example usage
# ----------------------------

def main():
    env = make_random_hetero_mrp(
        key=jax.random.key(42),
        num_agents=8,
        num_states=32,
        feature_dim=16,
        gamma=0.95,
        reward_noise_scale=0.1,
        heterogeneity_scale=0.8,
    )

    cfg = FedLSAConfig(
        num_rounds=10,
        local_steps=20,
        alpha=0.05,
        num_runs=10,   # many independent runs in parallel
        seed=7,
    )

    out = fedlsa_train_jit(env, cfg)

    print("theta_final shape:", out["theta_final"].shape)          # [256, 16]
    print("final_states shape:", out["final_states"].shape)        # [256, 8]
    print("history keys:", out["history"].keys())
    print("last 5 mean norms:", out["history"]["theta_mean_norm"][-5:])
    print("last 5 td-error^2:", out["history"]["local_td_error_sq"][-5:])

    # Example: estimated value function for each run
    vhat_rs = estimate_value(out["theta_final"], env.phi)
    print("value estimates shape:", vhat_rs.shape)                 # [256, 32]


if __name__ == "__main__":
    main()