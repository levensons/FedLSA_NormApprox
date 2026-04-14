"""
FedLSA normal approximation experiment.

Sweeps over (trajectory length, confidence level), parallelizing the
`n_traj` main trajectories across worker processes via
`ProcessPoolExecutor`. Each worker owns its own Garnet instance and its
own JAX state (so JIT compilations live per-worker, not per-call).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

from typing import Dict, Any, List
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ==============================================================
# Worker (top-level so it's picklable for ProcessPoolExecutor)
# ==============================================================

def _run_trajectory_batch(kwargs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Worker: runs one batch of `n_traj_batch` trajectories in a subprocess.

    Rebuilds Garnet from its config dict, runs FedLSA training for the
    batch, computes the per-trajectory multiplier-bootstrap CIs, and
    returns numpy arrays that can be pickled back to the main process.
    """
    # Imports are kept local so workers don't pay for them until they are used
    import numpy as np
    import jax
    import jax.numpy as jnp

    from garnet import Garnet
    from fedlsa import FedLSAConfig, fedlsa_train
    from inference import multiplier_bootstrap_ci_jit

    g = Garnet(**kwargs["garnet_cfg"])
    g.set_sample_rng(np.random.default_rng(kwargs["sample_seed"]))

    u = jnp.asarray(kwargs["u"])

    cfg = FedLSAConfig(
        num_rounds=kwargs["num_rounds"],
        local_steps=kwargs["local_steps"],
        alpha=kwargs["alpha"],
        n_traj=kwargs["n_traj_batch"],
        t0=kwargs["t0"],
        gamma_eta=kwargs["gamma_eta"],
        gamma_H=kwargs["gamma_H"],
    )
    out = fedlsa_train(
        g, cfg,
        num_bootstrap=kwargs["num_bootstrap"],
        boot_seed=kwargs["key_seed"],
        progress=False,
    )
    theta_hat = out["theta_final"]              # [R_batch, D]
    theta_boot = out["theta_boot_final"]         # [R_batch, B, D]
    eta_final = out["step_sizes"][-1]            # scalar

    # Projection onto u
    theta_proj = (theta_hat @ u)[:, None]                     # [R_batch, 1]
    theta_boot_proj = (theta_boot @ u)[..., None]             # [R_batch, B, 1]

    ci = multiplier_bootstrap_ci_jit(
        theta_boot=theta_boot_proj,
        theta_final=theta_proj,
        eta_final=eta_final,
        alpha=jnp.asarray(kwargs["ci_alpha"]),
    )

    return {
        "theta_proj":     np.asarray(theta_proj),                   # [R_batch, 1]
        "ci_quantile_lo": np.asarray(ci["ci_quantile_lo"]),         # [R_batch, 1]
        "ci_quantile_hi": np.asarray(ci["ci_quantile_hi"]),
        "ci_normal_lo":   np.asarray(ci["ci_normal_lo"]),
        "ci_normal_hi":   np.asarray(ci["ci_normal_hi"]),
    }


# ==============================================================
# Orchestrator
# ==============================================================

def run_one_trajectory_length(
    executor: ProcessPoolExecutor,
    garnet_cfg: dict,
    sample_seed_base: int,
    theta_star: np.ndarray,
    u: np.ndarray,
    num_rounds: int,
    local_steps: int = 20,
    alpha: float = 1.0,
    n_traj: int = 50,
    num_workers: int = 5,
    t0: float = 1.0,
    gamma_eta: float = 0.6,
    gamma_H: float = 0.0,
    num_bootstrap: int = 100,
    ci_alpha: float = 0.05,
    seed: int = 7,
) -> Dict[str, float]:
    """Fan out `n_traj` trajectories across `num_workers` worker processes."""
    # Split n_traj across workers (possibly uneven remainder)
    base, rem = divmod(n_traj, num_workers)
    batch_sizes = [base + (1 if i < rem else 0) for i in range(num_workers)]
    batch_sizes = [b for b in batch_sizes if b > 0]

    u_np = np.asarray(u)

    job_args = []
    for w, nb in enumerate(batch_sizes):
        job_args.append({
            "garnet_cfg": garnet_cfg,
            "sample_seed": sample_seed_base + w,      # independent streams
            "u": u_np,
            "num_rounds": num_rounds,
            "local_steps": local_steps,
            "alpha": alpha,
            "n_traj_batch": nb,
            "t0": t0,
            "gamma_eta": gamma_eta,
            "gamma_H": gamma_H,
            "num_bootstrap": num_bootstrap,
            "ci_alpha": ci_alpha,
            "key_seed": seed + 1000 + w,              # independent bootstrap rng
        })

    results: List[Dict[str, np.ndarray]] = list(
        executor.map(_run_trajectory_batch, job_args)
    )

    # Concatenate per-trajectory outputs across workers
    def cat(field: str) -> np.ndarray:
        return np.concatenate([r[field] for r in results], axis=0)

    theta_proj_all = cat("theta_proj")       # [n_traj, 1]
    q_lo = cat("ci_quantile_lo")
    q_hi = cat("ci_quantile_hi")
    n_lo = cat("ci_normal_lo")
    n_hi = cat("ci_normal_hi")

    theta_star_proj = float(np.asarray(theta_star) @ u_np)
    bias = float(abs(theta_proj_all.mean() - theta_star_proj))

    covered_q = (q_lo[:, 0] <= theta_star_proj) & (theta_star_proj <= q_hi[:, 0])
    covered_n = (n_lo[:, 0] <= theta_star_proj) & (theta_star_proj <= n_hi[:, 0])

    return {
        "num_rounds": num_rounds,
        "bias": bias,
        "cov_q": float(covered_q.mean()),
        "cov_n": float(covered_n.mean()),
        "ci_q_width": float((q_hi - q_lo).mean()),
        "ci_n_width": float((n_hi - n_lo).mean()),
    }


def main():
    # ==========================================================
    # Experiment configuration
    # ==========================================================
    garnet_cfg = dict(
        ns=30, na=2, b=2, p=5, gamma=0.95,
        nenvs=5,
        heteregoneity_kern=0.02,
        heteregoneity_reward=0.001,
        gen_seed=10,
    )
    sample_seed_base = 33

    fedlsa_cfg = dict(
        local_steps=2,
        alpha=1.0,
        n_traj=512,
        t0=1.0,
        gamma_eta=0.6,
        gamma_H=0.0,
    )

    num_workers = 32
    num_bootstrap = 256

    trajectory_lengths = [1000, 2000, 5000]
    confidence_levels = [0.05, 0.10, 0.20]

    seed = 7
    results_csv = "results_sweep.csv"
    # ==========================================================

    # Build one Garnet on the main process just to get θ* and features for u
    import jax
    import jax.numpy as jnp
    from garnet import Garnet
    from inference import sample_unit_vector

    garnet = Garnet(**garnet_cfg)
    theta_star = np.asarray(garnet.thetalim)       # [D]
    print("θ* :", theta_star, "  norm:", float(np.linalg.norm(theta_star)))

    u = np.asarray(sample_unit_vector(jax.random.key(99), garnet.p))
    print("u  :", u, "  u^T θ* =", float(theta_star @ u))

    total_configs = len(confidence_levels) * len(trajectory_lengths)
    sweep_bar = tqdm(total=total_configs, desc="sweep", leave=True)

    records = []
    # Use 'spawn' — forking a JAX-initialized main process deadlocks children
    # (JAX runs its own threads, and fork() is unsafe in multithreaded procs).
    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=num_workers, mp_context=spawn_ctx
    ) as executor:
        for ci_alpha in confidence_levels:
            print(f"\n=== confidence level 1 − α = {1 - ci_alpha:.2f} "
                  f"(α = {ci_alpha}) ===", flush=True)
            print(f"{'T':>6}  {'bias':>8}  {'cov_q':>6}  {'cov_n':>6}  "
                  f"{'w_q':>8}  {'w_n':>8}", flush=True)
            print("-" * 56, flush=True)

            for T in trajectory_lengths:
                res = run_one_trajectory_length(
                    executor,
                    garnet_cfg=garnet_cfg,
                    sample_seed_base=sample_seed_base,
                    theta_star=theta_star,
                    u=u,
                    num_rounds=T,
                    **fedlsa_cfg,
                    num_workers=num_workers,
                    num_bootstrap=num_bootstrap,
                    ci_alpha=ci_alpha,
                    seed=seed,
                )
                print(f"{T:>6}  {res['bias']:>8.4f}  {res['cov_q']:>6.3f}  "
                      f"{res['cov_n']:>6.3f}  {res['ci_q_width']:>8.4f}  "
                      f"{res['ci_n_width']:>8.4f}", flush=True)

                records.append({
                    "T": T,
                    "alpha": ci_alpha,
                    "confidence": 1.0 - ci_alpha,
                    "n_traj": fedlsa_cfg["n_traj"],
                    "num_workers": num_workers,
                    "num_bootstrap": num_bootstrap,
                    "bias": res["bias"],
                    "cov_q": res["cov_q"],
                    "cov_n": res["cov_n"],
                    "ci_q_width": res["ci_q_width"],
                    "ci_n_width": res["ci_n_width"],
                })
                pd.DataFrame(records).to_csv(results_csv, index=False)
                sweep_bar.update(1)

    sweep_bar.close()
    print(f"\nSaved {len(records)} rows to {results_csv}")


if __name__ == "__main__":
    main()
