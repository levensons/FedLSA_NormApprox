"""
FedLSA normal approximation experiment.

One-pass sweep: each worker runs a single FedLSA trajectory to T_max
(with bootstrap), snapshotting (θ, θ_boot, η) at each requested T_i,
and runs one Lyapunov baseline per T_i. Confidence-level α only enters
in post-processing, so we never rerun trajectories across α.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

from typing import Dict, Any, List, Sequence
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
    One-pass worker: runs a single FedLSA trajectory to T_max with the
    paired multiplier bootstrap, snapshotting at each round in
    `snapshot_rounds`. For each T_i, also runs the Lyapunov baseline.
    Returns per-T arrays; α is applied later in the orchestrator.
    """
    import numpy as np
    import jax.numpy as jnp

    from garnet import Garnet
    from fedlsa import FedLSAConfig, fedlsa_train
    from baseline_lyapunov import run_lyapunov_baseline

    snapshot_rounds = list(kwargs["snapshot_rounds"])
    T_max = max(snapshot_rounds)

    g = Garnet(**kwargs["garnet_cfg"])
    g.set_sample_rng(np.random.default_rng(kwargs["sample_seed"]))
    u = jnp.asarray(kwargs["u"])

    theta_star = jnp.asarray(kwargs["theta_star"]) if "theta_star" in kwargs else None

    cfg = FedLSAConfig(
        num_rounds=T_max,
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
        snapshot_rounds=snapshot_rounds,
        theta_star=theta_star,
    )
    theta_hist      = out["theta_hist"]        # [S, R, D]
    theta_boot_hist = out["theta_boot_hist"]   # [S, R, B, D]
    eta_hist        = out["eta_hist"]          # [S]
    rounds_hist     = out["rounds_hist"]       # [S]

    theta_proj_hist      = (theta_hist @ u)[..., None]        # [S, R, 1]
    theta_boot_proj_hist = (theta_boot_hist @ u)[..., None]   # [S, R, B, 1]

    # One-pass Lyapunov baseline across all T_i
    lyap_all = run_lyapunov_baseline(
        garnet_cfg=kwargs["garnet_cfg"],
        trajectory_lengths=snapshot_rounds,
        local_steps=kwargs["local_steps"],
        alpha_lr=kwargs["alpha"],
        t0=kwargs["t0"],
        gamma_eta=kwargs["gamma_eta"],
        gamma_H=kwargs["gamma_H"],
        n_traj=kwargs["n_traj_batch"],
        sample_seed=kwargs["sample_seed"],
        sigma_burn_in=kwargs.get("sigma_burn_in", 0),
    )
    theta_T_lyap_list = [lyap_all[T]["theta_T_lyap"] for T in snapshot_rounds]
    Sigma_list        = [lyap_all[T]["Sigma"]         for T in snapshot_rounds]
    eta_T_lyap_list   = [lyap_all[T]["eta_T_lyap"]    for T in snapshot_rounds]

    result = {
        "rounds_hist":          np.asarray(rounds_hist),                   # [S]
        "theta_proj_hist":      np.asarray(theta_proj_hist),               # [S, R, 1]
        "theta_boot_proj_hist": np.asarray(theta_boot_proj_hist),          # [S, R, B, 1]
        "eta_hist":             np.asarray(eta_hist),                      # [S]
        "theta_T_lyap_hist":    np.stack(theta_T_lyap_list, axis=0),       # [S, R, D]
        "Sigma_hist":           np.stack(Sigma_list,        axis=0),       # [S, R, D, D]
        "eta_T_lyap_hist":      np.asarray(eta_T_lyap_list, dtype=float),  # [S]
    }
    if "mse_hist" in out:
        result["mse_hist"] = np.asarray(out["mse_hist"])                   # [T_max]
    return result


# ==============================================================
# Orchestrator
# ==============================================================

def run_sweep(
    executor: ProcessPoolExecutor,
    garnet_cfg: dict,
    sample_seed_base: int,
    theta_star: np.ndarray,
    u: np.ndarray,
    trajectory_lengths: Sequence[int],
    confidence_levels: Sequence[float],
    local_steps: int,
    alpha: float,
    n_traj: int,
    num_workers: int,
    t0: float,
    gamma_eta: float,
    gamma_H: float,
    num_bootstrap: int,
    seed: int,
    sigma_burn_in: int = 0,
) -> List[Dict[str, float]]:
    """Fan out `n_traj` trajectories across workers ONCE. All (T, α) pairs
    are computed from the same snapshots in post-processing."""
    import jax.numpy as jnp
    from inference import multiplier_bootstrap_ci_jit
    from baseline_lyapunov import lyapunov_ci_batch

    Ts = sorted(int(T) for T in trajectory_lengths)
    u_np = np.asarray(u)

    # Split n_traj across workers (possibly uneven remainder)
    base, rem = divmod(n_traj, num_workers)
    batch_sizes = [base + (1 if i < rem else 0) for i in range(num_workers)]
    batch_sizes = [b for b in batch_sizes if b > 0]

    theta_star_np = np.asarray(theta_star)
    job_args = []
    for w, nb in enumerate(batch_sizes):
        job_args.append({
            "garnet_cfg": garnet_cfg,
            "sample_seed": sample_seed_base + w,      # independent streams
            "u": u_np,
            "snapshot_rounds": Ts,
            "local_steps": local_steps,
            "alpha": alpha,
            "n_traj_batch": nb,
            "t0": t0,
            "gamma_eta": gamma_eta,
            "gamma_H": gamma_H,
            "num_bootstrap": num_bootstrap,
            "key_seed": seed + 1000 + w,              # independent bootstrap rng
            "theta_star": theta_star_np,
            "sigma_burn_in": sigma_burn_in,
        })

    results: List[Dict[str, np.ndarray]] = list(
        executor.map(_run_trajectory_batch, job_args)
    )

    # Stack per-trajectory outputs across workers (axis=1 of S-leading arrays)
    rounds_hist = results[0]["rounds_hist"]                          # [S]
    def cat_axis1(field: str) -> np.ndarray:
        return np.concatenate([r[field] for r in results], axis=1)
    theta_proj_all      = cat_axis1("theta_proj_hist")               # [S, n_traj, 1]
    theta_boot_proj_all = cat_axis1("theta_boot_proj_hist")          # [S, n_traj, B, 1]
    theta_T_lyap_all    = cat_axis1("theta_T_lyap_hist")             # [S, n_traj, D]
    Sigma_all           = cat_axis1("Sigma_hist")                    # [S, n_traj, D, D]
    eta_hist            = results[0]["eta_hist"]                     # [S] (same across workers)
    eta_T_lyap_hist     = results[0]["eta_T_lyap_hist"]              # [S]

    # Aggregate MSE across workers (weighted mean by batch size)
    mse_hist = None
    if "mse_hist" in results[0]:
        mse_weighted = sum(
            nb * r["mse_hist"] for nb, r in zip(batch_sizes, results)
        )
        mse_hist = mse_weighted / n_traj                              # [T_max]

    theta_star_proj = float(np.asarray(theta_star) @ u_np)

    records: List[Dict[str, float]] = []
    for s, T in enumerate(rounds_hist):
        theta_proj      = jnp.asarray(theta_proj_all[s])             # [n_traj, 1]
        theta_boot_proj = jnp.asarray(theta_boot_proj_all[s])        # [n_traj, B, 1]
        eta_final       = float(eta_hist[s])
        theta_T_lyap    = theta_T_lyap_all[s]                        # [n_traj, D]
        Sigma           = Sigma_all[s]                               # [n_traj, D, D]
        eta_T_lyap      = float(eta_T_lyap_hist[s])

        bias = float(abs(np.asarray(theta_proj).mean() - theta_star_proj))

        for ci_alpha in confidence_levels:
            ci = multiplier_bootstrap_ci_jit(
                theta_boot=theta_boot_proj,
                theta_final=theta_proj,
                eta_final=jnp.asarray(eta_final),
                alpha=jnp.asarray(ci_alpha),
            )
            q_lo = np.asarray(ci["ci_quantile_lo"])[:, 0]
            q_hi = np.asarray(ci["ci_quantile_hi"])[:, 0]
            n_lo = np.asarray(ci["ci_normal_lo"])[:, 0]
            n_hi = np.asarray(ci["ci_normal_hi"])[:, 0]

            l_lo, l_hi = lyapunov_ci_batch(
                theta_T_lyap, Sigma, u_np, eta_T_lyap, float(ci_alpha)
            )

            ind_q = ((q_lo <= theta_star_proj) & (theta_star_proj <= q_hi)).astype(float)
            ind_n = ((n_lo <= theta_star_proj) & (theta_star_proj <= n_hi)).astype(float)
            ind_l = ((l_lo <= theta_star_proj) & (theta_star_proj <= l_hi)).astype(float)
            cov_q = float(ind_q.mean())
            cov_n = float(ind_n.mean())
            cov_l = float(ind_l.mean())
            n_r = ind_q.size
            cov_q_se = float(ind_q.std(ddof=1) / np.sqrt(n_r))
            cov_n_se = float(ind_n.std(ddof=1) / np.sqrt(n_r))
            cov_l_se = float(ind_l.std(ddof=1) / np.sqrt(n_r))

            records.append({
                "T": int(T),
                "alpha": float(ci_alpha),
                "confidence": 1.0 - float(ci_alpha),
                "bias": bias,
                "cov_q": cov_q,
                "cov_n": cov_n,
                "cov_l": cov_l,
                "cov_q_se": cov_q_se,
                "cov_n_se": cov_n_se,
                "cov_l_se": cov_l_se,
                "ci_q_width": float((q_hi - q_lo).mean()),
                "ci_n_width": float((n_hi - n_lo).mean()),
                "ci_l_width": float((l_hi - l_lo).mean()),
            })
    return records, mse_hist


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
        local_steps=20,
        alpha=1.0,
        n_traj=256,
        t0=1000.0,
        gamma_eta=0.6,
        gamma_H=0.0,
    )

    num_workers = 8
    num_bootstrap = 128
    sigma_burn_in = 500    # discard early θ_t from Σ̂_ε accumulator

    trajectory_lengths = [1000, 2000, 5000]
    confidence_levels = [0.05, 0.10, 0.20]

    seed = 7
    results_csv = "results_sweep.csv"
    # ==========================================================

    # Build one Garnet on the main process just to get θ* and features for u
    import jax
    from garnet import Garnet
    from inference import sample_unit_vector

    garnet = Garnet(**garnet_cfg)
    theta_star = np.asarray(garnet.thetalim)       # [D]
    print("θ* :", theta_star, "  norm:", float(np.linalg.norm(theta_star)))

    u = np.asarray(sample_unit_vector(jax.random.key(99), garnet.p))
    print("u  :", u, "  u^T θ* =", float(theta_star @ u))

    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=num_workers, mp_context=spawn_ctx
    ) as executor:
        records, mse_hist = run_sweep(
            executor,
            garnet_cfg=garnet_cfg,
            sample_seed_base=sample_seed_base,
            theta_star=theta_star,
            u=u,
            trajectory_lengths=trajectory_lengths,
            confidence_levels=confidence_levels,
            **fedlsa_cfg,
            num_workers=num_workers,
            num_bootstrap=num_bootstrap,
            seed=seed,
            sigma_burn_in=sigma_burn_in,
        )

    # Save and plot MSE trajectory
    if mse_hist is not None:
        mse_csv = "mse_trajectory.csv"
        mse_df = pd.DataFrame({
            "round": np.arange(1, len(mse_hist) + 1),
            "mse": mse_hist,
        })
        mse_df.to_csv(mse_csv, index=False)
        print(f"Saved MSE trajectory ({len(mse_hist)} rounds) to {mse_csv}")

        from plot_mse import plot_mse
        plot_mse(mse_csv, out_png="mse_trajectory.png")

    # Print and save
    by_T: Dict[int, List[Dict[str, float]]] = {}
    for rec in records:
        rec.update({
            "n_traj": fedlsa_cfg["n_traj"],
            "num_workers": num_workers,
            "num_bootstrap": num_bootstrap,
        })
        by_T.setdefault(rec["T"], []).append(rec)

    for ci_alpha in confidence_levels:
        print(f"\n=== confidence level 1 − α = {1 - ci_alpha:.2f} "
              f"(α = {ci_alpha}) ===", flush=True)
        print(f"{'T':>6}  {'bias':>8}  {'cov_q':>15}  {'cov_n':>15}  {'cov_l':>15}  "
              f"{'w_q':>8}  {'w_n':>8}  {'w_l':>8}", flush=True)
        print("-" * 99, flush=True)
        for T in sorted(by_T):
            r = next(r for r in by_T[T] if r["alpha"] == ci_alpha)
            cq = f"{r['cov_q']:.3f}±{r['cov_q_se']:.3f}"
            cn = f"{r['cov_n']:.3f}±{r['cov_n_se']:.3f}"
            cl = f"{r['cov_l']:.3f}±{r['cov_l_se']:.3f}"
            print(f"{T:>6}  {r['bias']:>8.4f}  {cq:>15}  {cn:>15}  {cl:>15}  "
                  f"{r['ci_q_width']:>8.4f}  {r['ci_n_width']:>8.4f}  "
                  f"{r['ci_l_width']:>8.4f}", flush=True)

    pd.DataFrame(records).to_csv(results_csv, index=False)
    print(f"\nSaved {len(records)} rows to {results_csv}")


if __name__ == "__main__":
    main()
