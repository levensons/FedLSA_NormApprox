# Plan for Confidence-Set Experiments

## Overview

We fix a projection direction `u` on the unit sphere and project all
confidence sets onto this direction. Coverage is always reported for the
scalar quantity `u^T θ*`.

We sweep two experiment configurations:

- **trajectory length** `T` (number of FedLSA rounds)
- **confidence level** `1 − α`

Current defaults (in `run_na_fedlsa.py::main`):
`T ∈ {50, 100, 200, 500, 1000}`, `α ∈ {0.01, 0.05, 0.10}`.

---

## Environment

We use a **Garnet** heterogeneous MRP (`garnet.py`, numpy), parameterised by
`(n_s, n_a, b, p, γ, n_envs, ε_kern, ε_reward, gen_seed)`. Garnet provides:

- `stat_dist[a]` — stationary distribution of agent `a` under the policy
- `reward_tr_kernels[a]` — the MRP transition matrix under the policy
- `rewards[a, s]` — deterministic reward
- `feat_map[s, :]` — shared feature map
- `thetalim` — the FedLSA fixed point `θ*`, used as ground truth
- `sample_A_and_b(n)` — draws `n` i.i.d. triples `(s, s', r)` from the
  stationary distribution and returns the design matrices

  ```
  A_c = φ(s) (φ(s) − γ φ(s'))ᵀ     b_c = r · φ(s)
  ```

All observations used by training and by the plug-in covariance come
through this sampler — **no Markovian sampling, no feature/gamma logic
duplicated on the JAX side.**

---

## FedLSA Training (`fedlsa.py`)

A Python-level outer loop over rounds `t = 0, …, T−1`, wrapping a jitted
single-round update. At every round:

1. **Step size** `η_t = η · (t + t₀)^{−γ_η}` with `γ_η = 0.6`.
2. **Local steps** `H_t = max(1, ⌈H₀ · (t+1)^{γ_H}⌉)` — supports a *growing*
   local-step schedule (`γ_H ≥ 0`).
3. **Sampling.** Call `garnet.sample_A_and_b(H_t · R)` to pull a fresh batch
   of i.i.d. `(A_c, b_c)` pairs per agent for this round only; reshape to
   `[H_t, R, N, D, D]` and `[H_t, R, N, D]`.
4. **Local TD(0).** For each local step `h`, parallel over runs × agents,

   ```
   θ ← θ − η_t (A_h θ − b_h)
   ```
5. **Aggregation.** Average local iterates across agents to produce the new
   global iterate.

The `(A, b)` tensors are discarded at the end of each round. Only θ-level
quantities are persisted:

- `theta_trajectory`  `[T+1, R, D]` — global iterate at every round (inc. θ₀)
- `theta_final`       `[R, D]`
- `deltas`            `[T, R, N, D]` — per-round, per-agent local update
  `θ_t^n − θ_t` (needed by the bootstrap)
- `step_sizes`        `[T]` — `η_t` per round
- `local_steps`       `[T]` — `H_t` per round

---

## Single-Experiment Procedure

For each configuration `(T, α)`:

1. Run FedLSA training (above) once, producing `n_traj` independent
   trajectories **in parallel** via the trajectory axis `R`.
2. For each main trajectory, generate `n_boot` multiplier bootstrap replicates
   (per-run, independent across runs).
3. For each main trajectory, build a confidence interval from **its own**
   bootstrap sample.
4. Check whether `u^T θ*` lies in each per-trajectory CI; report the
   fraction of hits as the **coverage**.

The final coverage per configuration is therefore
`(1/n_traj) Σ_r 1[ u^T θ* ∈ CI_r ]`.

---

## Confidence Interval Construction (`inference.py`)

The multiplier bootstrap replicates are built as

```
θ̃^b_r = (1/N) Σ_t Σ_n  e^b_{t,n} · Δ_{t,r,n}
e^b_{t,n} = 1 + (w − E[w]) / √Var[w],   w ~ Beta(0.5, 2)
```

so that each `e^b` has mean 1 and variance 1 and the bootstrap iterates
satisfy `E_b[θ̃^b_r | Δ] = θ_T_r`.

Two CIs are returned per trajectory:

### 1. Empirical-Quantile CI

```
[ q_{α/2}(θ̃^b_r) ,  q_{1−α/2}(θ̃^b_r) ]
```

### 2. η-Normalized Normal CI

```
σ̂_r = std_b( η_t^{−1/2} · (θ̃^b_r − θ_T_r) )
CI_r = θ_T_r ± z_{1−α/2} · σ̂_r
```

where `η_t` is the step size at the final round (`step_sizes[-1]`). The
normalization by `η_t^{−1/2}` is intended to make the estimated scale
stabilize as `T → ∞` under the FedLSA CLT.

The projection onto `u` is applied **before** the bootstrap: `deltas` are
projected to a scalar so the whole CI machinery runs in 1-D, giving scalar
CIs directly comparable with `u^T θ*`.

---

## Evaluation Metrics (`run_na_fedlsa.py::run_one_trajectory_length`)

For each `(T, α)`, report:

- `bias`        — `|mean_r(u^T θ̂_r) − u^T θ*|`
- `cov_q`       — empirical-quantile CI coverage
- `cov_n`       — η-normalized normal CI coverage
- `ci_q_width`  — mean width of the quantile CIs
- `ci_n_width`  — mean width of the normal CIs

---

## Notes

- `u` is sampled once at the start of `main()` and fixed across the entire
  sweep.
- Each configuration `(T, α)` is one experiment; the `R = n_traj`
  trajectories inside it are independent and parallelized on the trajectory axis.
- Bootstrap samples are independent across main trajectories.
- Variable local steps (`γ_H > 0`) are supported; the jitted single-round
  function caches one specialization per distinct `H_t`.
- The reference value `θ*` is taken directly from `garnet.thetalim`, not
  recomputed.
- A plug-in / empirical-covariance baseline is **not** implemented yet and
  was removed from the code; it may be added later.
