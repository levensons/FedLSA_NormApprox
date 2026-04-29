"""
Plot empirical coverage vs T for each confidence level, with shaded
±z·SE confidence bands on the coverage estimate. The `*_std` columns
in the input already contain SE = std/√n_traj. Reads
`coverage_arrays.npz` produced by run_na_fedlsa.main(); falls back to
results_sweep.csv if absent.

Usage:
    python plot_coverage.py                          # defaults to coverage_arrays.npz
    python plot_coverage.py coverage_arrays.npz      # explicit npz
    python plot_coverage.py results_sweep.csv        # csv fallback
"""
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_ESTIMATORS = [
    ("cov_q", "cov_q_std", "EQ", "tab:blue"),
    ("cov_n", "cov_n_std", "SDB",   "tab:orange"),
    ("cov_l", "cov_l_std", "PE",     "tab:green"),
]


def _load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _load_csv(path):
    df = pd.read_csv(path).sort_values(["alpha", "T"])
    Ts = np.array(sorted(df["T"].unique()), dtype=int)
    alphas = np.array(sorted(df["alpha"].unique()), dtype=float)
    out = {"T": Ts, "alphas": alphas}
    for col in ("cov_q", "cov_n", "cov_l", "cov_q_std", "cov_n_std", "cov_l_std"):
        if col in df.columns:
            out[col] = df.pivot(index="T", columns="alpha", values=col).reindex(
                index=Ts, columns=alphas).values
    return out


def plot_coverage(path: str = "coverage_arrays.npz",
                  out_png: str = "coverage_vs_T.png",
                  z: float = 1.96):
    if path.endswith(".npz"):
        data = _load_npz(path)
    else:
        data = _load_csv(path)
    Ts = data["T"]
    alphas = data["alphas"]

    fig, axes = plt.subplots(
        1, len(alphas), figsize=(18, 6), sharex=True
    )
    if len(alphas) == 1:
        axes = [axes]

    for ai, (ax, a) in enumerate(zip(axes, alphas)):
        for col, se_col, label, color in _ESTIMATORS:
            if col not in data:
                continue
            mean = data[col][:, ai]
            line, = ax.plot(Ts, mean, label=label, color=color,
                            marker="o", markersize=3, linewidth=1.4)
            if se_col in data:
                se = data[se_col][:, ai]
                lo = mean - z * se
                hi = mean + z * se
                ax.fill_between(Ts, lo, hi, color=color, alpha=0.3,
                                linewidth=0)

        ax.axhline(1.0 - a, linestyle="--", color="black", linewidth=1.4,
                   label=f"α = {1 - a:.2f}")
        ax.set_xlabel("T", fontsize=18)
        ax.set_title(f"α = {1 - a:.2f}", fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_xticks(Ts[::3])
        # ax.set_xticks(np.arange(Ts[0], Ts[-1] + 1, 2000))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=18, loc="lower right")

    # axes[-1].set_xlabel("Communication rounds  T")
    axes[0].set_ylabel("Coverage", fontsize=20)
    # fig.suptitle(f"Empirical coverage vs T  (bands = ±{z:.2f}·SE)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    elif os.path.exists("coverage_arrays_gamma_H=0.2.npz"):
        path = "coverage_arrays_gamma_H=0.2.npz"
    # else:
    #     path = "results_sweep.csv"
    out_png = "coverage_vs_T_gamma_H=0.2.png"
    plot_coverage(path, out_png)
