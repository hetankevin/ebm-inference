#!/usr/bin/env python3
"""Generate 1D CI/RI/PI plots for f(x)=sin(2πx)+0.5x² with coverage stats."""
import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for this script") from exc

plt.style.use("matplotlibrc")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from inferable_ebm_regressor import InferableEBMRegressor  # noqa: E402

COLOR_CI = "#00BEFF"
COLOR_PI = "#F8766D"
COLOR_RI = "#7CAE00"


def true_function(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return 2*np.sin(2 * np.pi * x) + 5 * x ** 2


def simulate_data(n: int, noise: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0.0, 1.0, size=n)
    y = true_function(x) + rng.normal(scale=noise, size=n)
    return x.reshape(-1, 1), y


def plot_panel(ax, x_train, y_train, grid, preds, lower, upper, mode_label, coverage, color):
    ax.scatter(x_train, y_train, c="0.6", alpha=0.35, s=18, label="Training Data")
    ax.plot(grid, preds, color=color, linewidth=2.0, label="Prediction")
    ax.fill_between(grid, lower, upper, color=color, alpha=0.25, label=mode_label)
    ax.plot(grid, true_function(grid), color="black", linestyle="--", linewidth=2.0, label="True Function")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-6.0, 6.0)
    ax.set_title(mode_label, fontsize=16)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(
        0.02,
        0.93,
        f"Coverage: {coverage:.2%}",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85, boxstyle="round,pad=0.3"),
    )
    ax.legend(framealpha=0.85, facecolor="white", edgecolor="0.7", loc="lower left", fontsize=10)


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--n-train", type=int, default=400, help="Number of training samples")
        parser.add_argument("--n-eval", type=int, default=400, help="Number of evaluation samples for coverage stats")
        parser.add_argument("--noise", type=float, default=1.0, help="Noise standard deviation")
        parser.add_argument("--level", type=float, default=0.95, help="Two-sided coverage level")
        parser.add_argument("--max-rounds", type=int, default=200)
        parser.add_argument("--max-bins", type=int, default=64)
        parser.add_argument("--subsample-rate", dest="subsample_rate", type=float, default=1.0)
        parser.add_argument("--truncation", type=float, default=3.0)
        parser.add_argument("--bin-level-inference", action="store_true")
        parser.add_argument(
            "--sample-level-inference",
            dest="bin_level_inference",
            action="store_false"
        )
        parser.add_argument(
            "--inference-space",
            choices=["auto", "samples", "bins"],
            default="auto",
        )
        parser.add_argument("--nystrom", dest='nystrom', action="store_true", help="Enable Nyström approximation")
        parser.add_argument("--no-nystrom", dest='nystrom', action="store_false", help="Disable Nyström approximation")
        parser.add_argument("--nystrom-rank", type=int, default=64)
        parser.add_argument("--nystrom-ridge", type=float, default=1e-6)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--out", type=str, default="plots/coverage_1d_plot.png")
        parser.add_argument("--show", action="store_true")
        args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x_train, y_train = simulate_data(args.n_train, args.noise, rng)
    x_eval, y_eval = simulate_data(args.n_eval, args.noise, rng)

    estimator = InferableEBMRegressor(
        max_rounds=args.max_rounds,
        max_bins=args.max_bins,
        subsample_rate=args.subsample_rate,
        truncation=args.truncation,
        random_state=args.seed,
        bin_level_inference=args.bin_level_inference,
        use_nystrom=args.nystrom,
        nystrom_rank=args.nystrom_rank,
        nystrom_ridge=args.nystrom_ridge,
    )
    estimator.fit(x_train, y_train)

    grid = np.linspace(0.0, 1.0, 500)
    X_grid = grid[:, None]
    preds_grid = estimator.predict(X_grid)

    inference_space = None if args.inference_space == "auto" else args.inference_space
    ci_l, ci_u, _ = estimator.predict_intervals(X_grid, level=args.level, mode="confidence", inference_space=inference_space)
    ri_l, ri_u, _ = estimator.predict_intervals(X_grid, level=args.level, mode="reproduction", inference_space=inference_space)
    pi_l, pi_u, _ = estimator.predict_intervals(X_grid, level=args.level, mode="prediction", inference_space=inference_space)

    # Coverage on evaluation samples
    X_eval = x_eval
    ci_l_eval, ci_u_eval, _ = estimator.predict_intervals(X_eval, level=args.level, mode="confidence", inference_space=inference_space)
    ri_l_eval, ri_u_eval, preds_eval = estimator.predict_intervals(X_eval, level=args.level, mode="reproduction", inference_space=inference_space)
    pi_l_eval, pi_u_eval, _ = estimator.predict_intervals(X_eval, level=args.level, mode="prediction", inference_space=inference_space)

    coverage_ci = float(np.mean((true_function(x_eval.ravel()) >= ci_l_eval) & (true_function(x_eval.ravel()) <= ci_u_eval)))
    coverage_ri = float(np.mean((preds_eval >= ri_l_eval) & (preds_eval <= ri_u_eval)))
    coverage_pi = float(np.mean((y_eval >= pi_l_eval) & (y_eval <= pi_u_eval)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_panel(axes[0], x_train.ravel(), y_train, grid, preds_grid, ci_l, ci_u, "Confidence Intervals", coverage_ci, COLOR_CI)
    plot_panel(axes[1], x_train.ravel(), y_train, grid, preds_grid, pi_l, pi_u, "Prediction Intervals", coverage_pi, COLOR_PI)
    plot_panel(axes[2], x_train.ravel(), y_train, grid, preds_grid, ri_l, ri_u, "Reproduction Intervals", coverage_ri, COLOR_RI)
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(args.out, dpi=300)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
