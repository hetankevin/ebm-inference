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
    return 2*np.sin(2 * np.pi * x) + 1 * x ** 2


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
        parser.add_argument("--n-train", type=int, default=1000, help="Number of training samples")
        parser.add_argument("--n-eval", type=int, default=400, help="Number of evaluation samples for coverage stats")
        parser.add_argument("--noise", type=float, default=1.0, help="Noise standard deviation")
        parser.add_argument("--level", type=float, default=0.95, help="Two-sided coverage level")
        parser.add_argument("--max-rounds", type=int, default=100)
        parser.add_argument("--max-bins", type=int, default=0)
        parser.add_argument("--max-leaves", type=int, default=2**5)
        parser.add_argument("--learning-rate", type=int, default=1)
        parser.add_argument("--subsample-rate", dest="subsample_rate", type=float, default=1.)
        parser.add_argument("--truncation", type=float, default=100.0)
        parser.add_argument("--warmup-rounds", type=int, default=0)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--leave-one-out", type=bool, default=False)
        parser.add_argument("--calibrate-intervals", default=True, action="store_true", help="Calibrate prediction intervals on a validation split")
        parser.add_argument(
            "--propagate-calibration",
            default=False,
            action="store_true",
            help="Apply the prediction-interval calibration factor to confidence and reproduction intervals",
        )
        parser.add_argument(
            "--auto-bins-scheme",
            choices=["quantile", "cube", "count"],
            default="quantile",
            help="Automatic numeric binning policy (default: quantile)",
        )
        parser.add_argument("--out", type=str, default="plots/coverage_1d_plot.png")
        parser.add_argument("--show", action="store_true")
        args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x_full, y_full = simulate_data(args.n_train, args.noise, rng)

    if args.calibrate_intervals and args.n_train > 1:
        perm = rng.permutation(args.n_train)
        n_cal = max(1, int(0.2 * args.n_train))
        cal_idx = perm[:n_cal]
        train_idx = perm[n_cal:]
        x_train = x_full[train_idx]
        y_train = y_full[train_idx]
        x_cal = x_full[cal_idx]
        y_cal = y_full[cal_idx]
    else:
        x_train = x_full
        y_train = y_full
        x_cal = None
        y_cal = None

    x_eval, y_eval = simulate_data(args.n_eval, args.noise, rng)

    estimator = InferableEBMRegressor(
        max_rounds=args.max_rounds,
        max_bins=args.max_bins,
        max_leaves = args.max_leaves,
        warmup_rounds = args.warmup_rounds,
        learning_rate = args.learning_rate,
        subsample_rate=args.subsample_rate,
        leave_one_out = args.leave_one_out,
        truncation=args.truncation,
        random_state=args.seed,
        auto_bins_scheme=args.auto_bins_scheme,
        n_jobs=-2,
    )
    estimator.fit(x_train, y_train)

    sigma_override = None
    if args.calibrate_intervals and x_cal is not None:
        resid_cal = y_cal - estimator.predict(x_cal)
        resid_cal = resid_cal[np.isfinite(resid_cal)]
        if resid_cal.size:
            sigma_override = float(np.std(resid_cal, ddof=1))
        if sigma_override is None or not np.isfinite(sigma_override) or sigma_override <= 0:
            sigma_override = None
        else:
            estimator.calibrate_intervals(
                x_cal,
                y_cal,
                level=args.level,
                mode="prediction",
                sigma=sigma_override,
                propagate_to_ci_ri=args.propagate_calibration,
            )

    grid = np.linspace(0.0, 1.0, 500)
    X_grid = grid[:, None]
    preds_grid = estimator.predict(X_grid)

    ci_l, ci_u, _ = estimator.predict_intervals(
        X_grid,
        level=args.level,
        mode="confidence",
        sigma=sigma_override,
    )
    ri_l, ri_u, _ = estimator.predict_intervals(
        X_grid,
        level=args.level,
        mode="reproduction",
        sigma=sigma_override,
    )
    pi_l, pi_u, _ = estimator.predict_intervals(
        X_grid,
        level=args.level,
        mode="prediction",
        sigma=sigma_override,
    )

    # Coverage on evaluation samples
    X_eval = x_eval
    ci_l_eval, ci_u_eval, _ = estimator.predict_intervals(
        X_eval,
        level=args.level,
        mode="confidence",
        sigma=sigma_override,
    )
    ri_l_eval, ri_u_eval, _ = estimator.predict_intervals(
        X_eval,
        level=args.level,
        mode="reproduction",
        sigma=sigma_override,
    )
    pi_l_eval, pi_u_eval, _ = estimator.predict_intervals(
        X_eval,
        level=args.level,
        mode="prediction",
        sigma=sigma_override,
    )

    coverage_ci = float(np.mean((true_function(x_eval.ravel()) >= ci_l_eval) & (true_function(x_eval.ravel()) <= ci_u_eval)))
    coverage_pi = float(np.mean((y_eval >= pi_l_eval) & (y_eval <= pi_u_eval)))

    rng_repro = np.random.default_rng(args.seed + 17)
    x_new, y_new = simulate_data(args.n_train, args.noise, rng_repro)
    estimator_retrain = InferableEBMRegressor(
        max_rounds=args.max_rounds,
        max_bins=args.max_bins,
        max_leaves = args.max_leaves,
        warmup_rounds = args.warmup_rounds,
        learning_rate = args.learning_rate,
        leave_one_out = args.leave_one_out,
        subsample_rate=args.subsample_rate,
        truncation=args.truncation,
        random_state=args.seed,
        auto_bins_scheme=args.auto_bins_scheme,
        n_jobs=-2,
    )
    estimator_retrain.fit(x_new, y_new)
    preds_retrain_eval = estimator_retrain.predict(X_eval)
    coverage_ri = float(np.mean((preds_retrain_eval >= ri_l_eval) & (preds_retrain_eval <= ri_u_eval)))

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
