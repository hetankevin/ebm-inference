#!/usr/bin/env python3
"""Plot per-feature predictions and confidence intervals against the Friedman oracle."""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('matplotlibrc')

# Ensure repository root on path so we can import the locally patched estimator.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from inferable_ebm_regressor import InferableEBMRegressor  # noqa: E402
from experiments.coverage_rates import make_friedman  # noqa: E402


def friedman_true_function(X: np.ndarray) -> np.ndarray:
    """Return the noise-free Friedman target for samples ``X``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix with columns in ``[0, 1]``.

    Returns
    -------
    ndarray of shape (n_samples,)
        Deterministic Friedman regression target.
    """
    X = np.asarray(X, dtype=float)
    f =  -5 + (10*np.sin(np.pi*X[:,0]) - 5*np.cos(np.pi*X[:,1])
          + 20*(X[:,2]-0.5)**2 + 10*X[:,3] - 5*X[:,4])
    return f


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4000, help="Number of training samples")
    parser.add_argument("--noise", type=float, default=1.0, help="Standard deviation of Gaussian noise")
    parser.add_argument("--level", type=float, default=0.95, help="Two-sided coverage level")
    parser.add_argument("--mode", choices=["confidence", "prediction", "reproduction"], default="confidence")
    parser.add_argument("--n-points", type=int, default=200, help="Grid size per feature")
    parser.add_argument("--inference-space", choices=["auto", "samples", "bins"], default="auto")
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--bin-level", action="store_true", help="Use bin-level inference when fitting")
    parser.add_argument("--output", type=str, default="feature_interval_plots.png", help="Output figure path")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(0)
    X, f_true = make_friedman(args.n, rng)
    y = f_true + rng.normal(scale=args.noise, size=args.n)

    model = InferableEBMRegressor(
        max_rounds=args.max_rounds,
        max_bins=args.n_bins,
        subsample_rate=1.0,
        truncation=3.0,
        random_state=0,
        bin_level_inference=args.bin_level,
    )
    model.fit(X, y)

    baseline = np.mean(X, axis=0)
    n_features = X.shape[1]
    grid = np.linspace(0.0, 1.0, args.n_points)

    fig, axes = plt.subplots(
        n_features,
        1,
        figsize=(8, 2.5 * n_features),
        sharex=True,
        constrained_layout=True,
    )
    if n_features == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        X_grid = np.repeat(baseline[None, :], args.n_points, axis=0)
        X_grid[:, j] = grid

        lower, upper, preds = model.predict_feature_intervals(
            j,
            grid,
            level=args.level,
            mode=args.mode,
            inference_space=args.inference_space,
            include_intercept=True,
        )
        oracle = friedman_true_function(X_grid)

        ax.plot(grid, oracle, label="True function", color="red", linewidth=1.5)
        ax.plot(grid, preds, label="EBM prediction", linewidth=1.5)
        ax.fill_between(
            grid,
            lower,
            upper,
            color="tab:blue",
            alpha=0.2,
            label=f"{args.level*100:.0f}% interval",
        )
        ax.set_ylabel(f"Feature {j}")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="best", fontsize="small")

    axes[-1].set_xlabel("Feature value")
    fig.suptitle("Feature-wise Predictions with Confidence Intervals")

    fig.savefig(args.output, dpi=200)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
