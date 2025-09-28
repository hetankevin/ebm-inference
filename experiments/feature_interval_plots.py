#!/usr/bin/env python3
"""Plot per-feature predictions and confidence intervals against the Friedman oracle."""
import argparse
import os
import sys
from tqdm import tqdm
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
    parser.add_argument("--n", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--noise", type=float, default=2, help="Standard deviation of Gaussian noise")
    parser.add_argument("--level", type=float, default=0.95, help="Two-sided coverage level")
    parser.add_argument("--mode", choices=["confidence", "prediction", "reproduction"], default="confidence")
    parser.add_argument("--n-points", type=int, default=200, help="Grid size per feature")
    parser.add_argument("--inference-space", choices=["auto", "samples", "bins"], default="auto")
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--bin-level-inference", default=True, help="Use bin-level inference when fitting")
    parser.add_argument("--output", type=str, default="plots/feature_interval_plots.png", help="Output figure path")
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
        bin_level_inference=args.bin_level_inference,
    )
    model.fit(X, y)
    print('Model fitted')

    baseline = np.mean(X, axis=0)
    n_features = X.shape[1]
    grid = np.linspace(0.0, 1.0, args.n_points)

    fig, axes = plt.subplots(
        1,
        n_features,
        figsize=(10, 3),
        sharey=True,
        constrained_layout=True,
    )
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.reshape(-1)
    print('Forming intervals')
    for j, ax in tqdm(enumerate(axes)):
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

        # Align the marginal mean with the oracle to resolve the GAM identification ambiguity.
        oracle_mean = float(np.mean(oracle))
        pred_mean = float(np.mean(preds))
        shift = oracle_mean - pred_mean
        if shift != 0.0:
            preds = preds + shift
            lower = lower + shift
            upper = upper + shift

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

        coverage = float(np.mean((oracle >= lower) & (oracle <= upper)))
        ax.text(
            0.02,
            0.95,
            f"Coverage: {coverage*100:.2f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.8},
        )
        ax.set_xlabel(f"Feature {j+1}", fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="lower left", fontsize="small")

    axes[0].set_ylabel("Value", fontsize=14)
    #fig.suptitle("Feature-wise Predictions with Confidence Intervals")
    plt.tight_layout()
    fig.savefig(args.output, dpi=300)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
