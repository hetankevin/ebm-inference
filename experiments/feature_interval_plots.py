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
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--n-bins", type=int, default=0)
    parser.add_argument("--max-leaves", type=int, default=2)
    parser.add_argument("--calibrate-intervals", action="store_true", help="Calibrate prediction intervals on a validation split")
    parser.add_argument(
        "--propagate-calibration",
        action="store_true",
        help="Apply the prediction-interval calibration factor to confidence and reproduction intervals",
    )
    parser.add_argument(
        "--auto-bins-scheme",
        choices=["quantile", "cube", "count"],
        default="quantile",
        help="Automatic numeric binning policy (default: quantile)",
    )
    parser.add_argument("--output", type=str, default="plots/feature_interval_plots.png", help="Output figure path")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(0)
    X, f_true = make_friedman(args.n, rng)
    y = f_true + rng.normal(scale=args.noise, size=args.n)

    if args.calibrate_intervals and args.n > 1:
        perm = rng.permutation(args.n)
        n_cal = max(1, int(0.2 * args.n))
        cal_idx = perm[:n_cal]
        train_idx = perm[n_cal:]
        X_cal = X[cal_idx]
        y_cal = y[cal_idx]
        X_train = X[train_idx]
        y_train = y[train_idx]
    else:
        X_train, y_train = X, y
        X_cal = None
        y_cal = None

    model = InferableEBMRegressor(
        max_rounds=args.max_rounds,
        max_bins=args.n_bins,
        subsample_rate=1.0,
        truncation=3.0,
        random_state=0,
        max_leaves=args.max_leaves,
        auto_bins_scheme=args.auto_bins_scheme,
    )
    model.fit(X_train, y_train)
    print('Model fitted')

    sigma_override = None
    if args.calibrate_intervals and X_cal is not None:
        resid_cal = y_cal - model.predict(X_cal)
        resid_cal = resid_cal[np.isfinite(resid_cal)]
        if resid_cal.size:
            sigma_override = float(np.std(resid_cal, ddof=1))
        if sigma_override is not None and np.isfinite(sigma_override) and sigma_override > 0:
            model.calibrate_intervals(
                X_cal,
                y_cal,
                level=args.level,
                mode="prediction",
                sigma=sigma_override,
                propagate_to_ci_ri=args.propagate_calibration,
            )
        else:
            sigma_override = None

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
            sigma=sigma_override,
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
