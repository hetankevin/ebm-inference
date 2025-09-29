#!/usr/bin/env python3
"""Fit InferableEBM on the Obesity dataset with CLI hyperparameters and report test MSE.

Usage example:
  python experiments/obesity_inferable_mse.py \
    --seed 0 --test-size 0.2 \
    --learning-rate 1.0 --max-rounds 200 \
    --min-samples-leaf 5 --reg-lambda 0.01 --max-leaves 2 
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Ensure repository root is on sys.path for local inferable_ebm_regressor module
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inferable_ebm_regressor import InferableEBMRegressor  # noqa: E402


def _sanitize_target(X: pd.DataFrame, y: pd.Series):
    mask = y.replace([np.inf, -np.inf], np.nan).notna()
    if not mask.all():
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
    return X, y


def load_obesity(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Obesity CSV not found at {csv_path}. Place the dataset there (see experiments/optuna_mse.py)."
        )
    df = pd.read_csv(csv_path, encoding="latin-1")
    if "Weight" not in df.columns:
        raise ValueError("Expected 'Weight' column in obesity CSV as the regression target.")
    y = df.pop("Weight")
    X, y = _sanitize_target(df, y)
    return X, y


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    # Data/split
    p.add_argument("--data-path", type=Path, default=_ROOT / "data" / "uci" / "ObesityData.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.2)

    # Core InferableEBM hyperparameters (align with src/_ebm.py defaults)
    p.add_argument("--learning-rate", type=float, default=1)
    p.add_argument("--max-rounds", type=int, default=100)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--reg-lambda", type=float, default=0.01)
    p.add_argument("--max-leaves", type=int, default=4)
    p.add_argument("--subsample-rate", type=float, default=1.0)
    p.add_argument("--truncation", type=float, default=100.0)
    p.add_argument("--warmup-rounds", type=float, default=0.2)
    p.add_argument("--max-bins-auto", type=int, default=255)
    p.add_argument(
        "--auto-bins-scheme",
        type=str,
        choices=["quantile", "cube", "count"],
        default="cube",
    )
    p.add_argument("--max-bins", type=int, default=0)
    # Inference mode and compute
    p.add_argument("--n-jobs", type=int, default=-2)
    p.add_argument("--bin-level-inference", dest="bin_level_inference", action="store_true")
    p.add_argument("--no-bin-level-inference", dest="bin_level_inference", action="store_false")
    p.set_defaults(bin_level_inference=True)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # Load data
    X, y = load_obesity(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    # Build model
    ebm = InferableEBMRegressor(
        learning_rate=args.learning_rate,
        max_rounds=args.max_rounds,
        min_samples_leaf=args.min_samples_leaf,
        reg_lambda=args.reg_lambda,
        max_leaves=args.max_leaves,
        subsample_rate=args.subsample_rate,
        truncation=args.truncation,
        warmup_rounds=args.warmup_rounds,
        max_bins_auto=args.max_bins_auto,
        auto_bins_scheme=args.auto_bins_scheme,
        bin_level_inference=args.bin_level_inference,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )

    # Fit and evaluate
    ebm.fit(X_train, y_train)
    preds = ebm.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))
    print(f"Test MSE: {mse:.6f}")


if __name__ == "__main__":
    main()
