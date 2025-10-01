#!/usr/bin/env python3
"""Fit InferableEBM on the Obesity dataset, report test MSE, and optionally
overlay per-feature effects from InferableEBM and standard EBM on the same
axes, saved under `plots/`.

Usage example:
  python experiments/obesity_inferable_mse.py \
    --seed 0 --test-size 0.2 \
    --learning-rate 0.0625 --max-rounds 200 \
    --min-samples-leaf 2 --reg-lambda 0.01 --max-leaves 16 \
    --warmup-rounds 100 --effects-plot plots/obesity_feature_effects.png --compare-effects
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('matplotlibrc')
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

EFFECT_INTERVAL_LEVEL = 0.95


# Ensure repository root is on sys.path for local inferable_ebm_regressor module
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inferable_ebm_regressor import InferableEBMRegressor  # noqa: E402
try:
    from interpret.glassbox import ExplainableBoostingRegressor  # type: ignore
except Exception:  # pragma: no cover
    # Fallback: after importing inferable_ebm_regressor, interpret path is available
    import importlib
    _ebm_mod = importlib.import_module("interpret.glassbox._ebm._ebm")
    ExplainableBoostingRegressor = getattr(_ebm_mod, "ExplainableBoostingRegressor")


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
    p.add_argument("--max-rounds", type=int, default=200)
    p.add_argument("--min-samples-leaf", type=int, default=16)
    p.add_argument("--reg-lambda", type=float, default=100)
    p.add_argument("--max-leaves", type=int, default=6)
    p.add_argument("--subsample-rate", type=float, default=1)
    p.add_argument("--leave-one-out", type=bool, default=False)
    p.add_argument("--truncation", type=float, default=10.0)
    p.add_argument("--warmup-rounds", type=int, default=0)
    p.add_argument("--max-bins-auto", type=int, default=255)
    p.add_argument(
        "--auto-bins-scheme",
        type=str,
        choices=["quantile", "cube", "count"],
        default="quantile",
    )
    p.add_argument("--max-bins", type=int, default=0)
    # Compute
    p.add_argument("--n-jobs", type=int, default=-2)
    # Effect comparison / plotting
    p.add_argument("--compare-effects", default=True, action="store_true", help="Also fit EBM and save overlaid feature effects plot")
    p.add_argument("--effects-plot", type=Path, default=_ROOT / "plots" / "obesity_feature_effects.png")
    p.add_argument("--predict-feature-intervals", type=bool, default=True)
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
        max_bins=args.max_bins,
        auto_bins_scheme=args.auto_bins_scheme,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        leave_one_out=args.leave_one_out,
    )

    # Fit and evaluate InferableEBM
    ebm.fit(X_train, y_train)
    preds = ebm.predict(X_test)
    mse = float(root_mean_squared_error(y_test, preds))
    print(f"Test RMSE: {mse:.6f}")

    # Fit and evaluate standard EBM as well
    ebm_std = ExplainableBoostingRegressor(
        max_rounds=args.max_rounds,
        outer_bags=4,
        interactions=0.0,
        random_state=args.seed,
        n_jobs=1,
    )
    ebm_std.fit(X_train, y_train)
    preds_std = ebm_std.predict(X_test)
    mse_std = float(root_mean_squared_error(y_test, preds_std))
    print(f"EBM Test RMSE: {mse_std:.6f}")

    if getattr(args, "compare_effects", False):
        compare_feature_effects(
            X_train=X_train,
            y_train=y_train,
            inferable=ebm,
            ebm_prefit=ebm_std,
            effects_path=args.effects_plot,
            seed=args.seed,
            max_rounds=args.max_rounds,
        )


def _baseline_row(df: pd.DataFrame) -> pd.Series:
    base = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            base[col] = float(s.median())
        else:
            mode = s.mode(dropna=True)
            base[col] = mode.iloc[0] if not mode.empty else (s.dropna().iloc[0] if s.dropna().size else None)
    return pd.Series(base, index=df.columns)


def _numeric_grid(s: pd.Series, n_points: int = 200) -> np.ndarray:
    lo = s.quantile(0.01) if s.notna().any() else 0.0
    hi = s.quantile(0.99) if s.notna().any() else 1.0
    lo = float(lo if np.isfinite(lo) else s.min() if np.isfinite(s.min()) else 0.0)
    hi = float(hi if np.isfinite(hi) else s.max() if np.isfinite(s.max()) else 1.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = 0.0, 1.0
    return np.linspace(lo, hi, n_points)


def compare_feature_effects(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    inferable: InferableEBMRegressor,
    effects_path: Path,
    seed: int = 0,
    max_rounds: int = 200,
    ebm_prefit: "ExplainableBoostingRegressor" | None = None,
) -> None:
    # Fit standard EBM for comparison using similar training setup if not provided
    ebm = ebm_prefit
    if ebm is None:
        ebm = ExplainableBoostingRegressor(max_rounds=max_rounds, outer_bags=4, random_state=seed, n_jobs=1)
        ebm.fit(X_train, y_train)

    n_features = X_train.shape[1]
    baseline = _baseline_row(X_train)
    inferable_interval_fn = getattr(inferable, "predict_feature_intervals", None)
    ebm_interval_fn = getattr(ebm, "predict_feature_intervals", None)
    labels_map = {
        "Gender": "Gender",
        "Age": "Age",
        "Height": "Height",
        "family_history_with_overweight": "Family History of Being Overweight",
        "FAVC": "Frequent High Calorie Food Intake",
        "FCVC": "Frequent Consumption of Vegetables",
        "NCP": "Number of Daily Meals",
        "CAEC": "Food Between Meals",
        "SMOKE": "Smoker",
        "CH2O": "Water Intake",
        "SCC": "Calorie Monitoring",
        "FAF": "Physical Activity Frequency",
        "TUE": "Time Using Technology",
        "CALC": "Alcohol Intake",
        "MTRANS": "Method of Transportation",
        "NObeyesdad": "Obesity Level",
    }

    def _pretty_label(col: str) -> str:
        return labels_map.get(col, col.replace("_", " ").title())

    def _plot_single(ax, col: str) -> None:
        s = X_train[col]
        feat_idx = X_train.columns.get_loc(col)
        if pd.api.types.is_numeric_dtype(s):
            grid = _numeric_grid(s)
            Xg = pd.DataFrame([baseline.values] * len(grid), columns=baseline.index)
            Xg[col] = grid
            y_ie = inferable.predict(Xg)
            y_ebm = ebm.predict(Xg)
            y_ie = y_ie - float(np.mean(y_ie))
            y_ebm = y_ebm - float(np.mean(y_ebm))
            (line_ie,) = ax.plot(grid, y_ie, label="InferableEBM", linewidth=1.8)
            (line_ebm,) = ax.plot(grid, y_ebm, label="EBM", linewidth=1.8, linestyle="--")
            if inferable_interval_fn is not None:
                try:
                    ci_l, ci_u, ci_pred = inferable_interval_fn(
                        feat_idx,
                        grid,
                        level=EFFECT_INTERVAL_LEVEL,
                        mode="confidence",
                        include_intercept=False,
                    )
                except Exception as exc:
                    print(exc)
                else:
                    ci_pred = np.asarray(ci_pred, dtype=float)
                    offset = float(np.mean(ci_pred))
                    ci_l_center = np.asarray(ci_l, dtype=float) - offset
                    ci_u_center = np.asarray(ci_u, dtype=float) - offset
                    ax.fill_between(grid, ci_l_center, ci_u_center, color=line_ie.get_color(), alpha=0.3)
            if ebm_interval_fn is not None:
                try:
                    ebm_ci_l, ebm_ci_u, ebm_ci_pred = ebm_interval_fn(
                        feat_idx,
                        grid,
                        level=EFFECT_INTERVAL_LEVEL,
                        mode="confidence",
                        include_intercept=False,
                    )
                except Exception as exc:
                    print(exc)
                else:
                    ebm_ci_pred = np.asarray(ebm_ci_pred, dtype=float)
                    offset_std = float(np.mean(ebm_ci_pred))
                    ebm_ci_l_center = np.asarray(ebm_ci_l, dtype=float) - offset_std
                    ebm_ci_u_center = np.asarray(ebm_ci_u, dtype=float) - offset_std
                    ax.fill_between(grid, ebm_ci_l_center, ebm_ci_u_center, color=line_ebm.get_color(), alpha=0.3)
        else:
            cats = s.dropna().unique().tolist()
            try:
                cats = sorted(cats)
            except Exception:
                pass
            if not cats:
                ax.axis("off")
                return
            Xg = pd.DataFrame([baseline.values] * len(cats), columns=baseline.index)
            Xg[col] = cats
            y_ie = inferable.predict(Xg)
            y_ebm = ebm.predict(Xg)
            y_ie = y_ie - float(np.mean(y_ie))
            y_ebm = y_ebm - float(np.mean(y_ebm))
            xs = np.arange(len(cats))
            (line_ie,) = ax.plot(xs, y_ie, marker="o", label="InferableEBM", linewidth=1.8)
            (line_ebm,) = ax.plot(xs, y_ebm, marker="s", label="EBM", linewidth=1.8, linestyle="--")
            if inferable_interval_fn is not None:
                try:
                    ci_l, ci_u, ci_pred = inferable_interval_fn(
                        feat_idx,
                        cats,
                        level=EFFECT_INTERVAL_LEVEL,
                        mode="confidence",
                        include_intercept=False,
                    )
                except Exception as exc:
                    print(exc)
                else:
                    ci_pred = np.asarray(ci_pred, dtype=float)
                    offset = float(np.mean(ci_pred))
                    ci_l_center = np.asarray(ci_l, dtype=float) - offset
                    ci_u_center = np.asarray(ci_u, dtype=float) - offset
                    ax.fill_between(xs, ci_l_center, ci_u_center, color=line_ie.get_color(), alpha=0.3)
            if ebm_interval_fn is not None:
                try:
                    ebm_ci_l, ebm_ci_u, ebm_ci_pred = ebm_interval_fn(
                        feat_idx,
                        cats,
                        level=EFFECT_INTERVAL_LEVEL,
                        mode="confidence",
                        include_intercept=False,
                    )
                except Exception as exc:
                    print(exc)
                else:
                    ebm_ci_pred = np.asarray(ebm_ci_pred, dtype=float)
                    offset_std = float(np.mean(ebm_ci_pred))
                    ebm_ci_l_center = np.asarray(ebm_ci_l, dtype=float) - offset_std
                    ebm_ci_u_center = np.asarray(ebm_ci_u, dtype=float) - offset_std
                    ax.fill_between(xs, ebm_ci_l_center, ebm_ci_u_center, color=line_ebm.get_color(), alpha=0.3)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in cats], rotation=30, ha="right")
        ax.set_xlabel(_pretty_label(col))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    def _render_grid(columns: list[str], n_cols: int, output_path: Path) -> None:
        if not columns:
            return
        n_cols = max(1, min(n_cols, len(columns)))
        n_rows = int(np.ceil(len(columns) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.8 * n_cols, 3.2 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        for idx, col in enumerate(columns):
            r, c = divmod(idx, n_cols)
            ax = axes[r][c]
            _plot_single(ax, col)
            if idx == 0:
                ax.legend(loc="best")
        for idx in range(len(columns), n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            fig.delaxes(axes[r][c])
        for r in range(n_rows):
            axes[r][0].set_ylabel("Effect on Weight")
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    print(X_train.columns)
    feature_list = list(X_train.columns)
    _render_grid(feature_list, min(4, n_features) or 1, effects_path)

    wide_features = ["TUE", "FCVC", "NCP", "FAF"]
    wide_features = [col for col in wide_features if col in X_train.columns]
    if wide_features:
        wide_path = effects_path.parent / "obesity_feature_effects_wide.png"
        _render_grid(wide_features, len(wide_features), wide_path)


if __name__ == "__main__":
    main()
