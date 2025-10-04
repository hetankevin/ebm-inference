#!/usr/bin/env python3
"""Benchmark multiple regressors with Optuna on the scikit-learn diabetes dataset."""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

import numpy as np
import optuna
import pandas as pd
from optuna import Trial
from optuna.samplers import TPESampler

# scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for this script") from exc

try:
    from interpret.glassbox import ExplainableBoostingRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ExplainableBoostingRegressor import failed") from exc

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inferable_ebm_regressor import InferableEBMRegressor  # noqa: E402

# Fixed plotting order for models
# Only models present in results will be plotted, in this order.
PLOT_MODEL_ORDER = [
    "InferableEBM",
    "EBM",
    "LightGBM",
    "XGBoost",
    "RandomForest",
    "GradientBoosting",
    "ElasticNet",
]

@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series


def load_diabetes_dataset(_n_samples: int, _noise_std: float, _seed: int) -> Dataset:
    """Load the diabetes regression dataset as a pandas DataFrame."""
    data_bundle = load_diabetes(as_frame=True)
    X = data_bundle.data.copy()
    y = data_bundle.target.copy()
    y.name = y.name or "target"
    return Dataset("Diabetes", X, y)


DATASET_LOADERS = {
    "diabetes": load_diabetes_dataset,
}


def _split_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def _make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ))
    return ColumnTransformer(transformers)


def _make_passthrough_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Pass numeric and categorical columns through unchanged.

    Useful for EBM-family models that perform their own binning/encoding.
    """
    transformers = []
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if cat_cols:
        transformers.append(("cat", "passthrough", cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


## Removed: interaction-expansion preprocessor (no longer needed)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = _split_columns(X)
    return _make_preprocessor(num_cols, cat_cols)


def cross_val_score_neg_mse(
    model,
    X,
    y,
    column_groups: Tuple[List[str], List[str]],
    n_splits: int = 2,
    random_state: int = 0,
):
    # Feed raw DataFrame to EBM-family; others get standard preprocessing
    if isinstance(model, (ExplainableBoostingRegressor, InferableEBMRegressor)):
        pipeline = Pipeline([("model", model)])
    else:
        preprocessor = _make_preprocessor(*column_groups)
        pipeline = Pipeline([("pre", preprocessor), ("model", model)])
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, scoring="neg_root_mean_squared_error", cv=cv)
    return scores.mean()


def tune_parameters(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    column_groups: Tuple[List[str], List[str]],
    args,
) -> Dict:
    def objective(trial: Trial):
        if model_name == "RandomForest":
            params = {
                "n_estimators": 500,
                "max_depth": 4,
                "random_state": args.seed,
                'min_samples_leaf' : 1,
                "n_jobs": 1,
            }
            model = RandomForestRegressor(**params)
        elif model_name == "GradientBoosting":
            params = {
                "n_estimators": 500,
                "learning_rate": 0.1,
                'min_samples_split' : 2,
                'min_samples_leaf' : 1,
                "max_depth": 4,
                "random_state": args.seed,
            }
            model = GradientBoostingRegressor(**params)
        elif model_name == "ElasticNet":
            params = {
                "alpha": 1e-2,
                "l1_ratio": 0.1,
                "max_iter": 10000,
                "random_state": args.seed,
            }
            model = ElasticNet(**params)
        elif model_name == "EBM":
            params = {
                "max_bins": 255,
                "max_rounds": 500,
                'interactions' : 0.0,
                'min_samples_leaf' : 1,
                "outer_bags": 1,
                "learning_rate": 0.1,
                "random_state": args.seed,
                'max_leaves' : 2**4,
                'n_jobs' : 1
            }
            model = ExplainableBoostingRegressor(**params)
        elif model_name == "InferableEBM":
            params = {
                "max_bins_auto": 255,
                "max_rounds": 500,
                'warmup_rounds' : 0,
                "subsample_rate": 1,
                "truncation": 2000,
                "random_state": args.seed,
                'n_jobs' : 1,
                "learning_rate": 1,
                'auto_bins_scheme': 'quantile',
                "reg_lambda": 0,
                'min_samples_leaf': 1,
                'max_leaves' : 2**4
            }
            model = InferableEBMRegressor(**params)
        elif model_name == "LightGBM":
            if lgb is None:
                raise optuna.exceptions.TrialPruned()
            params = {
                "n_estimators": 500,
                "learning_rate": 0.1,
                'colsample_bytree' : 0.8,
                'min_samples_split' : 2,
                'min_samples_leaf' : 1,
                "max_depth": 4,
                "random_state": args.seed,
                "n_jobs": 1,
                'verbose' : -1
            }
            model = lgb.LGBMRegressor(**params)
        elif model_name == "XGBoost":
            if xgb is None:
                raise optuna.exceptions.TrialPruned()
            params = {
                "n_estimators": 500,
                "eta": 0.1,
                'colsample_bytree' : 0.8,
                'min_samples_split' : 2,
                'min_samples_leaf' : 1,
                "max_depth": 4,
                "random_state": args.seed,
                "tree_method": "hist",
                'n_jobs' : 1
            }
            model = xgb.XGBRegressor(**params)
        else:
            raise ValueError(model_name)

        score = cross_val_score_neg_mse(
            model,
            X,
            y,
            column_groups,
            n_splits=2,
            random_state=args.seed,
        )
        return score

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    best_params = study.best_params
    # remove fixed keys
    for key in ["n_estimators", "max_rounds"]:
        best_params.pop(key, None)
    return best_params


def instantiate(model_name: str, params: Dict, ensemble_size: Optional[int], args) -> object:
    params = params.copy()
    if model_name == "RandomForest":
        params.update({"n_estimators": ensemble_size, "random_state": args.seed, 'n_jobs' : 1})
        return RandomForestRegressor(**params)
    if model_name == "GradientBoosting":
        params.update({"n_estimators": ensemble_size, "random_state": args.seed})
        return GradientBoostingRegressor(**params)
    if model_name == "ElasticNet":
        params.update({"max_iter": 10000, "random_state": args.seed})
        return ElasticNet(**params)
    if model_name == "EBM":
        params.update({"max_rounds": ensemble_size, "random_state": args.seed, 'n_jobs' : 4})
        return ExplainableBoostingRegressor(**params)
    if model_name == "InferableEBM":
        params.update({
            "max_rounds": ensemble_size,
            "random_state": args.seed,
            'n_jobs' : 1,
        })
        return InferableEBMRegressor(**params)
    if model_name == "LightGBM":
        if lgb is None:
            raise RuntimeError("LightGBM not installed")
        params.update({"n_estimators": ensemble_size, "random_state": args.seed, 'n_jobs' : 1, 'verbose' : -1})
        return lgb.LGBMRegressor(**params)
    if model_name == "XGBoost":
        if xgb is None:
            raise RuntimeError("xgboost not installed")
        params.update({"n_estimators": ensemble_size, "random_state": args.seed, "tree_method": "hist",
                        'n_jobs' : 1})
        return xgb.XGBRegressor(**params)
    raise ValueError(model_name)


def _evaluate_single_trial(
    model_name: str,
    params: Dict,
    column_groups: Tuple[List[str], List[str]],
    ensemble_sizes: List[int],
    trial_split: Tuple[int, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    args_dict: Dict,
    show_progress: bool,
) -> List[float]:
    print('Fitting '+model_name)
    seed, X_train, X_test, y_train, y_test = trial_split
    trial_args = SimpleNamespace(**args_dict)
    trial_args.seed = seed
    trial_mses: List[float] = []
    last_mse: Optional[float] = None
    size_iterable = tqdm(ensemble_sizes, leave=False) if show_progress else ensemble_sizes
    for size in size_iterable:
        if model_name == "ElasticNet" and last_mse is not None:
            trial_mses.append(last_mse)
            continue
        model = instantiate(model_name, params, size, trial_args)
        if model_name in ("EBM", "InferableEBM"):
            pipeline = Pipeline([("model", model)])
        else:
            preprocessor = _make_preprocessor(*column_groups)
            pipeline = Pipeline([("pre", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        last_mse = root_mean_squared_error(y_test, preds)
        trial_mses.append(last_mse)
    return trial_mses


def _evaluate_model_curve(
    dataset_name: str,
    model_name: str,
    params: Dict,
    column_groups: Tuple[List[str], List[str]],
    trial_splits: List[Tuple[int, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
    ensemble_sizes: List[int],
    args_dict: Dict,
    max_workers: int = 1,
    ) -> Tuple[str, str, List[List[float]]]:
    if max_workers is None:
        max_workers = 1
    effective_workers = max(1, min(max_workers, len(trial_splits))) if trial_splits else 1
    mses_by_trial: List[Optional[List[float]]] = [None] * len(trial_splits)

    if effective_workers == 1:
        trial_iterator = tqdm(trial_splits, desc=f"{model_name} trials", leave=False)
        results: List[List[float]] = []
        for trial_split in trial_iterator:
            trial_mses = _evaluate_single_trial(
                model_name,
                params,
                column_groups,
                ensemble_sizes,
                trial_split,
                args_dict,
                show_progress=True,
            )
            results.append(trial_mses)
        return dataset_name, model_name, results

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {}
        for idx, trial_split in enumerate(trial_splits):
            print(idx)
            future = executor.submit(
                _evaluate_single_trial,
                model_name,
                params,
                column_groups,
                ensemble_sizes,
                trial_split,
                args_dict,
                True,
            )
            futures[future] = idx
        for future in as_completed(futures):
            idx = futures[future]
            mses_by_trial[idx] = future.result()

    # mypy hint: all entries filled
    completed = [trial for trial in mses_by_trial if trial is not None]
    return dataset_name, model_name, completed


def plot_results(dataset_name: str, ensemble_sizes: List[int], mse_curves: Dict[str, Dict[str, List[float]]], ax):
    # Plot in a consistent, predefined order regardless of dict insertion order
    for name in PLOT_MODEL_ORDER:
        if name not in mse_curves:
            continue
        stats = mse_curves[name]
        mean = np.asarray(stats.get("mean", []), dtype=float)
        stderr = np.asarray(stats.get("stderr", []), dtype=float)
        if mean.size == 0:
            continue
        if stderr.size != mean.size:
            raise ValueError(f"Standard error for {name} must match ensemble sizes")
        lower = np.maximum(mean - 2*stderr, 1e-8)
        upper = mean + 2*stderr
        line_style = "--" if name == "ElasticNet" else "-"
        (line,) = ax.plot(ensemble_sizes, mean, label=name, linewidth=2, linestyle=line_style)
        ax.fill_between(ensemble_sizes, lower, upper, color=line.get_color(), alpha=0.3)
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("Ensemble Size")
    ax.set_ylabel("RMSE")
    #ax.set_title(dataset_name)
    ax.grid(True, linestyle="--", alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=["diabetes"], choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout per model (seconds)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=0,
        help="Unused placeholder; the diabetes dataset has a fixed sample size.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Unused placeholder; the diabetes dataset has no additive noise.",
    )
    parser.add_argument("--num-trials", type=int, default=50, help="Number of independent trials to average over")
    parser.add_argument(
        "--ensemble-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Explicit ensemble sizes to evaluate (default: 50 100 200)",
    )
    parser.add_argument("--ensemble-start", type=int, default=1)
    parser.add_argument("--ensemble-stop", type=int, default=501)
    parser.add_argument("--ensemble-step", type=int, default=10)
    parser.add_argument("--plot", type=str, default="plots/overfit_mse_diabetes.png")
    parser.add_argument("--show", action="store_true")
    cpu_total = os.cpu_count() or 1
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_total - 1),
        help="Number of parallel processes for evaluating num_trials (1 disables multiprocessing)",
    )
    args = parser.parse_args()

    default_ensemble_sizes = np.arange(50,500,50)
    if args.ensemble_sizes:
        ensemble_sizes = sorted(set(args.ensemble_sizes))
    elif any(value is not None for value in (args.ensemble_start, args.ensemble_stop, args.ensemble_step)):
        if args.ensemble_start is None or args.ensemble_stop is None or args.ensemble_step is None:
            raise ValueError("When using ensemble-start/stop/step, all three must be provided")
        ensemble_sizes = list(range(args.ensemble_start, args.ensemble_stop + 1, args.ensemble_step))
        if ensemble_sizes and args.ensemble_start != ensemble_sizes[0]:
            ensemble_sizes.insert(0, args.ensemble_start)
    else:
        ensemble_sizes = default_ensemble_sizes

    dataset_infos: List[SimpleNamespace] = []
    for dataset_key in args.datasets:
        loader = DATASET_LOADERS[dataset_key]
        base_dataset = loader(args.n_samples, args.noise_std, args.seed)
        print(f"\n=== Dataset: {base_dataset.name} ===")
        column_groups = _split_columns(base_dataset.X)
        base_models = ['InferableEBM', "EBM", "RandomForest", 'GradientBoosting']
        if lgb is not None:
            base_models.append("LightGBM")
        if xgb is not None:
            base_models.append("XGBoost")

        trial_splits: List[Tuple[int, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]] = []
        for trial_idx in range(args.num_trials):
            trial_seed = args.seed + trial_idx
            trial_dataset = loader(args.n_samples, args.noise_std, trial_seed)
            X_train, X_test, y_train, y_test = train_test_split(
                trial_dataset.X,
                trial_dataset.y,
                test_size=args.test_size,
                random_state=trial_seed,
            )
            trial_splits.append((trial_seed, X_train, X_test, y_train, y_test))

        dataset_infos.append(
            SimpleNamespace(
                name=base_dataset.name,
                column_groups=column_groups,
                base_models=base_models,
                trial_splits=trial_splits,
            )
        )

    fig, axes = plt.subplots(1, len(dataset_infos), figsize=(6 * len(dataset_infos), 5), sharey=False)
    if len(dataset_infos) == 1:
        axes = [axes]

    tuned_params_by_dataset = {info.name: {} for info in dataset_infos}
    args_dict = vars(args).copy()

    for info in dataset_infos:
        tuning_seed, tuning_X_train, _, tuning_y_train, _ = info.trial_splits[0]
        for model_name in info.base_models:
            print(f"Tuning {model_name} on {info.name} ...")
            try:
                params = tune_parameters(model_name, tuning_X_train, tuning_y_train, info.column_groups, args)
            except Exception as exc:
                print(f"  Skipping {model_name} on {info.name}: {exc}")
            else:
                tuned_params_by_dataset[info.name][model_name] = params

    curves_by_dataset: Dict[str, Dict[str, Dict[str, List[float]]]] = {info.name: {} for info in dataset_infos}
    args_dict_eval = vars(args).copy()
    if args.workers <= 0:
        trial_workers = min(args.num_trials, os.cpu_count() or 1)
    else:
        trial_workers = min(args.workers, args.num_trials)

    for info in dataset_infos:
        tuned_params = tuned_params_by_dataset.get(info.name, {})
        for model_name, params in tuned_params.items():
            try:
                _, result_model, mses = _evaluate_model_curve(
                    info.name,
                    model_name,
                    params,
                    info.column_groups,
                    info.trial_splits,
                    ensemble_sizes,
                    args_dict_eval,
                    max_workers=trial_workers,
                )
            except Exception as exc:
                print(f"  Evaluation failed for {model_name} on {info.name}: {exc}")
                continue
            print(f"  Evaluation succeeded for {model_name} on {info.name}")
            mses_array = np.asarray(mses)
            trial_count = max(1, mses_array.shape[0])
            curves_by_dataset[info.name][result_model] = {
                "mean": mses_array.mean(axis=0).tolist(),
                "stderr": (mses_array.std(axis=0, ddof=0) / np.sqrt(trial_count)).tolist(),
            }

    summary = {}
    for ax, info in zip(axes, dataset_infos):
        curves = curves_by_dataset.get(info.name, {})
        summary[info.name] = curves
        if curves:
            plot_results(info.name, ensemble_sizes, curves, ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'upper right')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.title('Diabetes')

    if args.plot:
        out_dir = os.path.dirname(args.plot)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(args.plot, dpi=300)
    if args.show:
        plt.show()
    plt.close(fig)

    # Save summary MSE curves for further analysis
    summary_path = Path(args.plot).with_suffix(".json") if args.plot else _ROOT / "optuna_mse_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
