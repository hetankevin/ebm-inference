#!/usr/bin/env python3
"""Benchmark multiple regressors with Optuna on UCI datasets and plot MSE curves."""
import argparse
import json
import os
import sys
import zipfile
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

CACHE_DIR = _ROOT / "data" / "uci"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# URLs for datasets
WINE_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
]
OBESITY_URL = "https://www.kaggle.com/api/v1/datasets/download/manvendrarajsingh/obesitydataset-raw-and-data-sinthetic"
AIR_QUALITY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"

# Fixed plotting order for models
# Only models present in results will be plotted, in this order.
PLOT_MODEL_ORDER = [
    "InferableEBM",
    "EBM",
    "LightGBM",
    "XGBoost",
    "RandomForest",
    "ElasticNet",
]

@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series


def _sanitize_target(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    mask = y.replace([np.inf, -np.inf], np.nan).notna()
    if not mask.all():
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
    return X, y


def download_file(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def load_wine_quality() -> Dataset:
    frames = []
    for url in WINE_URLS:
        local = download_file(url, CACHE_DIR / Path(url).name)
        df = pd.read_csv(local, sep=";")
        frames.append(df)
    data = pd.concat(frames, axis=0).reset_index(drop=True)
    y = data.pop("quality")
    data, y = _sanitize_target(data, y)
    return Dataset("Wine Quality", data, y)


def load_obesity() -> Dataset:
    local = download_file(OBESITY_URL, CACHE_DIR / "ObesityData.csv")
    df = pd.read_csv(local, encoding="latin-1")
    # Use weight (continuous) as target
    y = df.pop("Weight")
    # Drop columns with too many missing values, keep features
    df, y = _sanitize_target(df, y)
    return Dataset("Obesity", df, y)


def load_air_quality() -> Dataset:
    zip_path = download_file(AIR_QUALITY_URL, CACHE_DIR / "AirQualityUCI.zip")
    csv_path = CACHE_DIR / "AirQualityUCI.csv"
    if not csv_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("AirQualityUCI.csv") as zfile:
                content = zfile.read()
                csv_path.write_bytes(content)
    df = pd.read_csv(csv_path, sep=";", decimal=",", na_values=[-200], encoding="latin-1", engine="python")
    df = df.drop(columns=[col for col in df.columns if col.strip() == ""])  # drop empty column
    y = df.pop("CO(GT)")
    df = df.drop(columns=["Date", "Time"], errors="ignore")
    df, y = _sanitize_target(df, y)
    return Dataset("Air Quality", df, y)


DATASET_LOADERS = {
    "wine": load_wine_quality,
    "obesity": load_obesity,
    "air": load_air_quality,
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
                "n_estimators": 200,
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "random_state": args.seed,
                "n_jobs": 1,
            }
            model = RandomForestRegressor(**params)
        elif model_name == "GradientBoosting":
            params = {
                "n_estimators": 200,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "random_state": args.seed,
                "n_jobs": 1,
            }
            model = GradientBoostingRegressor(**params)
        elif model_name == "ElasticNet":
            params = {
                "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "max_iter": 10000,
                "random_state": args.seed,
            }
            model = ElasticNet(**params)
        elif model_name == "EBM":
            params = {
                "max_bins": trial.suggest_int("max_bins", 64, 256),
                "max_rounds": 200,
                'interactions' : 0.0,
                "outer_bags": 1,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.02),
                "random_state": args.seed,
                'n_jobs' : 4
            }
            model = ExplainableBoostingRegressor(**params)
        elif model_name == "InferableEBM":
            params = {
                "max_bins_auto": trial.suggest_int("max_bins_auto", 64, 512),
                "max_rounds": 200,
                'warmup_rounds' : trial.suggest_int("warmup_rounds", 0, 200),
                "subsample_rate": trial.suggest_float("subsample_rate", 0.8, 1.0),
                "truncation": trial.suggest_float("truncation", 100.0, 10000.0),
                "random_state": args.seed,
                'n_jobs' : 1,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 1., log=True),
                'auto_bins_scheme': trial.suggest_categorical('auto_bins_scheme', ['quantile', 'cube']),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 1000.0, log=True),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 64),
                'max_leaves' : trial.suggest_categorical('max_leaves', [2**i for i in range(1,8)])
            }
            model = InferableEBMRegressor(**params)
        elif model_name == "LightGBM":
            if lgb is None:
                raise optuna.exceptions.TrialPruned()
            params = {
                "n_estimators": 200,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "random_state": args.seed,
                "n_jobs": 1,
                'verbose' : -1
            }
            model = lgb.LGBMRegressor(**params)
        elif model_name == "XGBoost":
            if xgb is None:
                raise optuna.exceptions.TrialPruned()
            params = {
                "n_estimators": 200,
                "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
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
        params.update({"n_estimators": ensemble_size, "random_state": args.seed, 'n_jobs' : 1})
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


def _evaluate_model_curve(
    dataset_name: str,
    model_name: str,
    params: Dict,
    column_groups: Tuple[List[str], List[str]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ensemble_sizes: List[int],
    args_dict: Dict,
) -> Tuple[str, str, List[float]]:
    args_ns = SimpleNamespace(**args_dict)
    mses: List[float] = []
    last_mse: Optional[float] = None
    for size in tqdm(ensemble_sizes):
        if model_name == "ElasticNet" and last_mse is not None:
            mses.append(last_mse)
            continue
        model = instantiate(model_name, params, size, args_ns)
        if model_name in ("EBM", "InferableEBM"):
            pipeline = Pipeline([("model", model)])
        else:
            preprocessor = _make_preprocessor(*column_groups)
            pipeline = Pipeline([("pre", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        last_mse = root_mean_squared_error(y_test, preds)
        mses.append(last_mse)
    return dataset_name, model_name, mses


def _tune_model_task(
    dataset_name: str,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    column_groups: Tuple[List[str], List[str]],
    args_dict: Dict,
):
    args_ns = SimpleNamespace(**args_dict)
    params = tune_parameters(model_name, X, y, column_groups, args_ns)
    return dataset_name, model_name, params


def plot_results(dataset_name: str, ensemble_sizes: List[int], mse_curves: Dict[str, List[float]], ax):
    # Plot in a consistent, predefined order regardless of dict insertion order
    for name in PLOT_MODEL_ORDER:
        if name not in mse_curves:
            continue
        mses = mse_curves[name]
        if name == "ElasticNet":
            ax.hlines(mses[0], ensemble_sizes[0], ensemble_sizes[-1], color='black', linestyles="dashed", label=name)
        else:
            ax.plot(ensemble_sizes, mses, label=name, linewidth=2)
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("Ensemble Size")
    ax.set_ylabel("RMSE")
    ax.set_title(dataset_name)
    ax.grid(True, linestyle="--", alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    #"wine", "obesity", "air"
    parser.add_argument("--datasets", nargs="*", default=["wine", "obesity", "air"], choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout per model (seconds)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--ensemble-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Explicit ensemble sizes to evaluate (default: 50 100 200)",
    )
    parser.add_argument("--ensemble-start", type=int, default=20)
    parser.add_argument("--ensemble-stop", type=int, default=200)
    parser.add_argument("--ensemble-step", type=int, default=20)
    parser.add_argument("--plot", type=str, default="plots/optuna_mse.png")
    parser.add_argument("--show", action="store_true")
    cpu_total = os.cpu_count() or 1
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_total - 1),
        help="Number of parallel processes for model tuning (1 disables multiprocessing)",
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

    datasets = [DATASET_LOADERS[name]() for name in args.datasets]
    dataset_infos: List[SimpleNamespace] = []
    for dataset in datasets:
        print(f"\n=== Dataset: {dataset.name} ===")
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=args.test_size,
            random_state=args.seed,
        )
        column_groups = _split_columns(dataset.X)
        base_models = ["InferableEBM", "EBM", "RandomForest", "ElasticNet"]
        if lgb is not None:
            base_models.append("LightGBM")
        if xgb is not None:
            base_models.append("XGBoost")
        dataset_infos.append(
            SimpleNamespace(
                name=dataset.name,
                column_groups=column_groups,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                base_models=base_models,
            )
        )

    fig, axes = plt.subplots(1, len(dataset_infos), figsize=(6 * len(dataset_infos), 5), sharey=False)
    if len(dataset_infos) == 1:
        axes = [axes]

    tuned_params_by_dataset = {info.name: {} for info in dataset_infos}
    args_dict = vars(args).copy()

    if args.workers != 1:
        max_workers = args.workers if args.workers > 0 else (os.cpu_count()-1 or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for info in dataset_infos:
                for model_name in info.base_models:
                    print(f"Tuning {model_name} on {info.name} ...")
                    future = executor.submit(
                        _tune_model_task,
                        info.name,
                        model_name,
                        info.X_train,
                        info.y_train,
                        info.column_groups,
                        args_dict,
                    )
                    futures[future] = (info.name, model_name)
            for future in as_completed(futures):
                dataset_name, model_name = futures[future]
                try:
                    result_dataset, result_model, params = future.result()
                    print(model_name+' finished on '+dataset_name)
                except Exception as exc:
                    print(f"  Skipping {model_name} on {dataset_name}: {exc}")
                else:
                    tuned_params_by_dataset[result_dataset][result_model] = params
    else:
        for info in dataset_infos:
            for model_name in info.base_models:
                print(f"Tuning {model_name} on {info.name} ...")
                try:
                    params = tune_parameters(model_name, info.X_train, info.y_train, info.column_groups, args)
                except Exception as exc:
                    print(f"  Skipping {model_name} on {info.name}: {exc}")
                else:
                    tuned_params_by_dataset[info.name][model_name] = params

    curves_by_dataset: Dict[str, Dict[str, List[float]]] = {info.name: {} for info in dataset_infos}
    args_dict_eval = vars(args).copy()

    if args.workers != 1:
        max_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for info in dataset_infos:
                tuned_params = tuned_params_by_dataset.get(info.name, {})
                for model_name, params in tuned_params.items():
                    future = executor.submit(
                        _evaluate_model_curve,
                        info.name,
                        model_name,
                        params,
                        info.column_groups,
                        info.X_train,
                        info.y_train,
                        info.X_test,
                        info.y_test,
                        ensemble_sizes,
                        args_dict_eval,
                    )
                    futures[future] = (info.name, model_name)
            for future in as_completed(futures):
                dataset_name, model_name = futures[future]
                try:
                    result_dataset, result_model, mses = future.result()
                    print(f"  Evaluation succeeded for {model_name} on {dataset_name}")
                except Exception as exc:
                    print(f"  Evaluation failed for {model_name} on {dataset_name}: {exc}")
                else:
                    curves_by_dataset[result_dataset][result_model] = mses
    else:
        for info in dataset_infos:
            tuned_params = tuned_params_by_dataset.get(info.name, {})
            for model_name, params in tuned_params.items():
                _, result_model, mses = _evaluate_model_curve(
                    info.name,
                    model_name,
                    params,
                    info.column_groups,
                    info.X_train,
                    info.y_train,
                    info.X_test,
                    info.y_test,
                    ensemble_sizes,
                    args_dict_eval,
                )
                curves_by_dataset[info.name][result_model] = mses

    summary = {}
    for ax, info in zip(axes, dataset_infos):
        curves = curves_by_dataset.get(info.name, {})
        summary[info.name] = curves
        if curves:
            plot_results(info.name, ensemble_sizes, curves, ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.tight_layout(rect=[0, 0, 1, 0.93])

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
