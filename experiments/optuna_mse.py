#!/usr/bin/env python3
"""Benchmark multiple regressors with Optuna on UCI datasets and plot MSE curves."""
import argparse
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna import Trial
from optuna.samplers import TPESampler

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
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
    raise SystemExit("EBMRegressor import failed") from exc

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

ALGORITHM_COLORS = {
    "InferableEBM": "#00BEFF",
    "EBM": "#1f77b4",
    "RandomForest": "#FF7F0E",
    "GradientBoosting": "#9467BD",
    "LightGBM": "#7CAE00",
    "XGBoost": "#2CA02C",
    "ElasticNet": "#17BECF",
}

@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series


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
    return Dataset("Wine Quality", data, y)


def load_obesity() -> Dataset:
    local = download_file(OBESITY_URL, CACHE_DIR / "ObesityData.csv")
    df = pd.read_csv(local, encoding="latin-1")
    # Use weight (continuous) as target
    y = df.pop("Weight")
    # Drop columns with too many missing values, keep features
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
    return Dataset("Air Quality", df, y)


DATASET_LOADERS = {
    "wine": load_wine_quality,
    "obesity": load_obesity,
    "air": load_air_quality,
}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
    return ColumnTransformer(transformers)


def cross_val_score_neg_mse(model, X, y, preprocessor, n_splits=3, random_state=0):
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", model),
    ])
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, scoring="neg_mean_squared_error", cv=cv)
    return scores.mean()


def tune_parameters(model_name: str, X: pd.DataFrame, y: pd.Series, preprocessor, args) -> Dict:
    def objective(trial: Trial):
        if model_name == "RandomForest":
            params = {
                "n_estimators": 200,
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", 0.5, 0.8]),
                "random_state": args.seed,
                "n_jobs": -1,
            }
            model = RandomForestRegressor(**params)
        elif model_name == "GradientBoosting":
            params = {
                "n_estimators": 200,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": args.seed,
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
                "max_rounds": trial.suggest_int("max_rounds", 100, 400),
                "outer_bags": trial.suggest_int("outer_bags", 4, 16),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "random_state": args.seed,
            }
            model = EBMRegressor(**params)
        elif model_name == "InferableEBM":
            params = {
                "max_bins": trial.suggest_int("max_bins", 64, 256),
                "max_rounds": trial.suggest_int("max_rounds", 100, 400),
                "subsample_rate": trial.suggest_float("subsample_rate", 0.6, 1.0),
                "truncation": trial.suggest_float("truncation", 2.0, 6.0),
                "random_state": args.seed,
                "bin_level_inference": args.bin_level_inference,
            }
            model = InferableEBMRegressor(**params)
        elif model_name == "LightGBM":
            if lgb is None:
                raise optuna.exceptions.TrialPruned()
            params = {
                "n_estimators": 200,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": args.seed,
            }
            model = lgb.LGBMRegressor(**params)
        elif model_name == "XGBoost":
            if xgb is None:
                raise optuna.exceptions.TrialPruned()
            params = {
                "n_estimators": 200,
                "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
                "random_state": args.seed,
                "tree_method": "hist",
            }
            model = xgb.XGBRegressor(**params)
        else:
            raise ValueError(model_name)

        score = cross_val_score_neg_mse(model, X, y, preprocessor, n_splits=3, random_state=args.seed)
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
        params.update({"n_estimators": ensemble_size, "random_state": args.seed, "n_jobs": -1})
        return RandomForestRegressor(**params)
    if model_name == "GradientBoosting":
        params.update({"n_estimators": ensemble_size, "random_state": args.seed})
        return GradientBoostingRegressor(**params)
    if model_name == "ElasticNet":
        params.update({"max_iter": 10000, "random_state": args.seed})
        return ElasticNet(**params)
    if model_name == "EBM":
        params.update({"max_rounds": ensemble_size, "random_state": args.seed})
        return ExplainableBoostingRegressor(**params)
    if model_name == "InferableEBM":
        params.update({"max_rounds": ensemble_size, "random_state": args.seed, "bin_level_inference": args.bin_level_inference})
        return InferableEBMRegressor(**params)
    if model_name == "LightGBM":
        if lgb is None:
            raise RuntimeError("LightGBM not installed")
        params.update({"n_estimators": ensemble_size, "random_state": args.seed})
        return lgb.LGBMRegressor(**params)
    if model_name == "XGBoost":
        if xgb is None:
            raise RuntimeError("xgboost not installed")
        params.update({"n_estimators": ensemble_size, "random_state": args.seed, "tree_method": "hist"})
        return xgb.XGBRegressor(**params)
    raise ValueError(model_name)


def evaluate_curves(models: Dict[str, Dict], preprocessor, X_train, y_train, X_test, y_test, ensemble_sizes: List[int], args) -> Dict[str, List[float]]:
    results = {}
    for name, params in models.items():
        mses = []
        for size in ensemble_sizes:
            if name == "ElasticNet":
                if size != ensemble_sizes[0]:
                    mses.append(mses[-1])
                    continue
            model = instantiate(name, params, size, args)
            pipeline = Pipeline([("pre", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            mses.append(mean_squared_error(y_test, preds))
        results[name] = mses
    return results


def plot_results(dataset_name: str, ensemble_sizes: List[int], mse_curves: Dict[str, List[float]], ax):
    for name, mses in mse_curves.items():
        color = ALGORITHM_COLORS.get(name, None)
        if name == "ElasticNet":
            ax.hlines(mses[0], ensemble_sizes[0], ensemble_sizes[-1], colors=color, linestyles="dashed", label=name)
        else:
            ax.plot(ensemble_sizes, mses, label=name, color=color, linewidth=2)
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("Ensemble Size")
    ax.set_ylabel("MSE")
    ax.set_title(dataset_name)
    ax.grid(True, linestyle="--", alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=["wine", "obesity", "air"], choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout per model (seconds)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ensemble-start", type=int, default=10)
    parser.add_argument("--ensemble-stop", type=int, default=400)
    parser.add_argument("--ensemble-step", type=int, default=20)
    parser.add_argument("--plot", type=str, default="plots/optuna_mse.png")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--bin-level-inference", action="store_true")
    args = parser.parse_args()

    ensemble_sizes = list(range(args.ensemble_start, args.ensemble_stop + 1, args.ensemble_step))
    if args.ensemble_start != ensemble_sizes[0]:
        ensemble_sizes.insert(0, args.ensemble_start)

    datasets = [DATASET_LOADERS[name]() for name in args.datasets]

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    summary = {}
    for ax, dataset in zip(axes, datasets):
        print(f"\n=== Dataset: {dataset.name} ===")
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=args.test_size,
            random_state=args.seed,
        )
        preprocessor = build_preprocessor(dataset.X)
        base_models = ["InferableEBM", "EBM", "RandomForest", "GradientBoosting", "ElasticNet"]
        if lgb is not None:
            base_models.append("LightGBM")
        if xgb is not None:
            base_models.append("XGBoost")

        tuned_params = {}
        for name in base_models:
            print(f"Tuning {name} ...")
            try:
                params = tune_parameters(name, X_train, y_train, preprocessor, args)
                tuned_params[name] = params
            except Exception as exc:
                print(f"  Skipping {name}: {exc}")
        curves = evaluate_curves(tuned_params, preprocessor, X_train, y_train, X_test, y_test, ensemble_sizes, args)
        summary[dataset.name] = curves
        plot_results(dataset.name, ensemble_sizes, curves, ax)

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
