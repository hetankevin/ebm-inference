
#!/usr/bin/env python3
import argparse, warnings, sys
import numpy as np
import pandas as pd

# Common import helper: prefer local estimator file, else import from interpret if available
try:
    from inferable_ebm_regressor import InferableEBMRegressor
except Exception as _e:
    try:
        from interpret.glassbox import InferableEBMRegressor  # requires your package export
    except Exception as _e2:
        raise ImportError("Could not import InferableEBMRegressor from local file or interpret.glassbox. "
                          "Place inferable_ebm_regressor.py next to this script or export it in your package.")


try:
    import optuna
except Exception:
    optuna = None

def make_additive(n, p, rng, noise=1.0):
    X = rng.normal(size=(n, p))
    y = 2.0*X[:,0] - 3.0*(X[:,1]**2) + 0.5*X[:,2] + rng.normal(scale=noise, size=n)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", type=str, default="optuna_ebm.csv")
    args = ap.parse_args()

    if optuna is None:
        print("Optuna not installed; skipping.")
        sys.exit(0)

    rng = np.random.default_rng(0)
    X, y = make_additive(args.n, args.p, rng, args.noise)
    n = X.shape[0]
    tr = np.arange(int(0.7*n))
    te = np.arange(int(0.7*n), n)

    def objective(trial):
        rounds = trial.suggest_int("rounds", 50, 500)
        trunc = trial.suggest_float("truncation", 1.0, 5.0)
        subs = trial.suggest_float("subsample_rate", 0.5, 1.0)
        n_bins = trial.suggest_categorical("n_bins", [0, 32, 64, 128])
        m = InferableEBMRegressor(max_rounds=rounds, truncation=trunc, subsample_rate=subs, max_bins=n_bins, random_state=0).fit(X[tr], y[tr])
        mse = float(np.mean((y[te] - m.predict(X[te]))**2))
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)
    df = pd.DataFrame([study.best_params | {"best_value": study.best_value}])
    df.to_csv(args.out, index=False)
    print(df)

if __name__ == "__main__":
    main()
