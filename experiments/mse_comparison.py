
#!/usr/bin/env python3
import argparse, warnings
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


# Optional baselines
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
except Exception:
    RandomForestRegressor = None
    GradientBoostingRegressor = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None

def make_additive(n, p, rng, noise=1.0):
    X = rng.normal(size=(n, p))
    y = 2.0*X[:,0] - 3.0*(X[:,1]**2) + 0.5*X[:,2] + rng.normal(scale=noise, size=n)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--out", type=str, default="mse_curves.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    X, y = make_additive(args.n, args.p, rng, args.noise)
    n = X.shape[0]
    tr = np.arange(int(0.7*n))
    te = np.arange(int(0.7*n), n)

    # EBM curve vs rounds
    rows = []
    for r in np.linspace(20, args.rounds, 20, dtype=int):
        m = InferableEBMRegressor(max_rounds=int(r), random_state=0).fit(X[tr], y[tr])
        mse = float(np.mean((y[te] - m.predict(X[te]))**2))
        rows.append({"model":"EBM", "rounds":int(r), "mse":mse})

    # RandomForest depth curve (optional)
    if RandomForestRegressor is not None:
        for depth in [2,3,4,5,6,8,10]:
            rf = RandomForestRegressor(n_estimators=300, max_depth=depth, random_state=0, n_jobs=-1)
            rf.fit(X[tr], y[tr])
            mse = float(np.mean((y[te] - rf.predict(X[te]))**2))
            rows.append({"model":f"RF_d{depth}", "rounds":depth, "mse":mse})

    # GradientBoosting stages (optional)
    if GradientBoostingRegressor is not None:
        gbr = GradientBoostingRegressor(n_estimators=args.rounds, learning_rate=0.05, max_depth=3, random_state=0)
        gbr.fit(X[tr], y[tr])
        y_pred = list(gbr.staged_predict(X[te]))
        for t, pred in enumerate(y_pred, start=1):
            if t % max(1, args.rounds//20) == 0:
                mse = float(np.mean((y[te] - pred)**2))
                rows.append({"model":"GBR", "rounds":t, "mse":mse})

    # XGBoost (optional)
    if xgb is not None:
        dtr = xgb.DMatrix(X[tr], label=y[tr])
        dte = xgb.DMatrix(X[te], label=y[te])
        params = dict(objective="reg:squarederror", max_depth=3, eta=0.05, subsample=0.8, colsample_bytree=0.8, seed=0)
        bst = xgb.train(params, dtr, num_boost_round=args.rounds, evals=[(dte, "valid")], verbose_eval=False)
        for t in np.linspace(20, args.rounds, 20, dtype=int):
            pred_t = bst.predict(dte, iteration_range=(0, int(t)))
            mse = float(np.mean((y[te] - pred_t)**2))
            rows.append({"model":"XGB", "rounds":int(t), "mse":mse})

    # LightGBM (optional)
    if lgb is not None:
        train_set = lgb.Dataset(X[tr], label=y[tr])
        valid_set = lgb.Dataset(X[te], label=y[te], reference=train_set)
        params = dict(objective="regression", metric="l2", learning_rate=0.05, max_depth=3, num_leaves=15, subsample=0.8,
                      feature_fraction=0.8, seed=0, verbose=-1)
        bst = lgb.train(params, train_set, num_boost_round=args.rounds, valid_sets=[valid_set], valid_names=["valid"],
                        callbacks=[lgb.log_evaluation(0)])
        for t in np.linspace(20, args.rounds, 20, dtype=int):
            pred_t = bst.predict(X[te], num_iteration=int(t))
            mse = float(np.mean((y[te] - pred_t)**2))
            rows.append({"model":"LGB", "rounds":int(t), "mse":mse})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby("model")["mse"].min().sort_values())

if __name__ == "__main__":
    main()
