
#!/usr/bin/env python3
import argparse, warnings
import numpy as np
import pandas as pd

import numpy as np

# Import helper: prefer local estimator file, else from interpret if exported there
try:
    from inferable_ebm_regressor import InferableEBMRegressor
except Exception:
    try:
        from interpret.glassbox import InferableEBMRegressor
    except Exception:
        raise ImportError("Place inferable_ebm_regressor.py next to this script or export it in your package.")


# Optional baselines
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
except Exception:
    RandomForestRegressor = None
    GradientBoostingRegressor = None

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

    rows = []
    for r in np.linspace(20, args.rounds, 20, dtype=int):
        m = InferableEBMRegressor(max_rounds=int(r), random_state=0).fit(X[tr], y[tr])
        mse = float(np.mean((y[te] - m.predict(X[te]))**2))
        rows.append({"model":"EBM", "rounds":int(r), "mse":mse})

    if RandomForestRegressor is not None:
        for depth in [2,3,4,5,6,8,10]:
            rf = RandomForestRegressor(n_estimators=300, max_depth=depth, random_state=0, n_jobs=-1)
            rf.fit(X[tr], y[tr])
            mse = float(np.mean((y[te] - rf.predict(X[te]))**2))
            rows.append({"model":f"RF_d{depth}", "rounds":depth, "mse":mse})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby("model")["mse"].min().sort_values())

if __name__ == "__main__":
    main()
