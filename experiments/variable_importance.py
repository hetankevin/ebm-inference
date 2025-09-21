
#!/usr/bin/env python3
import argparse
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


def make_signal(n, p, rng, noise=1.0):
    X = rng.normal(size=(n, p))
    y = 2.5*X[:,1] + rng.normal(scale=noise, size=n)  # only feature 1 has signal
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--use-nystrom", action="store_true")
    ap.add_argument("--nystrom-rank", type=int, default=256)
    ap.add_argument("--nystrom-ridge", type=float, default=1e-6)
    ap.add_argument("--out", type=str, default="vi_results.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    X, y = make_signal(args.n, args.p, rng, args.noise)
    n = X.shape[0]
    tr = np.arange(int(0.7*n))
    te = np.arange(int(0.7*n), n)

    ebm = InferableEBMRegressor(max_rounds=args.rounds, random_state=0).fit(X[tr], y[tr])
    # If your class exposes variable_importance_test; otherwise compute a simple permutation VI as placeholder.
    try:
        res_true = ebm.variable_importance_test(X[te], y[te], groups=[1])
        res_none = ebm.variable_importance_test(X[te], y[te], groups=[])
        df = pd.DataFrame([{"case":"keep_signal", **res_true},
                           {"case":"remove_signal", **res_none}])
    except Exception:
        # Fallback: permutation score delta MSE
        base = float(np.mean((y[te] - ebm.predict(X[te]))**2))
        Xp = X[te].copy()
        rng = np.random.default_rng(0)
        rng.shuffle(Xp[:,1])
        perm = float(np.mean((y[te] - ebm.predict(Xp))**2))
        df = pd.DataFrame([{"case":"perm_vi_feature1", "delta_mse": perm - base}])
    df.to_csv(args.out, index=False)
    print(df)

if __name__ == "__main__":
    main()
