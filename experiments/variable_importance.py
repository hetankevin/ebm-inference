
#!/usr/bin/env python3
import argparse
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
    ap.add_argument("--out", type=str, default="vi_results.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    X, y = make_signal(args.n, args.p, rng, args.noise)
    n = X.shape[0]
    tr = np.arange(int(0.7*n))
    te = np.arange(int(0.7*n), n)

    ebm = InferableEBMRegressor(max_rounds=args.rounds, random_state=0).fit(X[tr], y[tr])
    res_true = ebm.variable_importance_test(X[te], y[te], groups=[1])
    res_none = ebm.variable_importance_test(X[te], y[te], groups=[])

    df = pd.DataFrame([{"case":"keep_signal", **res_true},
                       {"case":"remove_signal", **res_none}])
    df.to_csv(args.out, index=False)
    print(df)

if __name__ == "__main__":
    main()
