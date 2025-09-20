
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


def make_suite(rng, n=4000, p=5, noise=1.0):
    # three datasets
    X1 = rng.uniform(0, 1, size=(n, 10))
    f1 = (10*np.sin(np.pi*X1[:,0]*X1[:,1]) + 20*(X1[:,2]-0.5)**2 + 10*X1[:,3] + 5*X1[:,4])
    y1 = f1 + rng.normal(0, noise, size=n)

    X2 = rng.normal(size=(n, p))
    y2 = 2.0*X2[:,0] - 3.0*(X2[:,1]**2) + 0.5*X2[:,2] + rng.normal(scale=noise, size=n)

    X3 = rng.normal(size=(n, p))
    y3 = 3.0*np.sin(X3[:,0]) + 2.0*np.sign(X3[:,1]) + rng.normal(scale=noise, size=n)
    return [(X1,y1,"friedman"), (X2,y2,"additive"), (X3,y3,"nonlinear")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=str, default="matchups.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for rep in range(args.reps):
        for X,y,name in make_suite(rng, n=4000, p=5, noise=1.0):
            n = X.shape[0]
            tr = np.arange(int(0.7*n)); te = np.arange(int(0.7*n), n)
            m = InferableEBMRegressor(max_rounds=args.rounds, random_state=rep).fit(X[tr], y[tr])
            mse = float(np.mean((y[te] - m.predict(X[te]))**2))
            rows.append({"rep":rep, "dataset":name, "model":"EBM", "mse":mse})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
