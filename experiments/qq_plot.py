
#!/usr/bin/env python3
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

# Common import helper: prefer local estimator file, else import from interpret if available
try:
    from inferable_ebm_regressor import InferableEBMRegressor
except Exception as _e:
    try:
        from interpret.glassbox import InferableEBMRegressor  # requires your package export
    except Exception as _e2:
        raise ImportError("Could not import InferableEBMRegressor from local file or interpret.glassbox. "
                          "Place inferable_ebm_regressor.py next to this script or export it in your package.")


def make_additive(n, p, rng, noise=1.0):
    X = rng.normal(size=(n, p))
    f = 2.0*X[:,0] - 3.0*(X[:,1]**2) + 0.5*X[:,2]
    y = f + rng.normal(scale=noise, size=n)
    return X, y, f

def split3(X, y, rng, cal_frac=0.2, test_frac=0.3):
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n*test_frac)
    n_cal = int((n-n_test)*cal_frac)
    te = idx[:n_test]
    cal = idx[n_test:n_test+n_cal]
    tr = idx[n_test+n_cal:]
    return (X[tr], y[tr]), (X[cal], y[cal]), (X[te], y[te])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--out", type=str, default="qq_plot_ci.png")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    X, y, f = make_additive(args.n, args.p, rng, args.noise)
    (Xtr, ytr), (Xcal, ycal), (Xte, yte) = split3(X, y, rng)

    ebm = InferableEBMRegressor(max_rounds=args.rounds, subsample_rate=1.0, truncation=3.0, random_state=0).fit(Xtr, ytr)
    sigma = float(np.std(ycal - ebm.predict(Xcal), ddof=1))

    # Standardize residuals with influence norm for CI
    z = []
    for i in range(Xte.shape[0]):
        r = ebm._r_vector(Xte[i])  # CI influence
        nr = float(np.linalg.norm(r))
        if nr == 0:
            continue
        z.append( (yte[i] - ebm.predict(Xte[i:i+1]))[0] / (sigma * nr) )
    z = np.asarray(z).ravel()
    z.sort()
    # QQ plot against standard normal
    q = np.linspace(0.5/len(z), 1-0.5/len(z), len(z))
    theo = np.sqrt(2)*erfinv(2*q - 1)

    plt.figure(figsize=(6,6))
    plt.plot(theo, z, '.', ms=3)
    lo = min(theo.min(), z.min())
    hi = max(theo.max(), z.max())
    plt.plot([lo, hi], [lo, hi], '--')
    plt.xlabel("Theoretical N(0,1) quantiles")
    plt.ylabel("Standardized residual quantiles")
    plt.title("QQ plot (CI)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
