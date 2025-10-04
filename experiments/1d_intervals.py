
#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

# Import helper: prefer local estimator file, else from interpret if exported there
try:
    from inferable_ebm_regressor import InferableEBMRegressor
except Exception:
    try:
        from interpret.glassbox import InferableEBMRegressor
    except Exception:
        raise ImportError("Place inferable_ebm_regressor.py next to this script or export it in your package.")


def true_fun(x):
    return 2.0*np.sin(2*np.pi*x) - 1.5*(x-0.5)**2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--out", type=str, default="intervals_1d.png")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(args.n, 1))
    y = true_fun(X[:,0]) + rng.normal(0, args.noise, size=args.n)
    n = args.n
    tr = np.arange(int(0.7*n))
    cal = np.arange(int(0.7*n), int(0.85*n))
    te = np.arange(int(0.85*n), n)

    ebm = InferableEBMRegressor(max_rounds=args.rounds, random_state=0).fit(X[tr], y[tr])

    resid_cal = y[cal] - ebm.predict(X[cal])
    resid_cal = resid_cal[np.isfinite(resid_cal)]
    sigma = float(np.std(resid_cal, ddof=1)) if resid_cal.size else 1e-8

    xs = np.linspace(0,1,400)
    Xs = xs[:,None]
    ci_l, ci_u, yhat = ebm.predict_intervals(Xs, level=args.level, mode="confidence", sigma=sigma)
    pi_l, pi_u, _    = ebm.predict_intervals(Xs, level=args.level, mode="prediction", sigma=sigma)
    ri_l, ri_u, _    = ebm.predict_intervals(Xs, level=args.level, mode="reproduction", sigma=sigma)

    plt.figure(figsize=(7,4))
    plt.scatter(X[tr,0], y[tr], s=4, alpha=0.3, label="train")
    plt.plot(xs, true_fun(xs), 'k--', lw=1, label="true f(x)")
    plt.plot(xs, yhat, 'b', lw=1.5, label="EBM fit")
    plt.fill_between(xs, ci_l, ci_u, alpha=0.3, label="CI")
    plt.fill_between(xs, pi_l, pi_u, alpha=0.2, label="PI")
    plt.fill_between(xs, ri_l, ri_u, alpha=0.2, label="RI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
