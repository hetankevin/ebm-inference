
#!/usr/bin/env python3
import argparse, os
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


def make_friedman(n, rng):
    X = rng.uniform(0.0, 1.0, size=(n, 10))
    f = (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] + 5*X[:,4])
    return X, f

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

def run_rep(rep, args, base_seed=0):
    rng = np.random.default_rng(base_seed + 7919*rep)
    X, f_true = make_friedman(args.n, rng)
    y = f_true + rng.normal(0, args.noise, size=args.n)

    (Xtr, ytr), (Xcal, ycal), (Xte, yte) = split3(X, y, rng, cal_frac=args.cal_frac, test_frac=args.test_frac)
    f_te = f_true[:Xte.shape[0]]

    ebm = InferableEBMRegressor(
        max_rounds=args.rounds,
        max_bins=args.n_bins,
        subsample_rate=args.subsample_rate,
        truncation=args.truncation,
        random_state=base_seed + 1009*rep,
    ).fit(Xtr, ytr)

    # Calibrated sigma with guards
    resid_cal = ycal - ebm.predict(Xcal)
    resid_cal = resid_cal[np.isfinite(resid_cal)]
    sigma = float(np.std(resid_cal, ddof=1)) if resid_cal.size else 1e-8
    if not np.isfinite(sigma) or sigma <= 0:
        resid = ytr - ebm.predict(Xtr)
        resid = resid[np.isfinite(resid)]
        sigma = float(np.std(resid, ddof=1)) if resid.size else 1e-8
    sigma = float(np.clip(sigma, 1e-8, 1e6))

    ci_l, ci_u, yhat = ebm.predict_intervals(Xte, level=args.level, mode="confidence", sigma=sigma)
    pi_l, pi_u, _    = ebm.predict_intervals(Xte, level=args.level, mode="prediction", sigma=sigma)
    ri_l, ri_u, _    = ebm.predict_intervals(Xte, level=args.level, mode="reproduction", sigma=sigma)

    # Paper-faithful coverage definitions
    cov_ci = float(np.mean((f_te >= ci_l) & (f_te <= ci_u)))
    cov_pi = float(np.mean((yte  >= pi_l) & (yte  <= pi_u)))
    yhat_te = ebm.predict(Xte)
    cov_ri = float(np.mean((yhat_te >= ri_l) & (yhat_te <= ri_u)))

    w_ci = float(np.mean(ci_u - ci_l))
    w_pi = float(np.mean(pi_u - pi_l))
    w_ri = float(np.mean(ri_u - ri_l))
    return dict(rep=rep, cov_ci=cov_ci, cov_pi=cov_pi, cov_ri=cov_ri,
                w_ci=w_ci, w_pi=w_pi, w_ri=w_ri, sigma=sigma)

def main():
    np.seterr(over='ignore', invalid='ignore')  # avoid noisy warnings; we guard widths explicitly
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--cal-frac", dest="cal_frac", type=float, default=0.2)
    ap.add_argument("--test-frac", dest="test_frac", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--n-bins", dest="n_bins", type=int, default=0)
    ap.add_argument("--subsample-rate", dest="subsample_rate", type=float, default=1.0)
    ap.add_argument("--truncation", type=float, default=3.0)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--use-nystrom", dest="use_nystrom", action="store_true")
    ap.add_argument("--nystrom-rank", dest="nystrom_rank", type=int, default=256)
    ap.add_argument("--nystrom-ridge", dest="nystrom_ridge", type=float, default=1e-6)
    ap.add_argument("--out", type=str, default="coverage_summary.csv")
    args = ap.parse_args()

    rows = [run_rep(r, args) for r in range(args.reps)]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.describe())

if __name__ == "__main__":
    main()
