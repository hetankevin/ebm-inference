#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
import inspect

# Import helper: force use of patched estimator
try:
    from inferable_ebm_regressor import InferableEBMRegressor  # your local, patched file
    import inspect; print("predict_intervals from:", inspect.getsourcefile(InferableEBMRegressor.predict_intervals))
except Exception:
    try:
        # Import directly from the interpret/_ebm.py file (patched version)
        import importlib.util
        spec = importlib.util.spec_from_file_location("_ebm", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "interpret", "python", "interpret-core", "interpret", "glassbox", "_ebm", "_ebm.py"))
        _ebm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ebm_module)
        InferableEBMRegressor = _ebm_module.InferableEBMRegressor
        import inspect; print("predict_intervals from:", inspect.getsourcefile(InferableEBMRegressor.predict_intervals))
    except Exception:
        from interpret.glassbox import InferableEBMRegressor
        import inspect; print("predict_intervals from:", inspect.getsourcefile(InferableEBMRegressor.predict_intervals))


def make_friedman(n, rng):
    X = rng.uniform(0.0, 1.0, size=(n, 10))
    f = (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] + 5*X[:,4])
    return X, f

def split3(X, y, rng, cal_frac=0.2, test_frac=0.3):
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(n * test_frac))
    n_cal  = max(1, int((n - n_test) * cal_frac))
    te = idx[:n_test]
    cal = idx[n_test:n_test+n_cal]
    tr = idx[n_test+n_cal:]
    return (X[tr], y[tr], tr), (X[cal], y[cal], cal), (X[te], y[te], te)

def run_rep(rep, args, base_seed=0):
    rng = np.random.default_rng(base_seed + 7919*rep)
    X, f_true = make_friedman(args.n, rng)
    y = f_true + rng.normal(0, args.noise, size=args.n)

    (Xtr, ytr, tr), (Xcal, ycal, cal), (Xte, yte, te) = split3(X, y, rng, cal_frac=args.cal_frac, test_frac=args.test_frac)
    f_te = f_true[te]  # Fix oracle indexing

    ebm = InferableEBMRegressor(
        max_rounds=args.rounds,
        max_bins=args.n_bins,
        subsample_rate=args.subsample_rate,
        truncation=args.truncation,
        random_state=base_seed + 1009*rep,
    ).fit(Xtr, ytr)

    # Quick verification that we're using the correct implementation
    if rep == 0:  # Only debug first repetition
        lo, hi, _ = ebm.predict_intervals(Xte[:5], level=0.95, mode="prediction", sigma=1.0)
        print(f"Rep {rep}: PI widths with σ=1: {hi - lo}")
        if np.allclose(hi - lo, 0):
            print("WARNING: All PI widths are zero - check implementation!")
        
        # EBM estimation debugging
        print("\n=== EBM Estimation Debug ===")
        mu_y = float(np.mean(ebm.train_y_))
        print("intercept_:", getattr(ebm, "intercept_", None), "  y_train_mean:", mu_y)

        yhat_tr = ebm.predict(ebm.train_X_)
        resid_tr = ebm.train_y_ - yhat_tr
        print("train resid mean:", float(np.mean(resid_tr)))
        print("pred mean:", float(np.mean(yhat_tr)), "  pred min/max:", float(np.min(yhat_tr)), float(np.max(yhat_tr)))
        print("finite train preds:", np.isfinite(yhat_tr).all())

        # Per-term mean contribution on training distribution
        print("Per-term mean contributions:")
        for j in range(ebm.train_X_.shape[1]):
            try:
                bins = ebm.train_bins_by_feat_[j]
                # get per-bin score vector for feature j, adapt name to your storage
                scores = ebm.term_scores_[j]          # <-- replace with your actual array (shape [n_bins_j])
                w = np.bincount(bins, minlength=scores.shape[0]).astype(float)
                w /= w.sum()
                term_mean = float(np.dot(w, scores))
                print(f"feature {j}: term_mean={term_mean:.6g}")
            except Exception as e:
                print(f"feature {j}: error getting term mean - {e}")
        print("=== End EBM Debug ===\n")

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