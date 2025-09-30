#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

# Prefer local patched estimator, else installed one
try:
    from inferable_ebm_regressor import InferableEBMRegressor
except Exception:
    try:
        from interpret.glassbox import InferableEBMRegressor
    except Exception:
        raise ImportError("Place inferable_ebm_regressor.py next to this script or export it in your package.")

def make_additive(n, p, rng, noise=1.0):
    X = rng.normal(size=(n, p))
    f = 2.0*X[:,0] - 3.0*(X[:,1]**2) + 0.5*X[:,2]
    y = f + rng.normal(scale=noise, size=n)
    return X, y, f

def split3(X, y, rng, cal_frac=0.2, test_frac=0.3):
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(n * test_frac))
    n_cal  = max(1, int((n - n_test) * cal_frac))
    te  = idx[:n_test]
    cal = idx[n_test:n_test+n_cal]
    tr  = idx[n_test+n_cal:]
    return (X[tr], y[tr], tr), (X[cal], y[cal], cal), (X[te], y[te], te)

def ebm_target_mean(Xtr, ytr, Xte, *, rounds, subsample_rate, truncation,
                     replicates, use_nystrom, nystrom_rank, nystrom_ridge, seed0=1234):
    """Approximate E[f_hat(x)] by averaging K independent refits (different seeds)."""
    preds = []
    for k in range(replicates):
        m = InferableEBMRegressor(
            max_rounds=rounds,
            subsample_rate=subsample_rate,
            truncation=truncation,
            random_state=seed0 + 7919 * k,
            use_nystrom=use_nystrom,
            nystrom_rank=nystrom_rank,
            nystrom_ridge=nystrom_ridge,
        ).fit(Xtr, ytr)
        preds.append(m.predict(Xte))
    return np.mean(np.vstack(preds), axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--subsample-rate", type=float, default=0.6, help="Use <1.0 for algorithmic randomness")
    ap.add_argument("--truncation", type=float, default=3.0)
    ap.add_argument("--replicates", type=int, default=20, help="K refits to approximate the EBM target")
    ap.add_argument("--use-nystrom", action="store_true")
    ap.add_argument("--nystrom-rank", type=int, default=256)
    ap.add_argument("--nystrom-ridge", type=float, default=1e-6)
    ap.add_argument("--out", type=str, default="qq_plot_ci.png")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    X, y, f_true = make_additive(args.n, args.p, rng, args.noise)
    (Xtr, ytr, tr), (Xcal, ycal, cal), (Xte, yte, te) = split3(X, y, rng)

    # Working model (used for r(x) and sigma)
    ebm = InferableEBMRegressor(
        max_rounds=args.rounds,
        subsample_rate=args.subsample_rate,
        truncation=args.truncation,
        random_state=0,
    ).fit(Xtr, ytr)

    # Calibrate sigma on held-out calibration split (winsorize tails)
    resid_cal = ycal - ebm.predict(Xcal)
    resid_cal = resid_cal[np.isfinite(resid_cal)]
    if resid_cal.size:
        lo, hi = np.quantile(resid_cal, [0.005, 0.995])
        resid_cal = np.clip(resid_cal, lo, hi)
    sigma = float(np.std(resid_cal, ddof=1)) if resid_cal.size else 1e-8
    sigma = float(np.clip(sigma, 1e-8, 1e6))

    # CI target: EBM expectation (average of K refits)
    target_te = ebm_target_mean(
        Xtr, ytr, Xte,
        rounds=args.rounds,
        subsample_rate=args.subsample_rate,
        truncation=args.truncation,
        replicates=args.replicates,
        use_nystrom=args.use_nystrom,
        nystrom_rank=args.nystrom_rank,
        nystrom_ridge=args.nystrom_ridge,
        seed0=1234,
    )

    # Standardized z = (E[f̂]-f̂) / (σ||r||)
    preds_te = ebm.predict(Xte)
    z = []
    rnorm = []
    for i in range(Xte.shape[0]):
        r = ebm._r_vector(Xte[i])
        nr = float(np.linalg.norm(r))
        if not np.isfinite(nr) or nr < 1e-8:
            continue
        z.append((target_te[i] - preds_te[i]) / (sigma * nr))
        rnorm.append(nr)
    z = np.asarray(z).ravel(); z.sort()
    rnorm = np.asarray(rnorm)
    
    # Diagnostic: empirical vs theoretical spread (using already computed target_te)
    # Note: target_te is the mean of K refits, so we need to estimate the empirical std
    # For now, we'll use a simple approximation - this is just for diagnostic purposes
    step_sq_sum = getattr(ebm, "_step_sq_sum_", 1.0)
    theoretical_std = sigma * rnorm * np.sqrt(step_sq_sum)
    print(f"Step-squared sum: {step_sq_sum:.3f}")
    print(f"Mean theoretical std: {np.mean(theoretical_std):.3f}")
    print(f"Mean ||r||: {np.mean(rnorm):.3f}")

    # QQ plot
    q = np.linspace(0.5/len(z), 1 - 0.5/len(z), len(z))
    theo = np.sqrt(2.0) * erfinv(2*q - 1)
    plt.figure(figsize=(8,8))
    
    # Plot points with larger size and better visibility
    plt.scatter(theo, z, s=20, alpha=0.6, edgecolors='none', label='Data points')
    
    # Add diagonal reference line
    lo = min(theo.min(), z.min()); hi = max(theo.max(), z.max())
    plt.plot([lo, hi], [lo, hi], "--", color='red', linewidth=2, label='y=x')
    
    # Add some statistics to the plot
    z_std = np.std(z)
    z_mean = np.mean(z)
    plt.text(0.05, 0.95, f'Mean: {z_mean:.3f}\nStd: {z_std:.3f}\nN: {len(z)}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.xlabel("Theoretical N(0,1) quantiles")
    plt.ylabel("(E[f̂]-f̂)/ (σ‖r‖)")
    plt.title("QQ plot (CI vs EBM target)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")
    print(f"Standardized residuals: mean={z_mean:.4f}, std={z_std:.4f}")

if __name__ == "__main__":
    main()
