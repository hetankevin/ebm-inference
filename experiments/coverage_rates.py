#!/usr/bin/env python3
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import inspect
from typing import Optional

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_):
        return iterable

# Ensure repository root (with inferable_ebm_regressor module) is on sys.path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

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
    X = rng.uniform(0.0, 1.0, size=(n, 5))
    #f = (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] + 5*X[:,4])
    f = -5 + (10*np.sin(np.pi*X[:,0]) - 5*np.cos(np.pi*X[:,1])
          + 20*(X[:,2]-0.5)**2 + 10*X[:,3] - 5*X[:,4])
    return X, f


def _plot_distribution(ax, data, color, title, xlabel, target=None):
    try:
        import matplotlib.cm as cm
        cm.register_cmap = lambda *args, **kwargs: None
        import ptitprince as pt
    except Exception as e:  # pragma: no cover - optional dependency
        pt = None
        print(e)
        print('ptitprince import failed, falling back to')
        try:
            import seaborn as sns
        except Exception:
            sns = None

    data = np.asarray(data, dtype=float)
    if data.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=10)
        return

    if pt is not None:
        df_plot = pd.DataFrame({"group": "value", "value": data})
        pt.RainCloud(
            x="group",
            y="value",
            data=df_plot,
            palette=[color],
            bw=0.3,
            width_box=0.25,
            orient="h",
            ax=ax,
            alpha=1,
            dodge=True,
            box_showfliers = False,
            point_size=2.5,
            move=0.05,
        )
        ax.set_yticks([])
        ax.set_ylabel("")
    elif 'sns' in locals() and sns is not None:
        sns.violinplot(x=data, ax=ax, orient="h", color=color, inner=None)
        sns.boxplot(x=data, ax=ax, orient="h", color="white", width=0.15)
        sns.stripplot(x=data, ax=ax, orient="h", color=color, size=3, alpha=0.6)
    else:
        ax.hist(data, bins=20, density=True, color=color, alpha=0.7)
        ax.scatter(data, np.zeros_like(data), color=color, alpha=0.4, s=8)

    if target is not None:
        ax.axvline(target, linestyle="--", color="black", linewidth=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)


def build_interval_plot(
    df_models: pd.DataFrame,
    df_points: pd.DataFrame,
    level: float,
    output: Optional[str],
    show: bool,
    combined: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as ex:  # pragma: no cover - plotting optional
        print(f"Cannot create plot because matplotlib is not available: {ex}")
        return
    plt.style.use('matplotlibrc')

    modes = [
        ("Built-In Confidence Interval", "cov_ci", "w_ci", "#00BEFF"),
        ("Built-In Prediction Interval", "cov_pi", "w_pi", "#F8766D"),
        ("Built-In Reproduction Interval", "cov_ri", "w_ri", "#7CAE00"),
    ]

    n_rows = 2 if combined else 4
    fig, axes = plt.subplots(
        n_rows,
        len(modes),
        figsize=(3 * len(modes), 4 if combined else 8),
    )
    if axes.ndim == 1:
        axes = axes[:, None]

    for col, (title, cov_col, width_col, color) in enumerate(modes):
        model_coverages = df_models[cov_col].to_numpy()
        model_widths = df_models[width_col].to_numpy()
        point_coverages = df_points.groupby("pt")[cov_col].mean().to_numpy()
        point_widths = df_points.groupby("pt")[width_col].mean().to_numpy()

        if combined:
            coverages = np.concatenate([model_coverages, point_coverages])
            widths = np.concatenate([model_widths, point_widths])
            _plot_distribution(axes[0, col], coverages, color, title, "Coverage", target=level)
            #axes[0, col].set_ylabel("Models + Points", fontsize=10)
            _plot_distribution(axes[1, col], widths, color, "", "Width")
            #axes[1, col].set_ylabel("Models + Points", fontsize=10)
        else:
            _plot_distribution(axes[0, col], model_coverages, color, title, "Coverage", target=level)
            axes[0, col].set_ylabel("Models", fontsize=10)

            _plot_distribution(axes[1, col], model_widths, color, "Width", "Width")
            axes[1, col].set_ylabel("Models", fontsize=10)

            _plot_distribution(axes[2, col], point_coverages, color, "Coverage", "Coverage", target=level)
            axes[2, col].set_ylabel("Points", fontsize=10)

            _plot_distribution(axes[3, col], point_widths, color, "Width", "Width")
            axes[3, col].set_ylabel("Points", fontsize=10)
    plt.tight_layout()
    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

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

    loss_history = []

    def track_callback(bag_idx, step_idx, made_progress, best_score):
        if bag_idx == 0:
            loss_history.append(
                dict(step=step_idx, made_progress=bool(made_progress), best_score=float(best_score))
            )
        return False

    (Xtr, ytr, tr), (Xcal, ycal, cal), (Xte, yte, te) = split3(X, y, rng, cal_frac=args.cal_frac, test_frac=args.test_frac)
    f_te = f_true[te]  # Fix oracle indexing

    estimator_kwargs = dict(
        max_rounds=args.rounds,
        max_bins=args.n_bins,
        subsample_rate=args.subsample_rate,
        truncation=args.truncation,
        random_state=base_seed + 1009 * rep,
        max_leaves=args.max_leaves,
        n_jobs=args.n_jobs,
        outer_bags=args.outer_bags,
        bin_level_inference=args.bin_level_inference,
        use_nystrom=args.use_nystrom,
        nystrom_rank=args.nystrom_rank,
        nystrom_ridge=args.nystrom_ridge,
    )

    ebm = InferableEBMRegressor(**estimator_kwargs)
    ebm.callback = track_callback
    ebm = ebm.fit(Xtr, ytr)

    # Quick verification that we're using the correct implementation
    if rep == 0:  # Only debug first repetition
        debug_space = None if args.inference_space == "auto" else args.inference_space
        lo, hi, _ = ebm.predict_intervals(
            Xte[:5],
            level=0.95,
            mode="prediction",
            sigma=1.0,
            inference_space=debug_space,
        )
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

        def summarize_error(y_true, y_pred):
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if not np.any(mask):
                return float("nan"), float("nan")
            residuals = y_true[mask] - y_pred[mask]
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            denom = np.sum(np.abs(y_true[mask]))
            wape = float(np.sum(np.abs(residuals)) / denom) if denom > 1e-12 else float(np.sum(np.abs(residuals)))
            return rmse, wape

        train_rmse, train_wape = summarize_error(ytr, yhat_tr)
        yhat_cal = ebm.predict(Xcal)
        cal_rmse, cal_wape = summarize_error(ycal, yhat_cal)
        yhat_te = ebm.predict(Xte)
        test_rmse, test_wape = summarize_error(yte, yhat_te)

        print("RMSE/WAPE by split:")
        print(f"  train -> RMSE: {train_rmse:.6f}, WAPE: {train_wape:.6f}")
        print(f"  valid -> RMSE: {cal_rmse:.6f}, WAPE: {cal_wape:.6f}")
        print(f"  test  -> RMSE: {test_rmse:.6f}, WAPE: {test_wape:.6f}")

        if loss_history:
            loss_df = pd.DataFrame(loss_history)
            print("\nLoss curve (bag 0 best_score by boosting step):")
            print(loss_df.head())
            if len(loss_df) > 5:
                print("...")
                print(loss_df.tail())

        curve_rounds = np.unique(
            np.linspace(1, max(1, args.rounds), num=min(args.rounds, 10), dtype=int)
        )
        loss_curve_rows = []
        for r in curve_rounds:
            probe = InferableEBMRegressor(
                max_rounds=int(r),
                max_bins=args.n_bins,
                n_jobs=args.n_jobs,
                subsample_rate=args.subsample_rate,
                outer_bags=args.outer_bags,
                truncation=args.truncation,
                random_state=base_seed + 1009*rep,
                bin_level_inference=args.bin_level_inference
            ).fit(Xtr, ytr)
            yhat_tr_probe = probe.predict(Xtr)
            yhat_cal_probe = probe.predict(Xcal)
            tr_rmse, tr_wape = summarize_error(ytr, yhat_tr_probe)
            cal_rmse, cal_wape = summarize_error(ycal, yhat_cal_probe)
            loss_curve_rows.append(
                dict(round=int(r), train_rmse=tr_rmse, valid_rmse=cal_rmse,
                     train_wape=tr_wape, valid_wape=cal_wape)
            )

        if loss_curve_rows:
            loss_curve_df = pd.DataFrame(loss_curve_rows)
            print("\nLoss curve samples (by max_rounds probe):")
            print(loss_curve_df)

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

    inference_space = None if args.inference_space == "auto" else args.inference_space
    ci_l, ci_u, yhat = ebm.predict_intervals(
        Xte,
        level=args.level,
        mode="confidence",
        sigma=sigma,
        inference_space=inference_space,
    )
    pi_l, pi_u, _ = ebm.predict_intervals(
        Xte,
        level=args.level,
        mode="prediction",
        sigma=sigma,
        inference_space=inference_space,
    )
    ri_l, ri_u, _ = ebm.predict_intervals(
        Xte,
        level=args.level,
        mode="reproduction",
        sigma=sigma,
        inference_space=inference_space,
    )

    # Paper-faithful coverage definitions
    cov_ci = float(np.mean((f_te >= ci_l) & (f_te <= ci_u)))
    cov_pi = float(np.mean((yte  >= pi_l) & (yte  <= pi_u)))
    yhat_te = ebm.predict(Xte)
    cov_ri = float(np.mean((yhat_te >= ri_l) & (yhat_te <= ri_u)))

    w_ci = float(np.mean(ci_u - ci_l))
    w_pi = float(np.mean(pi_u - pi_l))
    w_ri = float(np.mean(ri_u - ri_l))
    return (
        dict(
            rep=rep,
            cov_ci=cov_ci,
            cov_pi=cov_pi,
            cov_ri=cov_ri,
            w_ci=w_ci,
            w_pi=w_pi,
            w_ri=w_ri,
            sigma=sigma,
        ),
        pd.DataFrame(
            {
                "rep": rep,
                "pt": np.arange(len(Xte)),
                "ci_l": ci_l,
                "ci_u": ci_u,
                "pi_l": pi_l,
                "pi_u": pi_u,
                "ri_l": ri_l,
                "ri_u": ri_u,
                "cov_ci": (f_te >= ci_l) & (f_te <= ci_u),
                "cov_pi": (yte >= pi_l) & (yte <= pi_u),
                "cov_ri": (yhat_te >= ri_l) & (yhat_te <= ri_u),
                "w_ci": ci_u - ci_l,
                "w_pi": pi_u - pi_l,
                "w_ri": ri_u - ri_l,
                "f": f_te,
                "y": yte,
                "yhat": yhat,
                "yhat_te": yhat_te,
            }
        ),
    )

def main():
    np.seterr(over='ignore', invalid='ignore')  # avoid noisy warnings; we guard widths explicitly
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--noise", type=float, default=1)
    ap.add_argument("--cal-frac", dest="cal_frac", type=float, default=0.1)
    ap.add_argument("--test-frac", dest="test_frac", type=float, default=0.2)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--max-leaves", type=int, default=2)
    ap.add_argument("--n-bins", dest="n_bins", type=int, default=0)
    inf_group = ap.add_mutually_exclusive_group()
    inf_group.add_argument(
        "--bin-level-inference",
        dest="bin_level_inference",
        action="store_true",
        help="Build inference objects in bin space.",
    )
    inf_group.add_argument(
        "--sample-level-inference",
        dest="bin_level_inference",
        action="store_false",
        help="Force inference objects to operate in sample space.",
    )
    ap.set_defaults(bin_level_inference=True)
    ap.add_argument(
        "--inference-space",
        choices=["auto", "samples", "bins"],
        default="auto",
        help="Space used when calling predict_intervals/importance (auto defers to model).",
    )
    ap.add_argument("--n-jobs", type=int, default=-2)
    ap.add_argument("--outer-bags", type=int, default=14)
    ap.add_argument("--subsample-rate", dest="subsample_rate", type=float, default=0.8)
    ap.add_argument("--truncation", type=float, default=100.0)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--use-nystrom", dest="use_nystrom", action="store_true")
    ap.add_argument("--no-nystrom", dest="use_nystrom", action="store_false")
    ap.add_argument("--nystrom-rank", dest="nystrom_rank", type=int, default=64)
    ap.add_argument("--nystrom-ridge", dest="nystrom_ridge", type=float, default=1e-6)
    ap.add_argument("--out", type=str, default="coverage.csv")
    ap.add_argument("--out-summary", dest='out_summary', type=str, default="coverage_summary.csv")
    ap.add_argument(
        "--plot",
        type=str,
        default='plots/coverage_rates.png',
        help="Optional path to save a coverage/width summary plot (requires matplotlib).",
    )
    ap.add_argument(
        "--plot-show",
        action="store_true",
        help="Display the coverage/width plot interactively after saving.",
    )
    ap.add_argument(
        "--plot-combined",
        action="store_true",
        default=True,
        help="Combine model and point distributions into two rows when plotting.",
    )
    args = ap.parse_args()

    max_workers = min(args.reps, os.cpu_count() or 1)

    dfs = []
    if max_workers <= 1:
        outputs = [run_rep(r, args) for r in tqdm(range(args.reps))]
        rows, dfs = zip(*outputs)
        rows = list(rows)
        dfs = list(dfs)
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_rep, r, args): r for r in range(args.reps)}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                row, df = fut.result()
                rows.append(row)
                dfs.append(df)

    rows.sort(key=lambda entry: entry["rep"])
    df = pd.DataFrame(rows)
    df.to_csv(args.out_summary, index=False)
    points_df = pd.concat(dfs, ignore_index=True).sort_values(['rep', 'pt'])
    points_df.to_csv(args.out, index=False)
    print(df.describe())

    if args.plot or args.plot_show:
        output_path = args.plot
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        build_interval_plot(
            df,
            points_df,
            level=args.level,
            output=output_path,
            show=args.plot_show,
            combined=args.plot_combined,
        )

if __name__ == "__main__":
    main()
