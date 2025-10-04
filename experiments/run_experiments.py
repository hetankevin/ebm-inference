
#!/usr/bin/env python3
import argparse, subprocess, sys, os

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="exp_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    py = sys.executable
    base: list[str] = []

    # Core experiments
    run([py, "coverage_rates.py", "--out", os.path.join(args.outdir, "coverage_summary.csv")] + base)
    run([py, "qq_plot.py", "--out", os.path.join(args.outdir, "qq_plot_ci.png")] + base)
    run([py, "1d_intervals.py", "--out", os.path.join(args.outdir, "intervals_1d.png")] + base)
    run([py, "variable_importance.py", "--out", os.path.join(args.outdir, "vi_results.csv")] + base)
    run([py, "mse_comparison.py", "--out", os.path.join(args.outdir, "mse_curves.csv")] + base)
    run([py, "matchups.py", "--out", os.path.join(args.outdir, "matchups.csv")] + base)
    
    # Additional experiments (with error handling)
    try:
        run([py, "coverage_1d_plot.py", "--out", os.path.join(args.outdir, "coverage_1d_plot.png")] + base)
    except Exception as e:
        print("coverage_1d_plot.py skipped:", e)
    
    try:
        run([py, "feature_interval_plots.py", "--out", os.path.join(args.outdir, "feature_intervals.png")] + base)
    except Exception as e:
        print("feature_interval_plots.py skipped:", e)
    
    try:
        run([py, "obesity_inferable_mse.py", "--out", os.path.join(args.outdir, "obesity_mse_results.csv")] + base)
    except Exception as e:
        print("obesity_inferable_mse.py skipped:", e)
    
    # Optional experiments (with error handling)
    try:
        run([py, "optuna_mse.py", "--out", os.path.join(args.outdir, "optuna_ebm.csv")] + base)
    except Exception as e:
        print("Optuna run skipped:", e)

if __name__ == "__main__":
    main()
