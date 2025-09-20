
#!/usr/bin/env python3
import argparse, subprocess, sys, os, json

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="exp_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    py = sys.executable
    run([py, "coverage_rates.py", "--out", os.path.join(args.outdir, "coverage_summary.csv")])
    run([py, "qq_plot.py", "--out", os.path.join(args.outdir, "qq_plot_ci.png")])
    run([py, "1d_intervals.py", "--out", os.path.join(args.outdir, "intervals_1d.png")])
    run([py, "variable_importance.py", "--out", os.path.join(args.outdir, "vi_results.csv")])
    run([py, "mse_comparison.py", "--out", os.path.join(args.outdir, "mse_curves.csv")])
    run([py, "matchups.py", "--out", os.path.join(args.outdir, "matchups.csv")])
    # Optuna is optional and may not be installed
    try:
        run([py, "optuna_mse.py", "--out", os.path.join(args.outdir, "optuna_ebm.csv")])
    except Exception as e:
        print("Optuna run skipped:", e)

if __name__ == "__main__":
    main()
