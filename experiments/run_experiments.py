
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

    run([py, "coverage_rates.py", "--out", os.path.join(args.outdir, "coverage_summary.csv")] + base)
    run([py, "qq_plot.py", "--out", os.path.join(args.outdir, "qq_plot_ci.png")] + base)
    run([py, "1d_intervals.py", "--out", os.path.join(args.outdir, "intervals_1d.png")] + base)
    run([py, "variable_importance.py", "--out", os.path.join(args.outdir, "vi_results.csv")] + base)
    run([py, "mse_comparison.py", "--out", os.path.join(args.outdir, "mse_curves.csv")] + base)
    run([py, "matchups.py", "--out", os.path.join(args.outdir, "matchups.csv")] + base)
    try:
        run([py, "optuna_mse.py", "--out", os.path.join(args.outdir, "optuna_ebm.csv")] + base)
    except Exception as e:
        print("Optuna run skipped:", e)

if __name__ == "__main__":
    main()
