"""Run the whole project end to end (about 10-15 minutes on a laptop CPU).

Usage:
    python -P run_all.py                    # download (skips existing files) + everything
    python -P run_all.py --skip-download    # use the raw files already in data/raw
    python -P run_all.py --skip-robustness  # HB_NORTH only
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SENSITIVITY = [["--e-max", "100", "--tag", "dur1h"], ["--e-max", "400", "--tag", "dur4h"],
               ["--rte", "0.85", "--tag", "rte85"], ["--rte", "0.95", "--tag", "rte95"],
               ["--kappa", "0", "--tag", "kappa0"], ["--kappa", "10", "--tag", "kappa10"],
               ["--kappa", "20", "--tag", "kappa20"]]
WEST = "robustness_hb_west"


def run(script, *args):
    cmd = [sys.executable, "-P", os.path.join("src", script), *args]
    print(">>", " ".join(cmd[2:]), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    print(f"   {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    if not a.skip_download:
        run("download_data.py", ".")
    run("build_dataset.py", ".")
    run("features.py", ".")
    run("pca_analysis.py", ".")
    run("forecast_baselines.py", ".")
    run("forecast_lgbm.py", ".")
    run("forecast_xgb.py", ".")
    run("forecast_lstm.py", ".")
    run("backtest.py", ".")
    for s in SENSITIVITY:
        run("backtest.py", ".", "--milp-only", *s)
    run("analysis.py", ".")
    run("figures.py", ".")
    run("experiments.py", ".")

    if not a.skip_robustness:
        for sub in ("data/processed", "results", "figures"):
            os.makedirs(os.path.join(ROOT, WEST, sub), exist_ok=True)
        run("build_dataset.py", WEST, "--hub", "HB_WEST", "--weather", "midland", "--raw", "data/raw")
        run("features.py", WEST, "--hub", "HB_WEST", "--weather", "midland", "--raw", "data/raw")
        run("forecast_baselines.py", WEST)
        run("forecast_lgbm.py", WEST)
        run("forecast_lstm.py", WEST, "--hub", "HB_WEST")
        run("backtest.py", WEST)

    print(f"All done in {(time.time() - t0) / 60:.1f} min. Tables: results/, figures: figures/, "
          f"summary: RESULTS.md")


if __name__ == "__main__":
    main()
