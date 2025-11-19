
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_weak_scaling(csv_path: Path, outdir: Path) -> Path:
    df = pd.read_csv(csv_path)
    # Expect columns: workers, grid, #snowmen, seconds, performance(1/s) or pixels_per_s, efficiency, speedup (ignored)
    n = df["workers"].to_numpy()
    e = df["efficiency"].to_numpy()
    y = n * e  # scaled efficiency

    # x-range for theory lines
    x = np.linspace(n.min(), n.max(), 200)
    # Ideal: y = x
    y_ideal = x
    # Gustafson with s=0.1: y = s + (1-s)*x
    s = 0.1
    y_gust = s + (1.0 - s) * x

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(n, y, marker='o', label="Measured: n·efficiency")
    plt.plot(x, y_ideal, linestyle='--', label="Ideal: y=x")
    plt.plot(x, y_gust, linestyle='--', label="Gustafson s=0.1: y=0.1+0.9x")
    plt.xlabel("Workers (n)")
    plt.ylabel("Scaled efficiency  (n·e)")
    plt.title("Weak scaling")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    outpath = outdir / "weak_scaling.png"
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    return outpath

def plot_strong_scaling(csv_path: Path, outdir: Path) -> Path:
    df = pd.read_csv(csv_path)
    n = df["workers"].to_numpy()
    s_meas = df["speedup"].to_numpy()
    s_ideal = df["ideal_speedup"].to_numpy()

    # Amdahl with s=0.1: S(x) = 1 / (s + (1-s)/x)
    s_serial = 0.1
    x = np.linspace(max(1, n.min()), n.max(), 300)
    s_amdahl = 1.0 / (s_serial + (1.0 - s_serial) / x)

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(n, s_meas, marker='o', label="Measured speedup")
    plt.plot(n, s_ideal, linestyle='--', label="Ideal speedup")
    plt.plot(x, s_amdahl, linestyle='--', label="Amdahl s=0.1: 1/(0.1 + 0.99/x)")
    plt.xlabel("Workers (n)")
    plt.ylabel("Speedup")
    plt.title("Strong scaling")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    outpath = outdir / "strong_scaling.png"
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    return outpath

def main():
    ap = argparse.ArgumentParser(description="Analyze weak & strong scaling CSVs and generate plots.")
    ap.add_argument("--weak", type=Path, required=True, help="Path to weak-scaling CSV")
    ap.add_argument("--strong", type=Path, required=True, help="Path to strong-scaling CSV")
    ap.add_argument("--outdir", type=Path, default=Path("."), help="Output directory for figures")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    weak_png = plot_weak_scaling(args.weak, args.outdir)
    strong_png = plot_strong_scaling(args.strong, args.outdir)

    print(f"Weak-scaling figure: {weak_png}")
    print(f"Strong-scaling figure: {strong_png}")

if __name__ == "__main__":
    main()
