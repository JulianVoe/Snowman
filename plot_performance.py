
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_performance_column(df):
    # Try common variants
    candidates = [
        "pixels_per_s",
        "performance(1/s)",
        "performance(pixel/s)",
        "performance",
        "throughput",
        "pixels_per_sec",
        "perf"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: try any column containing 'perf' (case-insensitive)
    for c in df.columns:
        if 'perf' in c.lower():
            return c
    raise KeyError("Could not find a performance column. Available columns: " + ", ".join(df.columns))

def plot_perf(csv_path: Path, title: str, outpath: Path):
    df = pd.read_csv(csv_path)
    if "workers" not in df.columns:
        raise KeyError("CSV must have a 'workers' column")

    perf_col = find_performance_column(df)
    n = df["workers"].to_numpy()
    y = df[perf_col].to_numpy()

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(n, y, marker='o', label="Measured performance")
    plt.xlabel("Workers (n)")
    plt.ylabel(f"Performance [{perf_col}]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    return outpath

def main():
    ap = argparse.ArgumentParser(description="Plot performance vs workers for weak and strong scaling.")
    ap.add_argument("--weak", type=Path, required=True, help="Path to weak-scaling CSV")
    ap.add_argument("--strong", type=Path, required=True, help="Path to strong-scaling CSV")
    ap.add_argument("--outdir", type=Path, default=Path("."), help="Output directory for figures")
    args = ap.parse_args()

    weak_out = args.outdir / "weak_performance.png"
    strong_out = args.outdir / "strong_performance.png"

    wpng = plot_perf(args.weak, "Weak scaling: Performance vs Workers", weak_out)
    spng = plot_perf(args.strong, "Strong scaling: Performance vs Workers", strong_out)

    print(f"Weak performance figure: {wpng}")
    print(f"Strong performance figure: {spng}")

if __name__ == "__main__":
    main()
