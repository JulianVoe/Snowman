#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_curve(csv_path: Path):
    xs, ys = [], []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # skip empty lines / malformed rows gracefully
            if not row.get("workers") or not row.get("grid") or not row.get("seconds"):
                continue
            workers = int(row["workers"])
            grid = float(row["grid"])
            seconds = float(row["seconds"])
            xs.append(workers)
            ys.append(grid * grid / seconds)

    if not xs:
        return [], []

    paired = sorted(zip(xs, ys), key=lambda p: p[0])
    xs_sorted, ys_sorted = zip(*paired)
    return list(xs_sorted), list(ys_sorted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()

    datadir: Path = args.datadir
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fig_strong, ax_strong = plt.subplots()
    fig_weak, ax_weak = plt.subplots()

    # iterate over immediate subdirectories of datadir
    for subdir in sorted(p for p in datadir.iterdir() if p.is_dir()):
        strong_csv = subdir / "strong_scaling.csv"
        weak_csv = subdir / "weak_scaling.csv"
        cfg_txt = subdir / "config.txt"

        if not (strong_csv.exists() and weak_csv.exists() and cfg_txt.exists()):
            continue

        label = cfg_txt.read_text(encoding="utf-8").strip()

        xs_s, ys_s = load_curve(strong_csv)
        xs_w, ys_w = load_curve(weak_csv)

        if xs_s:  # only plot if we actually have data
            ax_strong.plot(xs_s, ys_s, marker="o", linewidth=0.8, markersize=3, label=label)

        if xs_w:  # same for weak scaling
            ax_weak.plot(xs_w, ys_w, marker="o", linewidth=0.8, markersize=3, label=label)

    # strong scaling figure
    ax_strong.set_xlabel("workers")
    ax_strong.set_ylabel("grid² / seconds")
    ax_strong.set_title("Strong scaling: grid² / s vs workers")
    ax_strong.grid(True, linestyle="--", linewidth=0.5)
    if ax_strong.get_legend_handles_labels()[0]:
        ax_strong.legend()
    fig_strong.tight_layout()
    fig_strong.savefig(outdir / "strong_scaling.png", dpi=300)

    # weak scaling figure
    ax_weak.set_xlabel("workers")
    ax_weak.set_ylabel("grid² / seconds")
    ax_weak.set_title("Weak scaling: grid² / s vs workers")
    ax_weak.grid(True, linestyle="--", linewidth=0.5)
    if ax_weak.get_legend_handles_labels()[0]:
        ax_weak.legend()
    fig_weak.tight_layout()
    fig_weak.savefig(outdir / "weak_scaling.png", dpi=300)


if __name__ == "__main__":
    main()
