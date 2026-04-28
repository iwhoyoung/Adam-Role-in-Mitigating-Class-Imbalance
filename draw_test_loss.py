#!/usr/bin/env python3
"""plot_lr_curves.py

Generates loss–learning-rate curves for a set of models/optimizers from
CSV logs that follow the naming convention shown below.

Changes (2025-06-20)
====================
• **X-axis now categorical, uniformly spaced** — learning rates are
  plotted from *large → small* at equal spacing; tick labels are the LR
  values (scientific notation where appropriate).  This replaces the
  previous logarithmic scale.

File naming template
====================
    adapolycifar_r10_{model}_{optimizer}_batch256_200e_lr{lr}_seed0.csv

Example
-------
    adapolycifar_r10_vgg16bn_sgd_batch256_200e_lr0.005_seed0.csv

Each CSV must contain the columns:
    "train loss"   – training loss per epoch (one row per epoch)
    "test loss"    – test/val loss per epoch

The script extracts the *last* row (final epoch) from each file to obtain
one (learning-rate, loss) point.  For every model/optimizer it plots two
lines (train vs. test) across seven LR values — the best LR plus the
three larger and the three smaller neighbours taken from a discrete LR
grid between 5 and 1e-7.

Missing files/columns are reported on stderr; the corresponding points
are skipped (drawn as gaps).

Usage
=====
    python plot_lr_curves.py --data-dir /path/to/csv \
                             --output-dir /path/to/figures \
                             --image-format png

Dependencies
============
    pandas
    matplotlib>=3.5
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BEST_LRS = {
    "resnet18_cifar_r10": {"sgd": 0.5,  "adam": 0.001,  "adam_bn": 0.0001, "adam_ini": 0.05},
    "resnet50_cifar_r10": {"sgd": 0.1,  "adam": 0.001,  "adam_bn": 0.0001, "adam_ini": 0.05},
    "vgg16bn_cifar_r10":  {"sgd": 0.1,  "adam": 0.0005, "adam_bn": 1e-5,   "adam_ini": 0.005},
    "vitb_cifar_r10":     {"sgd": 0.05, "adam": 5e-5,   "adam_bn": 5e-6,  "adam_ini": 0.0005},
    "vits_cifar_r10":     {"sgd": 0.05, "adam": 5e-5,   "adam_bn": 5e-6,  "adam_ini": 0.001},
}

# Log-spaced learning-rate grid from 5 ↓ 1e-7
LR_SPACE = [5.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005,
            0.001, 0.0005, 0.0001, 5e-05, 1e-05,
            5e-06, 1e-06, 5e-07, 1e-07]

Y_MIN, Y_MAX = -1, 5

FILE_TEMPLATE = "adapolycifar_r10_{model}_{optimizer}_batch256_200e_lr{lr}_seed0.csv"

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def lr_neighbourhood(best: float, k: int = 3) -> list[float]:
    """Return list: [larger3 … larger1, best, smaller1 … smaller3] (descending)."""
    if best not in LR_SPACE:
        extended = sorted(set(LR_SPACE + [best]), reverse=True)
    else:
        extended = LR_SPACE

    idx = extended.index(best)
    left = extended[max(0, idx - k): idx]            # already descending
    right = extended[idx + 1: idx + 1 + k]           # descending
    return left + [best] + right                     # overall descending


def lr_to_strings(lr: float) -> list[str]:
    """Return possible string representations of *lr* suitable for filenames."""
    dec = ("{:.10f}".format(lr)).rstrip("0").rstrip(".")
    g   = ("{:g}".format(lr))
    sci = ("{:.0e}".format(lr))
    return list(dict.fromkeys([dec, g, sci]))


def find_csv(data_dir: Path, model: str, optimizer: str, lr: float) -> Path | None:
    lr_strings = lr_to_strings(lr)
    model_part = model.replace("_cifar_r10", "")
    for s in lr_strings:
        fname = FILE_TEMPLATE.format(model=model_part, optimizer=optimizer, lr=s)
    
        p = data_dir / fname
        print(p)
        if p.is_file():
            return p
    return None


def read_last_losses(csv_path: Path) -> tuple[float, float] | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        sys.stderr.write(f"[Error] Failed to read '{csv_path}': {exc}\n")
        return None

    for col in ("train loss","test loss"):
        if col not in df.columns:
            sys.stderr.write(f"[MissingColumn] '{csv_path.name}' lacks column '{col}'\n")
            return None

    return float(df["train loss"].iloc[-1]), float(df["test loss"].iloc[-1])


def fmt_lr(lr: float) -> str:
    """Pretty string for tick label."""
    return f"{lr:g}" if lr >= 1e-3 else f"{lr:.0e}"


def plot_curves(model: str,
                optimizer: str,
                lrs: list[float],
                train_vals: list[float | None],
                test_vals: list[float | None],
                out_dir: Path,
                image_format: str = "png") -> None:
    # Build arrays aligned to full lr list, replacing missing with NaN
    xs = list(range(len(lrs)))
    tr_plot = [v if v is not None else math.nan for v in train_vals]
    te_plot = [v if v is not None else math.nan for v in test_vals]

    if all(math.isnan(v) for v in tr_plot) and all(math.isnan(v) for v in te_plot):
        sys.stderr.write(f"[Skip] No data for {model} | {optimizer}\n")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_xlabel("Learning rate (big → small)")
    ax.set_ylabel("Loss")
    ax.set_ylim(Y_MIN, Y_MAX)

    ax.plot(xs, tr_plot, marker="o", label="train loss")
    ax.plot(xs, te_plot, marker="s", label="test loss")

    ax.set_xticks(xs)
    ax.set_xticklabels([fmt_lr(lr) for lr in lrs], rotation=45, ha="right")

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(f"{model}  |  {optimizer}")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"{model}_{optimizer}_loss_vs_lr.{image_format}"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"[Saved] {outfile}")

# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate loss–LR plots from CSV logs.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing CSV files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to store figures.")
    parser.add_argument("--image-format", default="png", choices=["png", "pdf", "svg"],
                        help="Figure file format.")
    args = parser.parse_args()

    out_dir = args.output_dir or args.data_dir

    for model, optim_dict in BEST_LRS.items():
        for optimizer, best_lr in optim_dict.items():
            lr_list = lr_neighbourhood(best_lr, k=3)
            train_vals, test_vals = [], []

            for lr in lr_list:
                print(lr)
                csv = find_csv(args.data_dir, model, optimizer, lr)
                if csv is None:
                    sys.stderr.write(f"[Missing] {model} | {optimizer} | lr={lr} : file not found\n")
                    train_vals.append(None)
                    test_vals.append(None)
                    continue
                losses = read_last_losses(csv)
                if losses is None:
                    train_vals.append(None)
                    test_vals.append(None)
                else:
                    train_vals.append(losses[0])
                    test_vals.append(losses[1])

            plot_curves(model, optimizer, lr_list, train_vals, test_vals, out_dir, args.image_format)


if __name__ == "__main__":
    main()