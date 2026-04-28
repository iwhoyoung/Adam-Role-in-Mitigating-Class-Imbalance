#!/usr/bin/env python3
"""plot_lr_curves.py

Generates accuracy–train-loss curves for a set of models/optimizers from
CSV logs. The x-axis is a categorical learning rate scale from large to
small. Each point is extracted from the final epoch of the CSV.

Changes (2025-06-21)
====================
• Main y-axis: accuracy (0–60)
• Secondary y-axis: train loss (0–1)
• x-axis: learning rates (large → small), uniformly spaced

Usage
=====
    python plot_lr_curves.py --data-dir /path/to/csv \
                             --output-dir /path/to/figures \
                             --image-format png
"""

import argparse
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BEST_LRS = {
    "resnet18_cifar_r10": {"sgd": 1.0,  "adam": 0.001,  "adam_bn": 0.0005, "adam_ini": 0.05,   "adam_sbn": 0.01},
    "resnet50_cifar_r10": {"sgd": 1.0,  "adam": 0.001,  "adam_bn": 0.0001, "adam_ini": 0.5,   "adam_sbn": 0.005},
    "vgg16bn_cifar_r10":  {"sgd": 0.1,  "adam": 0.0005, "adam_bn": 1e-5,   "adam_ini": 0.01,   "adam_sbn": 0.005},
    "vitb_cifar_r10":     {"sgd": 0.05, "adam": 5e-5,   "adam_bn": 5e-6,  "adam_ini": 0.005,   "adam_sbn": 0.0005},
    "vits_cifar_r10":     {"sgd": 0.05, "adam": 5e-5,   "adam_bn": 1e-5,  "adam_ini": 0.005,   "adam_sbn": 0.001},
    }


LR_SPACE = [50.0,10.0,5.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005,
            0.001, 0.0005, 0.0001, 5e-05, 1e-05,
            5e-06, 1e-06, 5e-07, 1e-07]

FILE_TEMPLATE = "adapolycifar_r10_{model}_{optimizer}_batch256_200e_lr{lr}_seed7_v2.csv"

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def lr_neighbourhood(best: float, k: int = 3) -> list[float]:
    if best not in LR_SPACE:
        extended = sorted(set(LR_SPACE + [best]), reverse=True)
    else:
        extended = LR_SPACE

    idx = extended.index(best)
    left = extended[max(0, idx - k): idx]
    right = extended[idx + 1: idx + 1 + k]
    return left + [best] + right

def lr_to_strings(lr: float) -> list[str]:
    """
    Return a list of possible string representations of a learning rate.
    Includes representations like '1', '1.0', '1.00', '1e+00', etc.
    """
    variants = set()
    variants.add("{:.10f}".format(lr).rstrip("0").rstrip("."))  # e.g., "0.1"
    variants.add("{:.1f}".format(lr))                            # e.g., "1.0"
    variants.add("{:g}".format(lr))                              # e.g., "1"
    variants.add("{:.0e}".format(lr))                            # e.g., "1e+00"

    return list(variants)


def find_csv(data_dir: Path, model: str, optimizer: str, lr: float) -> Path | None:
    lr_strings = lr_to_strings(lr)
    model_part = model.replace("_cifar_r10", "")
    for s in lr_strings:
        fname = FILE_TEMPLATE.format(model=model_part, optimizer=optimizer, lr=s)
        #print(fname)
        p = data_dir / fname
        if p.is_file():
            return p
    return None

def read_last_values(csv_path: Path) -> tuple[float, float] | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        sys.stderr.write(f"[Error] Failed to read '{csv_path}': {exc}\n")
        return None

    for col in ("acc", "train loss"):
        if col not in df.columns:
            sys.stderr.write(f"[MissingColumn] '{csv_path.name}' lacks column '{col}'\n")
            return None
    #print(float(df["train loss"].iloc[-1]))
    #print(np.nan)
    #print(np.isnan(df["train loss"].iloc[-1]))
    if np.isnan(df["train loss"].iloc[-1]):
        return float(df["acc"].iloc[-1]), 1000
    return float(df["acc"].iloc[-1]), float(df["train loss"].iloc[-1])

def fmt_lr(lr: float) -> str:
    return f"{lr:.0e}"

def plot_curves(model: str,
                optimizer: str,
                lrs: list[float],
                acc_vals: list[float | None],
                train_vals: list[float | None],
                out_dir: Path,
                image_format: str = "png") -> None:
    xs = list(range(len(lrs)))
    acc_plot = [v if v is not None else math.nan for v in acc_vals]
    tr_plot = [v if v is not None else math.nan for v in train_vals]

    if all(math.isnan(v) for v in acc_plot) and all(math.isnan(v) for v in tr_plot):
        sys.stderr.write(f"[Skip] No data for {model} | {optimizer}\n")
        return

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.set_ylim(10, 60)
    acc_line, = ax1.plot(xs, acc_plot, marker="o", color="tab:blue", label="Accuracy")
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=16)

    ax2 = ax1.twinx()
    ax2.set_ylim(-0.5, 2.0)
    tr_line, = ax2.plot(xs, tr_plot, marker="s", color="tab:red", label="Train loss")
    ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=16)

    ax1.set_xticks(xs)
    ax1.set_xticklabels([fmt_lr(lr) for lr in lrs], rotation=45, ha="right", fontsize=16)

    ax1.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    # 添加图例
    ax1.legend(handles=[acc_line, tr_line], loc="upper left", fontsize=16)


    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"{model}_{optimizer}_acc_loss_vs_lr.{image_format}"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

def find_best_lr(data_dir: Path, model: str, optimizer: str) -> tuple[float | None, float | None]:
    acc_map = {}
    for lr in LR_SPACE:
        csv = find_csv(data_dir, model, optimizer, lr)
        if csv is None:
            
            continue
        vals = read_last_values(csv)
        if vals is not None:
            acc_map[lr] = vals[0]  # 只记录acc

    if not acc_map:
        return None, None
    
    best_lr = max(acc_map, key=acc_map.get)
    return best_lr, acc_map[best_lr]

# ---------------------------------------------------------------------
# Main driver (已修改)
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate accuracy–loss plots from CSV logs.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing CSV files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to store figures.")
    parser.add_argument("--image-format", default="png", choices=["png", "pdf", "svg"],
                        help="Figure file format.")
    args = parser.parse_args()

    out_dir = args.output_dir or args.data_dir
    # for model in ["resnet50_cifar_r10"]:
    #     for optimizer in [ "adam_ini"]:
    # for model in ["resnet18_cifar_r10", "resnet50_cifar_r10", "vgg16bn_cifar_r10", "vitb_cifar_r10", "vits_cifar_r10"]:
    for model in ["resnet18_cifar_r10", "vits_cifar_r10"]:
        #for optimizer in ["sgd", "adam", "adam_bn", "adam_ini"]:
        for optimizer in ["rmsprop"]:
            best_lr, best_acc = find_best_lr(args.data_dir, model, optimizer)
            if best_lr is None:
                sys.stderr.write(f"[Skip] No data found for {model} | {optimizer}\n")
                continue

            print(f"[BestLR] {model:20} | {optimizer:8} | lr = {best_lr:<10g} | acc = {best_acc:.2f}")

            lr_list = lr_neighbourhood(best_lr, k=3)
            print(lr_list)
            acc_vals, train_vals = [], []

            for lr in lr_list:
                #print(lr)
                csv = find_csv(args.data_dir, model, optimizer, lr)
                print(csv)
                if csv is None:
                    sys.stderr.write(f"[Missing] {model} | {optimizer} | lr={lr} : file not found\n")
                    acc_vals.append(None)
                    train_vals.append(None)
                    continue
                vals = read_last_values(csv)
                #print(vals)
                if vals is None:
                    acc_vals.append(None)
                    train_vals.append(None)
                else:
                    acc_vals.append(vals[0])
                    train_vals.append(vals[1])

            plot_curves(model, optimizer, lr_list, acc_vals, train_vals, out_dir, args.image_format)

if __name__ == "__main__":
    main()