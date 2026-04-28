import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import matplotlib.cm as cm
import numpy as np

def find_matching_csv_files(directory):
    pattern = re.compile(r'adapolycifar_r10_(.+?)_batch.*\.csv')
    matching_files = []
    for filename in os.listdir(directory):
        if pattern.match(filename):
            matching_files.append(filename)
    return matching_files

def extract_model_and_optimizer(filename):
    # 从形如 adapolycifar_r10_resnet18_adam_batch... 中提取 model 与 optimizer
    match = re.match(r'adapolycifar_r10_(.+?)_(adam|sgd)_batch.*\.csv', filename)
    if match:
        model_raw = match.group(1).lower()
        optimizer = match.group(2).capitalize()
        # 简化模型名
        model_short = simplify_model_name(model_raw)
        return model_short, optimizer
    return None, None

def simplify_model_name(model):
    # 你可以根据需要进一步精简
    mapping = {
        'resnet18': 'Res18',
        'resnet50': 'Res50',
        'vit': 'ViT',
        'vits':'ViT-S',
        'vitb':'ViT-B',
        'vgg16bn':'VGG16',
        'mobilenetv2': 'MobV2',
        'convnext': 'ConvNeXt',
    }
    return mapping.get(model.lower(), model)

def find_acc_column(df):
    for key in ['train loss', 'test loss', 'acc']:
        matches = [col for col in df.columns if col.endswith(key)]
        if matches:
            return matches[0]
    raise ValueError("No accuracy column found.")

def plot_metrics_separate_per_model(csv_dir, output_dir):
    matching_files = find_matching_csv_files(csv_dir)
    if not matching_files:
        print("No matching CSV files found.")
        return

    # 整理：模型 → {SGD: path, Adam: path}
    model_to_files = {}
    for filename in matching_files:
        model, optimizer = extract_model_and_optimizer(filename)
        if not model or not optimizer:
            continue
        model_to_files.setdefault(model, {})[optimizer] = filename

    # 配色方案
    color_map = {'Sgd': 'tab:blue', 'Adam': 'tab:orange'}
    metric_keys = ['train loss', 'test loss', 'acc']
    metric_titles = {
        'train loss': 'Training Loss',
        'test loss': 'Test Loss',
        'acc': 'Accuracy'
    }
    ylabels = {
        'train loss': 'Loss',
        'test loss': 'Loss',
        'acc': 'Accuracy'
    }

    for model, entries in model_to_files.items():
        for metric_key in metric_keys:
            plt.figure(figsize=(8, 5))
            found_any = False
            for optimizer in ['Sgd', 'Adam']:
                filename = entries.get(optimizer)
                if not filename:
                    print(f"Warning: {model} missing {optimizer}")
                    continue

                filepath = os.path.join(csv_dir, filename)
                df = pd.read_csv(filepath)

                # 寻找对应的列
                matches = [col for col in df.columns if col.endswith(metric_key)]
                if not matches:
                    print(f"Skipping {filename}: no '{metric_key}' column found.")
                    continue
                metric_col = matches[0]

                linestyle = '--' if optimizer == 'Sgd' else '-'
                label = f"{optimizer} ({model})"
                plt.plot(df[metric_col], label=label, color=color_map[optimizer], linestyle=linestyle, linewidth=2)
                print(f"Plotted {label} {metric_key} from {filename}")
                found_any = True

            if not found_any:
                plt.close()
                continue

            plt.xlabel("Epoch", fontsize=16)
            plt.ylabel(ylabels[metric_key], fontsize=16)
            plt.title(f"{metric_titles[metric_key]} Comparison: {model}", fontsize=16)
            plt.legend(fontsize=10, ncol=2)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            os.makedirs(output_dir, exist_ok=True)
            key_sanitized = metric_key.replace(" ", "_")
            output_file = os.path.join(output_dir, f"{model}_{key_sanitized}_adam_vs_sgd.png")
            plt.savefig(output_file, dpi=300)
            plt.close()
            print(f"Saved plot to {output_file}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot accuracy curves with style differentiation.")
    parser.add_argument('--csv_dir', default='log/select_result', help='Directory containing CSV files')
    parser.add_argument('--output', default='plots/loss_adam_vs_sgd.png', help='Output image path')
    args = parser.parse_args()

    plot_metrics_separate_per_model(args.csv_dir, args.output)
