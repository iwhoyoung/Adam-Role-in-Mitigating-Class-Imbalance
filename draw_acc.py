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
    for key in ['acc', 'val_acc', 'top1_acc', 'test_acc']:
        matches = [col for col in df.columns if col.endswith(key)]
        if matches:
            return matches[0]
    raise ValueError("No accuracy column found.")

def plot_acc_with_styles(csv_dir, output_path):
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

    # 准备配色（统一模型色）
    model_names = sorted(model_to_files.keys())
    color_map = {model: color for model, color in zip(model_names, cm.tab10(np.linspace(0, 1, len(model_names))))}
    print(model_to_files)
    plt.figure(figsize=(9, 6))

    for model in model_names:
        entries = model_to_files.get(model, {})
        color = color_map[model]

        for optimizer in ['Sgd', 'Adam']:  # 固定顺序：SGD在前
            filename = entries.get(optimizer)
            if not filename:
                continue

            filepath = os.path.join(csv_dir, filename)
            df = pd.read_csv(filepath)

            try:
                acc_col = find_acc_column(df)
            except ValueError as e:
                print(f"Skipping {filename}: {e}")
                continue

            linestyle = '--' if optimizer == 'Sgd' else '-'
            label = f"{optimizer.upper()} ({model})"

            plt.plot(df[acc_col], label=label, color=color, linestyle=linestyle, linewidth=2)
            print(f"Plotted: {label} from {filename}")

    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot accuracy curves with style differentiation.")
    parser.add_argument('--csv_dir', default='log/select_result', help='Directory containing CSV files')
    parser.add_argument('--output', default='plots/acc_adam_vs_sgd.png', help='Output image path')
    args = parser.parse_args()

    plot_acc_with_styles(args.csv_dir, args.output)
