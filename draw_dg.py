import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def find_columns(df, suffix):
    return [col for col in df.columns if col.endswith(suffix)]

def plot_series(df1, df2, columns, labels, ylabel, title, output_dir):
    plt.figure(figsize=(10, 6))
    for col in columns:
        plt.plot(df1[col].values, label=f"{labels[0]} {col}")
        plt.plot(df2[col].values, label=f"{labels[1]} {col}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()

import matplotlib.cm as cm
import numpy as np

import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

def plot_top5_last5(df1, suffix, label1, label2, ylabel, output_dir, csv1_name, model_name, opt_name):
    cols1 = find_columns(df1, suffix)
    common_cols = list(set(cols1))

    if len(common_cols) < 10:
        print(f"Not enough columns with suffix '{suffix}' to plot top 5 and last 5 classes.")
        return

    # 提取并按 class index 排序
    def extract_class_idx(col_name):
        return int(col_name.split('_')[1])
    
    common_cols_sorted = sorted(common_cols, key=extract_class_idx)
    #print(common_cols_sorted)
    
    # 支持任意选择方式：等间隔 or 自定义
    selected_indices = np.linspace(0, 99, 11, dtype=int)
    selected_cols = [common_cols_sorted[i] for i in selected_indices]
    #print(selected_cols)
    class_indices = [extract_class_idx(col) for col in selected_cols]
    colors = [cm.viridis(idx / 99) for idx in class_indices]  # 颜色映射基于 class index

    # 获取group_0的数据作为横轴
    if 'group_0' + suffix in selected_cols:
        group_0_idx = selected_cols.index('group_0' + suffix)
    else:
        print("Warning: group_0 not found in selected columns. Using first column as reference.")
        group_0_idx = 0
    
    # 提取数据
    x_data = df1[selected_cols[group_0_idx]].values[:]  # 使用group_0作为横轴
    #print(x_data)
    y_min, y_max = -1, 8  # Y轴范围
    
    def plot(df, filename_suffix):
        plt.figure(figsize=(5, 4))
        
        
        # 绘制其他组相对于group_0的曲线
        for i, col in enumerate(selected_cols):
            y_data = df[col].values[:]
            plt.plot(x_data, y_data, color=colors[i], linewidth=2, label=f'Class {class_indices[i]}')
            #plt.scatter(x_data, y_data, color=colors[i], s=5, label=f'Class {class_indices[i]}')
        
        plt.xlabel(f'group_0{suffix}', fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylim(y_min, y_max)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 只在图例中显示前5个和后5个类别的标签
        handles, labels = plt.gca().get_legend_handles_labels()
        # if len(handles) > 10:  # 如果标签太多，只显示部分
        #     plt.legend(handles=[handles[0]] + handles[1:6] + handles[-5:], 
        #               labels=[labels[0]] + labels[1:6] + labels[-5:], 
        #               loc='upper right', fontsize=10)
        # else:
        #     plt.legend(loc='upper right', fontsize=10)

        filename = f"{model_name}_{filename_suffix}_top5_last5_{suffix}_vs_group0.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved plot to {filepath}")

    # 绘制图表
    plot(df1, opt_name)

def plot_accuracy(csv1, csv2, output_dir):
    # Load data
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # # 1. Loss curves for columns ending with 'r-100_train'
    # loss_cols1 = set(find_columns(df1, 'r-100_train'))
    # loss_cols2 = set(find_columns(df2, 'r-100_train'))
    # common_loss = sorted(list(loss_cols1 & loss_cols2))
    # if common_loss:
    #     plot_series(df1, df2, common_loss, ['Adam', 'SGD'], 'Loss', 'Loss Convergence', output_dir)
    # else:
    #     print("No matching 'r-100_train' columns found.")

    # 2. Accuracy (acc) columns
    acc_cols1 = set(find_columns(df1, 'acc'))
    acc_cols2 = set(find_columns(df2, 'acc'))
    common_acc = sorted(list(acc_cols1 & acc_cols2))
    if common_acc:
        plot_series(df1, df2, common_acc, ['Adam', 'SGD'], 'Accuracy', 'Accuracy Curve', output_dir)
    else:
        print("No matching 'acc' columns found.")

def plot_loss_and_accuracy(csv1, output_dir,model_name,opt_name):
    # Load data
    df1 = pd.read_csv(csv1)
    #df2 = pd.read_csv(csv2)

    os.makedirs(output_dir, exist_ok=True)

    # Get base name without extension
    csv1_name = os.path.splitext(os.path.basename(csv1))[0]

    # Plot acc
    #plot_top5_last5(df1, df2, 'acc', 'Adam', 'SGD', 'Accuracy', output_dir, csv1_name)

    # Plot acc5
    #plot_top5_last5(df1, df2, 'acc5', 'Adam', 'SGD', 'Top-5 Accuracy', output_dir, csv1_name)

    # Plot loss (optional: uncomment if needed)
    plot_top5_last5(df1,  '_train', 'Adam', 'SGD', 'Loss', output_dir, csv1_name,model_name,opt_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot top-5 and last-5 class curves.')
    parser.add_argument('--csv1', default='log/select_result/adapolycifar_r10_resnet18_adam_batch256_200e_lr0.001_seed0.csv', help='Path to first CSV file')
    parser.add_argument('--csv2', default='log/select_result/adapolycifar_r10_resnet18_sgd_batch256_200e_lr0.5_seed0.csv', help='Path to second CSV file')
    parser.add_argument('--output_dir', default='plots/acc_dg', help='Directory to save output figures')
    args = parser.parse_args()
    
    best_lrs =  {
    "resnet18_cifar_r10": {"sgd": 1.0,  "adam": 0.001,  "adam_bn": 0.0005, "adam_ini": 0.05},
    "resnet50_cifar_r10": {"sgd": 1.0,  "adam": 0.001,  "adam_bn": 0.0001, "adam_ini": 0.05},
    "vgg16bn_cifar_r10":  {"sgd": 0.1,  "adam": 0.0005, "adam_bn": 1e-5,   "adam_ini": 0.01},
    "vitb_cifar_r10":     {"sgd": 0.05, "adam": 5e-5,   "adam_bn": 5e-6,  "adam_ini": 0.005},
    "vits_cifar_r10":     {"sgd": 0.05, "adam": 5e-5,   "adam_bn": 1e-5,  "adam_ini": 0.005},
    }
    
    for model_name, optim_dict in best_lrs.items():
        for optimizer, lr in optim_dict.items():
            model_short = model_name.split("_")[0]  # 如 resnet18、vgg、vitb
            csv_path = f"log/adapolycifar_r10_{model_short}_{optimizer}_batch256_200e_lr{lr}_seed0.csv"
            plot_loss_and_accuracy(csv_path, args.output_dir, model_name, optimizer)



    
