import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np
import os
from collections import defaultdict  # 补充缺失的导入

def smooth_curve(values, window_size=500):
    return pd.Series(values).rolling(window=window_size, center=True, min_periods=1).mean().values

def preprocess_group_loss(df, group_cols):
    """补全缺失值（线性插值 + ffill + bfill）"""
    for col in group_cols:
        df[col] = (
            df[col]
            .interpolate(method="linear", limit_direction="both")
            .fillna(method="bfill")
            .fillna(method="ffill")
        )
    return df

def count_avg_token_frequency_per_group(token_to_group, token_freq):
    # 用于统计每个 group 的 token 总频次和 token 数量
    group_token_sum = defaultdict(int)
    group_token_count = defaultdict(int)

    for token, freq in token_freq.items():
        group = token_to_group[token]
        group_token_sum[group] += freq
        group_token_count[group] += 1

    # 计算每个 group 内平均每个 token 的频率
    avg_freq = {
        group: group_token_sum[group] / group_token_count[group]
        for group in group_token_sum
    }

    # 转为 DataFrame 保存
    df = pd.DataFrame({
        "group": list(avg_freq.keys()),
        "avg_token_frequency": list(avg_freq.values()),
        "token_count": [group_token_count[g] for g in avg_freq],
        "total_token_frequency": [group_token_sum[g] for g in avg_freq]
    }).sort_values(by="group")

    df.to_csv("token_avg_frequency_per_group.csv", index=False)
    return avg_freq

def plot_group_loss(df, group_cols, steps, output_path, y_range=None):
    plt.figure(figsize=(10, 6))

    # 提取 group index（如 group_0、group_1）
    class_indices = [int(col.split("_")[1]) for col in group_cols]
    print(class_indices)
    max_class_idx = max(class_indices + [1])  # 避免除以0
    colors = [cm.viridis(idx / (max_class_idx + 1)) for idx in class_indices]

    for idx, col in enumerate(group_cols):
        #smoothed = smooth_curve(df[col].values, window_size=500)
        print(df[col].isna().sum())
        print(df[col].describe())

        plt.plot(steps, df[col], color=colors[idx], linewidth=2)

    ax = plt.gca()

    # 横轴格式为 1k, 2k 等
    def format_k(x, pos):
        return f'{int(x / 50)}k' if x >= 50 else str(int(x))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_k))

    # 设置 y 轴范围（可选）
    if y_range is not None:
        plt.ylim(*y_range)

    # 坐标轴线条样式
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6)

    # 字体大小统一
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel("Step", fontsize=40)
    plt.ylabel("Train Loss", fontsize=40)

    # 网格样式统一
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存图像
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[✓] Saved plot to {output_path}")

def generate_plot_from_csv(csv_path, optimizer_name, output_dir):
    """主函数，读取 CSV、预处理、绘图"""
    df = pd.read_csv(csv_path)
    max_value = df["group_0"].max()

    # 创建布尔 Series：哪些行的 group_0 等于最大值
    rows_with_max = df["group_0"] == max_value
    
    # 返回所有包含最大值的行
    result = df.loc[rows_with_max]  # 如果你只想看 group_0 列，可以加个 ["group_0"]

    group_cols = sorted([col for col in df.columns if col.startswith("group_")], key=lambda x: int(x.split("_")[1]))
    df = preprocess_group_loss(df, group_cols)
    steps = df["step"]
    output_path = os.path.join(output_dir, f"{optimizer_name.lower()}_group_loss_interp_style.png")
    plot_group_loss(df, group_cols, steps, output_path, y_range=(0, 8))  # 设置y轴范围

if __name__ == "__main__":
    # 自定义路径：根据你的文件名修改
    adam_csv = "group_loss_adam_lr0.0001.csv"
    sgd_csv = "grid_search_results/group_loss_sgd_lr0.01.csv"
    output_dir = "plots"

    # 一键生成
    generate_plot_from_csv(adam_csv, "Adam", output_dir)
    generate_plot_from_csv(sgd_csv, "SGD", output_dir)