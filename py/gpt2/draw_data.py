import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import matplotlib.cm as cm
from matplotlib.ticker import LogLocator, NullLocator

def plot_token_frequency_groups(csv_path, output_path="plots/token_frequency_groups.png"):
    # 从CSV文件读取数据
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    # 读取并按频率降序排序
    df = pd.read_csv(csv_path)
    df = df.sort_values("frequency", ascending=False).reset_index(drop=True)
    
    # 准备基础数据
    freqs = df["frequency"].values
    ranks = np.arange(1, len(freqs) + 1)
    total_frequency = freqs.sum()
    target_group_sum = total_frequency / 10  # 每组的目标总频率
    
    # 按频率总和均等分为10类
    groups = np.zeros_like(ranks)
    current_sum = 0
    current_group = 0
    
    for i, freq in enumerate(freqs):
        current_sum += freq
        groups[i] = current_group
        
        if current_sum >= target_group_sum and current_group < 9:
            current_sum = 0
            current_group += 1
    
    # 为10个组准备颜色（使用viridis配色方案）
    cmap = cm.viridis
    group_colors = cmap(np.linspace(0, 1, 10))
    point_colors = [group_colors[int(group)] for group in groups]
    
    # 创建画布和坐标轴
    plt.figure(figsize=(10, 6))
    ax = plt.gca()  # 获取当前坐标轴对象
    
    # 绘制分组散点图
    plt.scatter(
        ranks, 
        freqs, 
        color=point_colors, 
        s=120, 
        edgecolors='none'
    )
    
    # 坐标轴设置 - 只保留10²和10⁴刻度，不显示更细致的刻度
    # x轴设置：仅显示10²(100)和10⁴(10000)
    ax.set_xscale('log')
    ax.set_xticks([10**2, 10**4])  # 只保留这两个刻度
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))  # 主要刻度定位器
    ax.xaxis.set_minor_locator(NullLocator())  # 禁用次要刻度
    
    # y轴设置：简化刻度，只保留主要数量级
    ax.set_yscale('log')
    max_freq = freqs.max()
    if max_freq >= 10**6:
        ax.set_yticks([10**0,10**3, 10**6])  # 当频率足够高时显示10³和10⁶
    else:
        ax.set_yticks([10**0,10**3])  # 否则只显示10³
    ax.yaxis.set_minor_locator(NullLocator())  # 禁用次要刻度
    
    # 坐标轴标签和字体设置
    plt.xlabel("Class", fontsize=40)
    plt.ylabel("Samples", fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 优化显示
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印每组的实际总频率
    for i in range(10):
        group_total = freqs[groups == i].sum()
        print(f"组 {i+1} 总频率: {group_total:.0f} (目标: {target_group_sum:.0f})")
    print(f"[✓] 已保存简化刻度的散点图至 {output_path}")

def main():
    csv_path = "token_frequency_all.csv"
    plot_token_frequency_groups(csv_path)

if __name__ == "__main__":
    main()
    