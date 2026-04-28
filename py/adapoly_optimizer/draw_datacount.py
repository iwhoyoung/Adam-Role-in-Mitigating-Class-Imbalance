# import torch
# from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
#from cifar100_1class import CIFAR100LT  # 替换成你定义 CIFAR100LT 的文件名
import argparse
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_class_distribution(dataset, output_path='class_distribution.png'):
    # 统计每个类别的样本数
    counter = Counter(dataset.targets)

    # 按样本数从高到低排序
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_items)

    # 生成颜色渐变（和 viridis 一致，从深色到浅色）
    num_classes = len(classes)
    colors = cm.viridis(np.linspace(0, 1, num_classes))  # 倒序，让高→低是深→浅

    # 绘图
    plt.figure(figsize=(5, 4))
    for i in range(num_classes - 1):
        plt.plot([i, i+1], [counts[i], counts[i+1]], color=colors[i], linewidth=2)

    plt.title('Samples/Class', fontsize=18)
    plt.xlabel('Class Index', fontsize=16)
    plt.ylabel('Samples', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved class distribution plot to {output_path}")


def plot_colorbar_only(min_val, max_val, output_path='/home/wangjzh/adam_optimizer/plots/cifar100_loss/colorbar_only.png'):
    # 设置颜色映射
    cmap = cm.viridis.reversed()
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    

    # 创建虚拟的 ScalarMappable，仅用于 colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # 创建颜色条图像
    fig, ax = plt.subplots(figsize=(1.3, 6))  # 纵向瘦条状
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    cbar.set_label('Samples/Class', fontsize=18)
    cbar.ax.tick_params(labelsize=16) 

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved colorbar-only plot to {output_path}")
   

if __name__ == '__main__':

    # dataset = CIFAR100LT(
    #     root='/home/wangjzh/adam_optimizer/data/cifar100_lt_outputs/cifar100-lt-r-10',
    #     version='r-10',
    #     train=True,
    #     transform=None
    # )

    #plot_class_distribution(dataset, output_path='/home/wangjzh/adam_optimizer/plots/cifar100_loss/class_distribution.png')
    plot_colorbar_only(0, 500)
