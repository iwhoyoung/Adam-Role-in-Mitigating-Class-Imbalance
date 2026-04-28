import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties, fontManager

if __name__ == '__main__':
    
    # 准备数据
    labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    values = [10, 20, 30, 25, 15]
    
    fig, ax = plt.subplots()
    # 创建柱状图
    ax.bar(labels, values)
        
    # 添加一些文本以标注条形图
    ax.set_xlabel('Model', fontsize=15)
    ax.set_ylabel('Neuron Activation Similarity', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # ax.set_title('Grouped Bar Chart')
    ax.set_ylim(0,1)
    # ax.set_xticklabels(['VGG16', 'Res18', 'Res50', 'Swin', 'ViT-S', 'ViT-B'])  # 设置x轴标签
    ax.legend(fontsize=15)
    fig.set_size_inches(6, 4)
    # 显示图表
    plt.tight_layout()

    plt.savefig('/lichenghao/huY/ada_optimizer/submit/long_tail_data_bin_chart.png', dpi=400, bbox_inches = 'tight')