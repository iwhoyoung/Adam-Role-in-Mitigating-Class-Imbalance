import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties, fontManager
import matplotlib.ticker as ticker




if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    df_datas = pd.read_csv('/home/wangjzh/adam_optimizer/log/grad/merged_polysemanticity_adam.csv', index_col=0).T.iloc[:-1]
    data = df_datas.transpose().to_numpy()[:,:-1]
    # 创建一个热力图
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data, cmap='OrRd_r')
    xticks = np.linspace(-0.5, 8.5, 10)
    # yticks = np.linspace(-0.5, 4.5, 6)
    plt.xticks(xticks)
    ax.set_yticklabels(['', 'VGG16', 'ResNet18', 'ResNet50',  'ViT-S', 'ViT-B'])
    ax.set_xticklabels(['0','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    cbar = fig.colorbar(heatmap,fraction=0.04,pad=0.04,shrink=0.3)
    cbar.ax.set_ylabel('Frequency', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    for i in [-0.5,0.5,1.5,2.5,3.5]:
        plt.axhline(y=i, color='w', linestyle='-', linewidth=3)
    ax.grid(color='w', linestyle='-', linewidth=3,axis='x')
    plt.xlabel('Gradient Orthogonality', fontsize=16)#Differential
    fig.tight_layout()
    plt.savefig('/home/wangjzh/adam_optimizer/plots/heatmap_origin_diff.png', dpi=400, bbox_inches = 'tight')