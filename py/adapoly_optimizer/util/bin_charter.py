import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    group1 = [0.5827836066303831, 0.639408766127596, 0.6918993588047798, 0.5716548288470567, 0.5978910175236789]
    groupn = [
        [0.1770229222030952, 0.3011573678014254, 0.3294078626777186, 0.4465792657570405, 0.4432825681236055],
        [0.1263205921251063, 0.2595615250113035, 0.3501571517400067, 0.4519483882610244, 0.4900724210100944]
    ]

    labels = ['Initial Model', 'Trained Model (SGD)', 'Trained Model (Adam)']
    markers = ['.', 'x', '^']
    colors = ['gray', 'blue', 'orange']
    r1 = np.arange(len(group1))

    fig, ax = plt.subplots()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # 绘制初始模型
    ax.plot(r1, group1, ls='-', marker=markers[0], color=colors[0], label=labels[0])

    # 绘制其他模型
    for i in range(len(groupn)):
        ax.plot(r1, groupn[i], ls='--', marker=markers[i + 1], color=colors[i + 1], label=labels[i + 1])

    # 设定坐标轴与标签
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Gradient Orthogonality', fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_xticks(r1)
    ax.set_xticklabels(['VGG16', 'ResNet18', 'ResNet50', 'ViT-S', 'ViT-B'])

    # 图例与布局
    ax.legend(fontsize=12, loc='lower right')
    fig.set_size_inches(6, 3)
    plt.tight_layout()

    # 保存图像
    plt.savefig('/home/wangjzh/adam_optimizer/plots/longtail_bin_chartv2_nosecondary.png', dpi=400, bbox_inches='tight')
