import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties, fontManager
import matplotlib.ticker as ticker


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    df_datas = pd.read_csv('/home/hy/nn_property/submit/performance_pareto_cifar10.csv', index_col=0)
    num=8    
    column_name = df_datas.columns.tolist()
    y = df_datas[column_name[0]]
    x = df_datas[column_name[1]]
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 3)
    ax.plot(x, y, marker='.')
    plt.ylabel('Task (↑)', fontsize=24)#(↓) Active defense
    plt.xlabel('Privacy (↓)', fontsize=24)#Unknown
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(55,20)
    plt.ylim(80,100)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.grid(True)
    fig.tight_layout()
    plt.savefig('/home/hy/nn_property/submit/performance_pareto_cifar10.png', dpi=400,bbox_inches = 'tight')