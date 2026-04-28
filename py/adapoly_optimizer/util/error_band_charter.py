import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties, fontManager
import matplotlib.ticker as ticker

    

if __name__ == '__main__':
    time = np.linspace(0,10,100)
    data = np.sin(time) + np.random.normal(0,0.1,len(time))
    # plt.rcParams['font.family'] = 'Times New Roman'
    df_datas = pd.read_csv('/home/hy/robust_model/submit/weight_cifar10_res18_sgd_200e_diffseed.csv', index_col=0)
    num=8    
    column_name = df_datas.columns.tolist()
    avg = df_datas[column_name[0]]
    std = df_datas[column_name[1]]
    x = []
    for i in range(1, len(avg)+1):
        x.append(i)
        # x.append(i.__str__())
    fig, ax = plt.subplots()
    ax.plot(x,avg,label='avg', marker='.')
    ax.fill_between(x, avg -2*std, avg +2*std, alpha=0.3,label='std')
    # plt.legend()
    plt.xlabel('Depth', fontsize=15)#(↓) Active defense
    plt.ylabel('Weight Difference', fontsize=15)#Unknown
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlim(0,25)
    plt.ylim(0,0.4)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    # ax.grid(True)
    plt.fill_between([0.5,21.5],0,0.4,facecolor='gray',alpha=0.1) 
    plt.fill_between([7.5,8.5],0,0.4,facecolor='green',alpha=0.1)# 7.5,8.5 14.5,15.5
    plt.fill_between([12.5,13.5],0,0.4,facecolor='green',alpha=0.1)# 12.5,13.5 24.5,25.5
    plt.fill_between([17.5,18.5],0,0.4,facecolor='green',alpha=0.1)# 17.5,18.5 34.5,35.5
    plt.fill_between([20.5,21.5],0,0.4,facecolor='blue',alpha=0.1) 
    fig.tight_layout()
    plt.savefig('/home/hy/robust_model/submit/weight_diff_errorband.png', dpi=400, bbox_inches = 'tight')