import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties, fontManager
import matplotlib.ticker as ticker


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # import numpy as np

    # layers = list(range(1,20)) + ["last"]
    # x_positions = np.arange(len(layers))

    # gradient = [0.737,0.083,0.081,0.071,0.071,0.042,0.066,0.073,0.048,
    #             0.057,0.053,0.025,0.032,0.026,0.023,0.015,0.028,0.023,
    #             0.016,1.625]

    # weight_diff = [0.544,0.030,0.116,0.081,0.074,0.006,0.113,0.063,0.006,
    #             0.099,0.046,0.009,0.079,0.051,0.027,0.003,0.083,0.047,
    #             0.003,0.537]

    # plt.figure(figsize=(12, 6))

    # # 绘制双纵轴
    # ax1 = plt.gca()
    # ax2 = ax1.twinx()

    # # 梯度曲线（左侧坐标轴）
    # ax1.plot(x_positions, gradient, 'b-o', label="Average Absolute Gradient")
    # ax1.set_ylabel("Average Absolute Gradient", color='b')
    # ax1.tick_params(axis='y', colors='b')
    # ax1.set_ylim(0, 0.3)

    # # 权重差异曲线（右侧坐标轴）
    # ax2.plot(x_positions, weight_diff, 'r-s', label="Weight Difference")
    # ax2.set_ylabel("Weight Difference", color='r')
    # ax2.tick_params(axis='y', colors='r')
    # ax2.set_ylim(0, 0.6)

    # # 通用设置
    # plt.title("Layer-wise Gradient and Weight Differences")
    # plt.xticks(x_positions, layers)
    # plt.xlabel("Network Layer")
    # plt.grid(True)

    # # 合并图例
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # plt.legend(lines + lines2, labels + labels2, loc="upper left")

    # # 突出显示last层
    # ax1.annotate('Last Layer', xy=(19, 1.625), xytext=(15, 1.4),
    #             arrowprops=dict(facecolor='black', shrink=0.05))

    # plt.tight_layout()
    # plt.savefig('/lichenghao/huY/adapoly_optimizer/submit/1.png', dpi=400, bbox_inches = 'tight')


    plt.rcParams['font.family'] = 'Times New Roman'
    df_datas = pd.read_csv('/home/hy/wsol/submit/acc_gen_ours.csv', index_col=0)
    num=8    
    column_name = df_datas.columns.tolist()
    method = df_datas[column_name[0]]
    bubble_size = df_datas[column_name[1]]
    x = df_datas[column_name[2]]
    y = df_datas[column_name[3]]

    min_bubble_size = min(bubble_size)
    max_bubble_size = max(bubble_size)
    norm_bubble_size=[]
    close=[]
    for i in range(num):
        norm_bubble_size.append((7*(bubble_size[i]-50)/(max_bubble_size-50))**3)
    for i in range(num):
        # close.append((0.003*(num-1-i)/(num-1)))
        close.append((0.003*(i)/(num-1)))
    
    
    fig, ax = plt.subplots()
    # last_x = x.pop(num-1)
    # last_y = y.pop(num-1)
    ax.scatter(x, y, c=close, s=norm_bubble_size, alpha=1,cmap='turbo')
    # for i in range(8):      
        # plt.plot(x[i], y[i], marker='o', alpha=0.3, MarkerBorderWidth=2, ms=norm_bubble_size[i])
    for i in range(4):
        # dis=1
        # for j in range(4):
        #     if abs(y[i] - y[j]) < abs(dis):
        #         dis = y[i] - y[j]
        #     if dis < 0:
        #         rect_y = y[i] - 1 + abs(y[i] - y[j])
        #     else:
        #         rect_y = y[i] + 1 - abs(y[i] - y[j])
        plt.text(x[i]-0.82*len(method[i]), y[i]-0.6*(3-i)%2, method[i],fontsize=18)#,fontsize=8
        # plt.text(x[i]-0.87*len(method[i]), y[i], method[i],fontsize=20)#,fontsize=8 #0.87 0.55 0.82 0.87
        # plt.text(x[i]-0.3*len(method[i]), y[i], method[i])#,fontsize=8
    for i in range(4,num):
        plt.text(x[i]+0.65, y[i], method[i],fontsize=20)#0.7 0.52 0.65 0.7
        # plt.text(x[i]+0.5, y[i], method[i])#lfw
        # ax.annotate(method[i], (x[i], y[i]))
    # plt.text(x[num-1]+0.52, y[num-1], method[num-1],fontsize=24)#,color='#A52A2A' 0.7 0.52
    plt.xlabel('Privacy 1 (↓)', fontsize=24)#(↓) Active defense
    plt.ylabel('Privacy 2 (↓)', fontsize=24)#Unknown
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.xlim(53,74)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.grid(True)
    fig.tight_layout()
    plt.savefig('/home/hy/wsol/submit/acc_gen_ours.png', dpi=400, bbox_inches = 'tight')