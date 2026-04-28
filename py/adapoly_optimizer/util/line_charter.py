import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties, fontManager

if __name__ == '__main__':
    # font_times = FontProperties(fname=r'C:\Windows\Fonts\Times New Roman.ttc')
    x_data = [5, 15, 30, 50, 80, 120, 160, 200]
    # x_data = [32, 64, 128, 256]
    y_data = [50.13, 29.07, 20.47, 17.77, 18.87, 18.62, 21.30, 21.94]
    # y_data = [59.71, 29.78, 21.80, 20.54, 19.19, 18.62, 21.57, 23.79]
    # y_data1 = [59.71, 29.78, 21.80, 20.54, 19.19, 18.62, 21.38, 23.79]
    # x_data = [1, 0.5, 0.1, 0.05, 0.03]
    # y_data = [43.27, 39.78, 26.26, 20.33, 19.45]
    # x_data = ['32', '64', '128', '256', '512']
    # y_data = [33.85, 26.84, 22.11, 18.93, 18.40]
    # blcd_real_data = [8.62, 8.62, 9.25, 8.46, 10.79]
    # blcd_bin_data = [9.72, 11.42, 13.74, 19.27, 27.08]
    # btl_data = [8.55, 10.60, 13.12, 18.82, 26.54]

    # fig = plt.figure()
    # ax = fig.add_axes([0, 0.1, 0.6, 0.75])
    # plt.plot(x_data, y_data, color='red', linewidth=2.0, linestyle='--')
    # plt.plot(x_data, y_data2, color='blue', linewidth=3.0, linestyle='-.')
    # plt.show()

    # Create a figure of size 8x6 inches, 80 dots per inch
    # plt.figure(figsize=(8, 6), dpi=80)

    # # Create a new subplot from a grid of 1x1
    # plt.subplot(1, 1, 1)
    #
    # # Set x limits
    # plt.xlim(-4.0, 4.0)
    # # Set x ticks
    # plt.xticks(np.linspace(-4, 4, 9, endpoint=True))
    #
    # # set ylimits
    # plt.ylim(15, 62)
    # # Set y ticks
    # plt.yticks(np.linspace(-1.2, 1.2, 5, endpoint=True))
    #
    # X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    # C, S = np.cos(X), np.sin(X)

    # marker_on = [0, 5, 10, 15]
    fig, ax = plt.subplots()

    # ax.plot([0.03, 0.022, 0], [19.45, 95, 95], ls='--', color='r', ms=4, linewidth=1.0, markevery=None)
    ax.plot(x_data, y_data, marker='o', ls='-', ms=4, linewidth=1.0, markevery=None)
    #
    plt.xticks([5, 15, 30, 50, 80, 120, 160], fontsize=20)
    # plt.xticks([0.02], [0.02])
    #
    # ax.plot(x_data, blcd_real_data, marker='o', ls='-', ms=4, linewidth=1.0, markevery=None, label='BLCD(real-valued)')
    # ax.plot(x_data, blcd_bin_data, marker='o', ls='-', ms=4, linewidth=1.0, markevery=None, label='BLCD(binary)')
    # ax.plot(x_data, btl_data, marker='o', ls='-', ms=4, linewidth=1.0, markevery=None, label='BLCD with BTL(binary)')

    # 设置数字标签
    for a, b in zip(x_data, y_data):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=20)
    # for a, b in zip(x_data, blcd_real_data):
    #     plt.text(a, b, b, ha='center', va='bottom', color='dodgerblue', fontsize=10)
    # for a, b in zip(x_data, blcd_bin_data):
    #     plt.text(a, b, b, ha='center', va='bottom', color='orange', fontsize=10)
    # for a, b in zip(x_data, btl_data):
    #     plt.text(a, b, b, ha='center', va='top', color='green', fontsize=10)
    # plt.xlim(0, 180)
    plt.ylim(15, 62)
    # plt.xscale("log")
    # plt.gca().invert_xaxis()
    # plt.xticks([1, 0.5, 0.1, 0.05, 0.03, 0.02], [1, 0.5, 0.1, 0.05, 0.03, 0.02], fontsize=20)
    plt.yticks(fontsize=20)
    #, title=''
    # ax.set(xlabel='Hyper-paramter $\\tau$', ylabel='FPR@95 (%)')
    ax.set_xlabel('Hyper-paramter $\eta$', fontsize=20)
    ax.set_ylabel('FPR@95 (%)', fontsize=20)
    # ax.set(xlabel='dimension(bits)', ylabel='FPR@95 (%)')
    ax.grid(linestyle='--')
    # plt.legend(loc='upper left')
    # plt.title('FPR@95 of different dimension descriptor on Phototour ')
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    #, title=''
    # Plot cosine with a blue continuous line of width 1 (pixels) linestyle=':', color="#fd3c06", marker='*', label="cosine"
    # plt.plot(x_data, y_data, marker='o', ls='-', ms=4, linewidth=1.0, markevery=None)
    # plt.plot(X,C)

    # Plot sine with a green continuous line of width 1 (pixels)
    # plt.plot(X, S, color="green", linewidth=1.0, linestyle="--", marker='s', markevery=marker_on, label="sine")
    # Adding a legend
    # plt.legend(loc=5)

    # Save figure using 100dots per inch
    # plt.savefig("try.jpg", dpi=100, bbox_inches='tight')
    # Show result on screen
    plt.show()