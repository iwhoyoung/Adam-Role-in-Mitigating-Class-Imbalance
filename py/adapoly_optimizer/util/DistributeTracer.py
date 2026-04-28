
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class DistributeTracer:
    def __init__(self, resolution=None, span=None, bins=10):
        self.resolution = resolution
        self.span = span
        self.bins = bins

    def hist(self, input_data, input_class="dataframe", column='0', label=None, alpha=1., fitting_curve_graph=False, norm=True):
        if input_class == "file":
            input_data = pd.read_csv(input_data)
        else:
            input_data = input_data
        data = input_data[column]
        if norm:
            mu = np.mean(data)
            data = data - mu
            # sigma = np.std(data)
            # data = data / sigma
        if self.resolution is not None and self.span is not None:
            span = self.span[1]-self.span[0]
            bins = int(span/self.resolution)
            n, bins, patches = plt.hist(data, bins=bins, range=self.span, label=label, density=True, alpha=alpha)
        else:
            n, bins, patches = plt.hist(data, bins=self.bins, label=label, alpha=alpha)
        if fitting_curve_graph:
            mu = np.mean(data)
            sigma = np.std(data)
            # 正态分布拟合曲线
            y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                 np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
            plt.plot(bins, y, '--',  label=label)  # 'r--' represents color red and dashed line
        if label is not None:
            plt.legend(loc='upper right')
        pass

    def scatter(self, input_data, input_class="dataframe", x_column='0', y_column='1', label=None, alpha=1., norm=True, ticks=False):
        if input_class == "file":
            input_data = pd.read_csv(input_data)
        else:
            input_data = input_data
        x = input_data[x_column]
        y = input_data[y_column]
        if norm:
            mu_x = np.mean(x)
            mu_y = np.mean(y)
            x = x - mu_x
            y = y - mu_y
            sigma_x = np.std(x)
            sigma_y = np.std(y)
            if sigma_x > sigma_y:
                sigma = sigma_x
            else:
                sigma = sigma_y
            x = x / sigma
            y = y / sigma
        plt.xlim(-7.5, 7.5)
        plt.ylim(-7.5, 7.5)
        if not ticks:
            plt.xticks([])
            plt.yticks([])
            # plt.axis('off')
        # #AFAFAF-origin #E78037,#99CE9A,#EFADAE-ours，darkcyan, brown-latest
        plt.scatter(x, y, s=5, color='brown', label=label, alpha=alpha)
        # plt.scatter(x, y, s=5, label=label, alpha=alpha)
        if label is not None:
            plt.legend(loc='upper right')
        pass

    def show(self):
        plt.show()


# demo
if __name__ == '__main__':
    # tracer = DistributeTracer(resolution=0.05, span=(-5, 5))
    tracer = DistributeTracer(bins=100)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10m264t2.csv', input_class="file", column='3', alpha=0.2, label="ours_t=2", curve_graph=True)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10m264t2_no_norm.csv', input_class="file", column='1', alpha=0.3, label="ours_t=2", curve_graph=True)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10m264t5.csv', input_class="file", column='1', alpha=0.3, label="t=5", curve_graph=False)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10m264t10.csv', input_class="file", column='1', alpha=0.3, label="t=10", curve_graph=False)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10m264t50.csv', input_class="file", column='1', alpha=0.3, label="t=50", curve_graph=False)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10m264t80.csv', input_class="file", column='3', alpha=0.2, label="ours_t=80", curve_graph=False)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10m264t80_no_norm.csv', input_class="file", column='1', alpha=0.3, label="ours_t=80", curve_graph=False)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t10.csv', input_class="file", column='1', alpha=0.3, label="com_t=10", curve_graph=True)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t1.csv', input_class="file", column='1', alpha=0.3, label="com_t=1", curve_graph=False)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t1_no_norm.csv', input_class="file", column='3', alpha=0.3, label="com_t=1", curve_graph=False)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t.5.csv', input_class="file", column='3', alpha=0.3, label="com_t=0.5", curve_graph=False)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t.1.csv', input_class="file", column='3', alpha=0.3, label="com_t=0.1", curve_graph=False)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t.05.csv', input_class="file", column='2', alpha=0.3, label="com_t=0.05", curve_graph=True)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t.03.csv', input_class="file", column='1', alpha=0.3, label="com_t=0.03", curve_graph=False)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t.03_no_norm.csv', input_class="file", column='3', alpha=0.3, label="com_t=0.03", curve_graph=False)

    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t10.csv', input_class="file", column='6', alpha=0.3,
    #             label="com_t=10", curve_graph=True)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t10.csv', input_class="file", column='7', alpha=0.3,
    #             label="com_t=10", curve_graph=True)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t10.csv', input_class="file", column='8', alpha=0.3,
    #             label="com_t=10", curve_graph=True)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t10.csv', input_class="file", column='9', alpha=0.3,
    #             label="com_t=10", curve_graph=True)
    # tracer.hist('../asset/btl/discriptor_distribution/Cifar10sim64t10.csv', input_class="file", column='10', alpha=0.3,
    #             label="com_t=10", curve_graph=True)

    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar102orig.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="untrained")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t1add101.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=1")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t5.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=5")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t10.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=10")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t15add1.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=15")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t5add1eps6.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=15")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t47.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=47")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t30.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=30")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t35.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=35")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t38.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=38")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t40.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=40")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t45.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=45")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t50.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=50")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t55.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=55")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t60.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=60")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t100.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=100")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t200.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=200")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t500.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="ours_t=500")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10sim2t1.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="com_t=1")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10sim2t.5.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="com_t=0.5")
    tracer.scatter('../asset/btl/discriptor_distribution/Cifar10m22t500.csv', input_class="file", x_column='0', y_column='1', alpha=0.3)
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10sim2t.1.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="com_t=0.1")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10sim2t.05.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="com_t=0.05")
    # tracer.scatter('../asset/btl/discriptor_distribution/Cifar10sim2t.03.csv', input_class="file", x_column='0', y_column='1', alpha=0.3, label="com_t=0.03")

    tracer.show()
    pass