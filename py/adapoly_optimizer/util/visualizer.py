import logging
import os
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
# 设置宋体为默认字体

class Visualizer:
    """
    A tool to visualize training progress
    Usage:
        logger = visualizer.Visualizer(['train loss', 'ssim', 'Qabf'], ['r-o', 'b-o', 'y-o'], './log.txt')
        ...training code...
        logger.write_log([loss, ssim, qabf])
        logger.plot_loss()
    """

    def __init__(self, line_name: list, cut_num: int, line_color: list = None, log_path: str = None, is_log=False, name="log", label="", first_epoch_index=1):
        """
        :param pattern: name of loss
        :param log_path: log file name and path
        :param line_color: param to set line type when drawing, len(line_color) should be larger than len(pattern)
        """
        # assert len(pattern) <= len(line_color), 'line_color should contains more elements'
        self.name = name
        self.line_name = line_name
        self.cut_num = cut_num
        self.log_path = log_path
        self.line_color = line_color
        self.label = label
        self.first_epoch_index = first_epoch_index
        self.loss_curve = []
        if is_log:
            self.init_logger()
        # plt.ion()
        self.fig, self.ax_1 = plt.subplots()
        if cut_num != len(line_name):
            self.ax_2 = self.ax_1.twinx()
        pass

    def init_logger(self):
        if self.log_path is None:
            return
        # logging.basicConfig(filename=os.path.join(self.log_path, self.name+'.log'), level=logging.DEBUG,
        #                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        # logging.info('---LOG START---\n')
        f = open(self.log_path + self.name +".txt", 'a+')
        f.write('---LOG AT %s---\n' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.close()

    def record(self, loss: list):
        """
        Pass in loss data which needs to be manipulated.
        :param loss: the loss data which needs to be manipulated and it has to be list.
        :return:none
        """
        self.loss_curve.append(loss)

    def log(self):
        """
        print the current loss.
        :return: none
        """
        loss = self.loss_curve[-1]
        log_item = [self.line_name[i] + ': %.4f, ' % loss[i] for i in range(0, len(loss))]
        #print(self.label+' epoch: %03d, ' % len(self.loss_curve) + ''.join(log_item))
        pass

    def write_log(self):
        """
        write the current loss into file.
        :return: none
        """
        if self.log_path is None:
            return
        loss = self.loss_curve[-1]
        log_item = [self.line_name[i] + ': %.4f, ' % loss[i] for i in range(0, len(loss))]
        # log_item = ['%.4f, ' % loss[i] for i in range(0, len(loss))]
        # logging.info('epoch: %03d, ' % len(self.loss_curve) + ''.join(log_item) + '\n')
        f = open(self.log_path + self.name +".txt", 'a+')
        f.write(self.label+', epoch:, %03d, ' % (len(self.loss_curve)-1+self.first_epoch_index) + ''.join(log_item) + '\n')
        f.close()

    def write_other_log(self, content: str):
        """
        write the other info you want to write into file.
        :param content: the content you want to write into file.
        :return:none
        """
        if self.log_path is None:
            return
        # logging.info(content + '\n')
        f = open(self.log_path + self.name + ".txt", 'a+')
        f.write(content + '\n')
        f.close()

    def save_data(self):
        df_datas = pd.DataFrame(data=self.loss_curve, columns=self.line_name)
        df_datas.to_csv(self.log_path + self.label + '.csv')

    def plot_saved_data(self, path, is_show=False):
        """
        save the figure.
        :param path: the path to save the data
        :param is_show: whether show the figure or not.
        :return:none.
        """
        plt.cla()
        lines = []
        label = ['', '']
        ls = ['-', '--', '-.', ':']
        df_datas = pd.read_csv(path, index_col=0)
        for idx, curve_name in enumerate(df_datas.columns):
            loss = df_datas[curve_name]
            # loss = [item for item in df_datas[:, idx]]
            if idx < self.cut_num:
                if self.line_color is None:
                    line, = self.ax_1.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), loss, ls=ls[0])
                else:
                    line, = self.ax_1.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), loss, self.line_color[idx], ls=ls[0])
                label[0] = label[0] + curve_name + ' / '
            else:
                if self.line_color is None:
                    line, = self.ax_2.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), loss, ls=ls[1])
                else:
                    line, = self.ax_2.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), loss, self.line_color[idx], ls=ls[1])
                label[1] = label[1] + curve_name + ' / '
            lines.append(line)
        plt.legend(lines, self.line_name, loc='upper left')
        plt.xlabel('Epoch')
        self.ax_1.set_ylabel(label[0][0: -2])
        if self.cut_num != len(self.line_name):
            self.ax_2.set_ylabel(label[1][0: -2])
        plt.draw()
        plt.savefig(self.log_path + self.label + '.png')
        if is_show:
            plt.show()

    def plot_saved_data_w_xaxis(self, path, x_title, y_title, x_ticks=None, x_ticks_label=None, is_show=False):
        """
        save the figure.
        :param path: the path to save the data
        :param is_show: whether show the figure or not.
        :return:none.
        """
        plt.cla()
        # plt.rcParams.update({'font.size': 15})
        lines = []
        label = ['', '']
        ls = ['-', '--', '-.', ':']
        df_datas = pd.read_csv(path, index_col=0).T
        for idx, curve_name in enumerate(df_datas.columns):
            loss = df_datas[curve_name]
            # loss = [item for item in df_datas[:, idx]]
            if idx < self.cut_num:
                if self.line_color is None:
                    # line, = self.ax_1.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index))[0:200], loss[0:200], ls=ls[0],alpha=0.5)#, marker='.'
                    line, = self.ax_1.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), loss, ls=ls[0], marker='.')
                else:
                    line, = self.ax_1.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), loss, self.line_color[idx], ls=ls[0])
                label[0] = label[0] + str(curve_name) + ' / '
            else:
                if self.line_color is None:
                    line, = self.ax_2.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), loss, ls=ls[1], marker='.')
                else:
                    line, = self.ax_2.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), loss, self.line_color[idx], ls=ls[1])
                label[1] = label[1] + str(curve_name) + ' / '
            lines.append(line)
        # plt.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), [84.49,84.49,84.49,84.49,84.49],lines[0]._color, ls=ls[1])
        # plt.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), [76.96,76.96,76.96,76.96,76.96],lines[1]._color, ls=ls[1])
        # plt.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), [72.42,72.42,72.42,72.42,72.42],lines[2]._color, ls=ls[1])
        plt.legend(lines, self.line_name, loc='upper right',fontsize=14)#,ncol=2 lower upper left right
        # plt.xlabel(x_title, fontsize=15)
        self.ax_1.set_xlabel(x_title, fontdict={'fontsize': 14})
        plt.xticks(x_ticks,x_ticks_label, fontsize=14)
        plt.yticks(fontsize=14)
        # plt.xlim(0,22)
        # self.ax_1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        # self.ax_1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        # plt.ylim(0,1)
        # for i in range(41):
        #     if i % 2 == 1:
        #         plt.fill_between([i+0.5,i+1.5],0,7,facecolor='gray',alpha=0.1)
        # for i in range(4, 152):
        #     if (i-4) % 12 == 0:
        #         plt.fill_between([i+0.5,i+6.5],0,7,facecolor='gray',alpha=0.1)
        #     if (i-10) % 12 == 0:
        #         plt.fill_between([i+0.5,i+6.5],0,7,facecolor='blue',alpha=0.1)    
        # for i in range(4, 52):
        #     if (i-4) % 4 == 0:
        #         plt.fill_between([i+0.5,i+2.5],0,7,facecolor='gray',alpha=0.1)
        #     if (i-6) % 4 == 0:
        #         plt.fill_between([i+0.5,i+2.5],0,7,facecolor='blue',alpha=0.1)   
        
        # # vgg16
        # plt.fill_between([0.5,16.5],0,1,facecolor='gray',alpha=0.1) 
        # plt.fill_between([13.5,16.5],0,1,facecolor='blue',alpha=0.1) 
        # # resnet18
        # plt.fill_between([0.5,21.5],0,1,facecolor='gray',alpha=0.1) 
        # plt.fill_between([7.5,8.5],0,1,facecolor='green',alpha=0.1)# 7.5,8.5 14.5,15.5
        # plt.fill_between([12.5,13.5],0,1,facecolor='green',alpha=0.1)# 12.5,13.5 24.5,25.5
        # plt.fill_between([17.5,18.5],0,1,facecolor='green',alpha=0.1)# 17.5,18.5 34.5,35.5
        # plt.fill_between([20.5,21.5],0,1,facecolor='blue',alpha=0.1) 
        # # resnet50
        # plt.fill_between([0.5,54.5],0,1,facecolor='gray',alpha=0.1) 
        # plt.fill_between([4.5,5.5],0,1,facecolor='green',alpha=0.1)# 7.5,8.5 14.5,15.5
        # plt.fill_between([14.5,15.5],0,1,facecolor='green',alpha=0.1)# 12.5,13.5 24.5,25.5
        # plt.fill_between([27.5,28.5],0,1,facecolor='green',alpha=0.1)# 17.5,18.5 34.5,35.5
        # plt.fill_between([46.5,47.5],0,1,facecolor='green',alpha=0.1)# 17.5,18.5 34.5,35.5
        # plt.fill_between([53.5,54.5],0,1,facecolor='blue',alpha=0.1) 
        # # ViT
        # for i in range(49):
        #     if i==0:
        #         continue
        #     if (i-1)%4==0:
        #         plt.fill_between([i+0.5,i+2.5],0,1,facecolor='orange',alpha=0.1)
        #     elif (i-1)%4==2:
        #         plt.fill_between([i+0.5,i+2.5],0,1,facecolor='blue',alpha=0.1)
        # plt.fill_between([49.5,50.5],0,1,facecolor='blue',alpha=0.1)
                
        # plt.fill_between([13.5,16.5],0,7,facecolor='blue',alpha=0.1) 
        # plt.fill_between([26.5,32.5],0,7,facecolor='blue',alpha=0.1) 
        # plt.fill_between([14.5,15.5],0,7,facecolor='green',alpha=0.1)# 7.5,8.5 14.5,15.5
        # plt.fill_between([24.5,25.5],0,7,facecolor='green',alpha=0.1)# 12.5,13.5 24.5,25.5
        # plt.fill_between([34.5,35.5],0,7,facecolor='green',alpha=0.1)# 17.5,18.5 34.5,35.5
        # plt.fill_between([40.5,41.5],0,7,facecolor='blue',alpha=0.1) 
        
        plt.gcf().subplots_adjust(left=0.18,bottom=0.12)#0.12 0.16 0.18
        # plt.gcf().set_size_inches(10,5)
        if y_title is None:
            self.ax_1.set_ylabel(label[0][0: -2], fontdict={'fontsize': 14})
        else:
            self.ax_1.set_ylabel(y_title,fontsize=15)
        if self.cut_num != len(self.line_name):
            self.ax_2.set_ylabel(label[1][0: -2], labelpad=-385, fontdict={'fontsize': 14})
        self.fig.set_size_inches(6, 2.5)
        plt.draw()
        plt.savefig(self.log_path + self.label + '.png', dpi=400, bbox_inches = 'tight')# 
        if is_show:
            plt.show()

    def plot_loss(self, is_show=False):
        """
        save the figure.
        :param is_show: whether show the figure or not.
        :return:none.
        """
        plt.cla()
        lines = []
        label = ['', '']
        ls = ['-', '--', '-.', ':']

        for idx, curve_name in enumerate(self.line_name):
            loss = [item[idx] for item in self.loss_curve]
            if idx < self.cut_num:
                if self.line_color is None:
                    #range(1, len(loss) + 1).astype(dtype=np.str)
                    line, = self.ax_1.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), loss, ls=ls[0])
                else:
                    line, = self.ax_1.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), loss, self.line_color[idx], ls=ls[0])
                label[0] = label[0] + curve_name + ' / '
            else:
                if self.line_color is None:
                    line, = self.ax_2.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), loss, ls=ls[1])
                else:
                    line, = self.ax_2.plot(list(range(self.first_epoch_index, len(loss) + self.first_epoch_index)), loss, self.line_color[idx], ls=ls[1])
                label[1] = label[1] + curve_name + ' / '
            lines.append(line)
        plt.legend(lines, self.line_name, loc='upper left')
        plt.xlabel('Epoch')
        self.ax_1.set_ylabel(label[0][0: -2])
        if self.cut_num != len(self.line_name):
            self.ax_2.set_ylabel(label[1][0: -2])    
        plt.draw()
        plt.savefig(self.log_path + self.label + '.png')
        if is_show:
            plt.show()

# demo
if __name__ == '__main__':
    import random
    # logger = Visualizer(['ResNet18','ResNet18 w/o shortcut','ResNet18 w/o first 2 shortcut','ResNet18 w/o last 2 shortcut'], 4, log_path='/home/hy/wsol/submit/',name="init_acc_lr",label="diff_flower17_resnet18wpartres_sgd_batch64_200e_lr01_seed0")
    # logger = Visualizer(['ResNet18','ResNet18 w/o shortcut'], 2, log_path='/home/hy/wsol/submit/',name="init_acc_lr",label="dynamics_rate_flower17_resnet18origin_sgd_batch64_200e_lr01_seed0_mean")
    # logger = Visualizer(['Lr 10$^{-2}$','Lr 10$^{-3}$','Lr 10$^{-2}$ (pre.)','Lr 10$^{-3}$ (pre.)'], 4, log_path='/home/hy/wsol/submit/',name="init_acc_lr",label="fig_dif_flower17_adam_irrelevant_data_pretrain_pre25ecifar10")
    # logger = Visualizer(['CIFAR10','CIFAR100','Food101'], 3, log_path='/home/hy/wsol/log/',name="init_acc_lr",label="weight_diff_bn_vit_adassd1_200e")
    # logger = Visualizer(['Setup 0','Setup 1','Setup 2','Setup 3','AdaSSD 1','AdaSSD 2'], 6, log_path='/home/hy/wsol/log/',name="init_acc_lr",label="weight_diff_bn_vit_ablationadassd_200e")
    # logger = Visualizer(['SGD','Adam','AdaSSD 1','AdaSSD 1 w/ EMA'], 4, log_path='/home/hy/wsol/log/',name="init_acc_lr",label="weight_cifar10_resnet18_sgd_batch128_200e_lr05")
    # logger = Visualizer(['二值表征表征','实值表征表征'], 2, log_path='/home/hy/wsol/log/',name="init_acc_lr",label="weight_cifar10_resnet18_sgd_batch128_200e_lr05_zh")
    # logger = Visualizer(['ViT-S (lr=1e-5)','ViT-S (lr=5e-5)','ViT-S (lr=1e-4)'], 3, log_path='/lichenghao/huY/ada_optimizer/submit/',name="init_acc_lr",label="weight_diff_cifar10_vits4_rmsprop_difflr")
    logger = Visualizer(['ResNet18','ViT-S'], 2, log_path='/lichenghao/huY/ada_optimizer/submit/',name="init_acc_lr",label="hyperparameter_beta")
    # logger = Visualizer(['lr=0.01','lr=0.05','lr=0.1','lr=0.5','lr=1','lr=5','lr=10'], 7, log_path='/home/hy/gen_study/log/',name="init_acc_lr",label="lu_mean_cifar10_resnet18_he_sgd_batch64_seed0_nobia_log")
    # logger = Visualizer(['epoch 1','epoch 5','epoch 20','epoch 180'], 4, log_path='/home/hy/wsol/submit/',name="init_acc_lr",label="dynamics_individual_rate_std_flower17_resnet18wores_sgd_batch64_200e_lr01_seed0")
    # logger = Visualizer(['SGD','Adam'], 2, log_path='/home/hy/wsol/submit/',name="init_acc_lr",label="fig_dif_cifar10_resnet50v")
    # logger.plot_saved_data_w_xaxis('/home/hy/gen_study/log/fig2_acc_cifar10_resnet18_he_sgd_seed0.csv',x_title='Epoch',y_title='Accuracy')
    logger.plot_saved_data_w_xaxis('/lichenghao/huY/ada_optimizer/submit/hyperparameter_beta.csv',x_title='Beta',y_title='Accuracy', x_ticks=[1,2,3,4,5], x_ticks_label=['0.9','0.95','0.99','0.995','0.999'])

    # logger.plot_saved_data_w_xaxis('/home/LAB/lufh/hy/ada_optimizer/log/role_of_trans_num.csv',x_title='Transformation Number',y_title=None, x_ticks=[1,2,3,4,5], x_ticks_label=['1','2','4','8','12'])
    # , x_ticks=[1,2,3,4,5], x_ticks_label=['1','2','3','4','5']
    # logger.plot_saved_data_w_xaxis('/home/LAB/lufh/hy/ada_optimizer/submit/role_of_trans_num.csv',x_title='Depth',y_title='Normalized Absolute Gradient')#Random Similarity Depth Iteration Accuracy Normalized Std. of Log of Learning Utility Sorted Singular Value Index Normalized Singular Value
    # logger.plot_saved_data_w_xaxis('/home/hy/gen_study/log/weight_cifar10_res18_heinit_flipcrop_batch128_80e_lr05_seed0_0.csv',x_title='Weight Index',y_title='Normalized Weight')#Normalized Layer Index
    # generate a visualizer
    # logger = Visualizer(['train loss', 'ssim', 'Qabf'], 1, log_path='./',name="log")
    # logger = Visualizer(['train loss', 'test loss', 'train acc', 'test acc',], 2, log_path='./',name="log", label="log")
    # # during training
    # for i in range(0, 300):
    #     train_loss = 10 - i + random.random()
    #     test_loss = 50 - i + random.random()
    #     ssim = (i + random.random()) / 5
    #     qabf = (i + random.random()) / 10
    #     logger.record([train_loss, test_loss, ssim, qabf])
    #     # plot and write log
    #     logger.log()
    #     # logger.write_log()
    # logger.save_data()
    # logger.plot_saved_data('./log.csv')

