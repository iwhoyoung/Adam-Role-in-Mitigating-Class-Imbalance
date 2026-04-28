from statistics import mode
import sys
import pandas as pd


sys.path.append("/home/hy/")
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from torch import nn
from torchvision import transforms, models
from torchvision.utils import make_grid

def get_picture_rgb(picture_path):
    '''
    该函数实现了显示图片的RGB三通道颜色
    '''
    a = cv.imread(picture_path)
    # i:j:k i表示开始 j表示结束 k表示步长
    picture_dir = cv.imread(picture_path)[:, :, ::-1]
    # 取单一通道值显示
    for i in range(3):
        img = picture_dir[:,:,i]
        # nrows, ncols, index
        ax = plt.subplot(1, 3, i + 1)
        ax.set_title('Feature {}'.format(i))
        ax.axis('off')
        plt.imshow(img, cmap='gray')

    # r = picture_dir.copy()
    # r[:,:,0:2]=0
    # ax = plt.subplot(1, 4, 1)
    # ax.set_title('B Channel')
    # # ax.axis('off')
    # plt.imshow(r)
    #
    # g = picture_dir.copy()
    # g[:,:,0]=0
    # g[:,:,2]=0
    # ax = plt.subplot(1, 4, 2)
    # ax.set_title('G Channel')
    # # ax.axis('off')
    # plt.imshow(g)
    #
    # b = picture_dir.copy()
    # b[:,:,1:3]=0
    # ax = plt.subplot(1, 4, 3)
    # ax.set_title('R Channel')
    # # ax.axis('off')
    # plt.imshow(b)

    plt.show()


# 中间特征提取
# class FeatureExtractor(nn.Module):
#     def __init__(self, submodule, extracted_layers):
#         super(FeatureExtractor, self).__init__()
#         self.submodule = submodule
#         self.extracted_layers = extracted_layers
#
#     def forward(self, x):
#         outputs = []
#         print('---------', self.submodule._modules.items())
#         for name, module in self.submodule._modules.items():
#             if "fc" in name:
#                 x = x.view(x.size(0), -1)
#             print(module)
#             x = module(x)
#             print('name', name)
#             if name in self.extracted_layers:
#                 outputs.append(x)
#         return outputs

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.eval_modules = []
        self.extracted_layers = extracted_layers.copy()
        self.init_model()

    def forward(self, x):
        outputs = []
        submodule_list = []
        # print('---------', self.submodule._modules.items())
        submodule_list = list(self.submodule._modules.items())
        while len(submodule_list) != 0:
            name,module = submodule_list[0]
            # isinstance(module,CatDiscreteConv)需要import的路径一致，否则相同类也不会判断为同类
            # if  len(module._modules.items()) == 0 or isinstance(module,CatDiscreteConv):
            #     x = module(x)
            #     submodule_list.pop(0)
            #     if name in self.extracted_layers:
            #         outputs.append(x)
            #         if len(outputs) == 5:
            #             return outputs
            # else:
            #     sup_name = name
            #     submodule_list.pop(0)
            #     for elem in list(module._modules.items())[::-1]:
            #         submodule_list.insert(0,elem)
            if name == self.extracted_layers[0]:
                if len(self.extracted_layers) == 1:
                    x = module(x)   
                    outputs.append(x)
                    return outputs
                self.extracted_layers.pop(0)
                submodule_list.pop(0)
                for elem in list(module._modules.items())[::-1]:
                    submodule_list.insert(0,elem)
            else:
                x = module(x)   
                submodule_list.pop(0)
        return outputs

class ModuleExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(ModuleExtractor, self).__init__()
        self.submodule = submodule
        self.eval_modules = []
        self.extracted_layers = extracted_layers.copy()
        self.init_model()
        
    def reset(self, submodule, extracted_layers):
        self.submodule = submodule
        self.extracted_layers = extracted_layers.copy()

    def init_model(self):
        submodule_list = []
        submodule_list = list(self.submodule._modules.items())
        while len(submodule_list) != 0:
            name,module = submodule_list[0]
            if name == self.extracted_layers[0]:
                if len(self.extracted_layers) == 1:
                    self.eval_modules.append(module)
                    return
                self.extracted_layers.pop(0)
                submodule_list.pop(0)
                for elem in list(module._modules.items())[::-1]:
                    submodule_list.insert(0,elem)
            else:
                self.eval_modules.append(module)
                submodule_list.pop(0)

    def forward(self, x):
        for module in self.eval_modules:
            x = module(x)
        return x

account = "/home"
def get_feature(img_path, model, exact_list, name='',device=torch.device('cuda'), norm = False):  # 特征可视化
    # 输入数据
    img = cv.imread(img_path)[:, :, ::-1]
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    input_tensor = transforms.ToTensor()(img).to(device).unsqueeze(0)
    # input_tensor = dec2bin(input_tensor)
    # 特征输出
    net = model.to(device)
    # net.load_state_dict(torch.load('./model/net_050.pth'))
    if exact_list is None:
        # bn1 relu
        exact_list = ['cnn','layer1','0','conv2']
        # exact_list = ['cnn','conv1']
    myexactor = FeatureExtractor(net, exact_list)  # 输出是一个网络
    x = myexactor(input_tensor)
    feats = x[0]
    size = feats.shape
    if norm:
        feats = feats.reshape(size[0], size[1], -1)  # [B, C, H*W]
        mean = torch.mean(feats, 2, keepdim=True)  # [B, C, 1]
        min = torch.min(feats, 2, keepdim=True).values  # [B, C, 1]
        max = torch.max(feats, 2, keepdim=True).values  # [B, C, 1]
        diff = max - min
        feats = (feats - mean) / diff + 0.5
        feats = feats.reshape(size)  # [B, C, H*W]
    num = size[1]
    # num=6
    #3.4
    plt.figure(figsize=(20, 60))
    # fig, (axes1, axes2) = plt.subplots(2, 1, figsize=(100, 10))
    # 特征输出可视化
    for i in range(num):
        # ax = plt.subplot(1, 6, i + 1)
        ax = plt.subplot(int(num/8)+1, 8, i + 1)
        ax.set_title('Feature {}'.format(i))
        ax.axis('off')
        ax.set_title(' ')
        # plt.imshow(x[0].data.cpu().numpy()[0, i, :, :], cmap='jet')
        plt.imshow(feats.data.cpu().numpy()[0, i, :, :], cmap='gray')

    plt.tight_layout()  # 图像每次都不一样，是因为模型每次都需要前向传播一次，不是加载的与训练模型
    plt.savefig(name+"_{}.png".format(exact_list[0]))
    # plt.savefig("fig_disc_finalv1nomax_stage4_0_{}.png".format(exact_list[0]))


def get_avg_feature(train_loader, model, target_layer=None, device=torch.device('cuda')):  # 特征可视化
    # 特征输出
    model.eval()
    net = model.to(device)
    AvgOut = 0
    AvgAbsOut = 0
    if target_layer is None:
        # bn1 relu
        target_layer = ['cnn','layer1','0','conv2']
        # exact_list = ['cnn','conv1']
    myexactor = ModuleExtractor(net, target_layer)  # 输出是一个网络
    for i, (input, _) in enumerate(train_loader):
        # myexactor.reset(net, target_layer)
        input = input.to(device)
        x = myexactor(input)
        out = torch.mean(x[0])
        absout = torch.mean(torch.abs(x[0]))
        AvgOut = (AvgOut *i+ out.detach())/(i+1)
        AvgAbsOut = (AvgAbsOut *i+ absout.detach())/(i+1)
        input.detach()
    print('AvgOut:', AvgOut)
    print('AvgAbsOut:', AvgAbsOut)
    return AvgOut, AvgAbsOut

def get_avg_output(train_loader, model, device=torch.device('cuda')):  # 特征可视化
    # 特征输出
    model.eval()
    net = model.to(device)
    AvgOut = 0
    AvgAbsOut = 0
    for i, (input, _) in enumerate(train_loader):
        # myexactor.reset(net, target_layer)
        input = input.to(device)
        x = net(input)
        out = torch.mean(x[0])
        absout = torch.mean(torch.abs(x[0]))
        AvgOut = (AvgOut *i+ out.detach())/(i+1)
        AvgAbsOut = (AvgAbsOut *i+ absout.detach())/(i+1)
        input.detach()
    print('AvgOut:', AvgOut)
    print('AvgAbsOut:', AvgAbsOut)
    return AvgOut, AvgAbsOut

def dec2bin(decimal):
        decimal = torch.floor(decimal*255)
        bins = []
        for i in range(8):
            cur_bin = torch.sign((decimal - 2**(7-i)))
            bins.append(cur_bin)
            decimal = decimal - 0.5*(cur_bin+1)*2**(7-i)
        return torch.cat(bins,dim=1)

def get_output(img_path, model, device=torch.device('cuda')):
    # 输入数据
    img = cv.imread(img_path)[:, :, ::-1]
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    input_tensor = transforms.ToTensor()(img).to(device).unsqueeze(0)

    # 特征输出
    net = model.to(device)
    model.eval()
    return net(input_tensor)

def get_bn_params(model, check_name):
    df_params = pd.DataFrame()
    list_mean=[]
    list_var=[]
    list_name=[]
    with torch.no_grad():
        for name, param in model.named_parameters():
            #print(name,param.data.shape)
            if check_name[0] in name and check_name[1] in name:
                p = param.reshape(-1)
                df_params = df_params.append(pd.Series(p), ignore_index=True)
                list_mean.append(torch.mean(p).numpy())
                list_var.append(torch.var(p).numpy())
                list_name.append(name)
                # df_params = df_params.append(pd.DataFrame(list_mean), ignore_index=True)
                # df_params = df_params.append(pd.DataFrame(list_var), ignore_index=True)
    # df_params.reset_index(drop=True, inplace=True)
    df_params = pd.DataFrame(df_params.values.T)
    df_params = df_params.append(pd.Series(list_name), ignore_index=True)
    df_params = df_params.append(pd.Series(list_mean), ignore_index=True)
    df_params = df_params.append(pd.Series(list_var), ignore_index=True)
    df_params.to_csv("params_{}.csv".format(check_name))
            
def get_conv_params(model, check_name):
    df_params = pd.DataFrame()
    with torch.no_grad():
        for name, param in model.named_parameters():
            #print(name,param.data.shape)
            if check_name[0] in name and check_name[1] in name:
                size = param.size()
                # ps = param.reshape(size[0]*size[1],size[2],size[3])
                for i in range(size[0]):
                    input_params = pd.DataFrame()
                    list_mean=[]
                    list_var=[]
                    for j in range(size[1]):
                        p = param[i,j]
                        input_params = pd.concat([input_params, pd.DataFrame(p.numpy())],axis=1, ignore_index=True)
                        # list_mean.append(torch.mean(p).numpy())
                        list_mean.append(torch.mean(torch.abs(p)).numpy())
                        list_var.append(torch.var(p).numpy())
                    df_params = df_params.append(input_params, ignore_index=True)
                    df_params = df_params.append(pd.Series(list_mean), ignore_index=True)
                    df_params = df_params.append(pd.Series(list_var), ignore_index=True)
    df_params.to_csv("params_{}.csv".format(check_name))    
    

if __name__ == '__main__':
    # generate_pure_color_picture(600, 600, 0, "0.jpg")
    # get_picture_rgb("../asset/disc_conv/127_disc_lr001_class13_acc100_no_clip1.jpg")
    # get_feature("../asset/dataset/17flowers/jpg/0/image_0001.jpg", torch.load(account + '/hy/discConv/model/model_flower17_disc_res80_ceil_test.mdl'), None)
    pass