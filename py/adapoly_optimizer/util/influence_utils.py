import sys
from unicodedata import decimal
import pandas as pd


sys.path.append("/home/hy/")
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms, models
from torchvision.utils import make_grid
from mpl_toolkits.mplot3d import axes3d

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model              #用于储存模型
        self.target_layer = target_layer#目标层的名称
        self.gradients = None           #最终的梯度图

    def save_gradient(self, grad):
        self.gradients = grad           #用于保存目标特征图的梯度（因为pytorch只保存输出，相对于输入层的梯度
                                        #，中间隐藏层的梯度将会被丢弃，用来节省内存。如果想要保存中间梯度，必须
                                        #使用register_hook配合对应的保存函数使用，这里这个函数就是对应的保存
                                        #函数其含义是将梯度图保存到变量self.gradients中，关于register_hook
                                        #的使用方法我会在开一个专门的专题，这里不再详述
    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        # conv_output = None
        # for name,layer in self.model._modules.items():      
        #     if name == "fc":
        #         break
        #     x = layer(x)
        #     if self.target_layer in name:  
        #         x.requires_grad_()      
        #         conv_output = x                      #将目标特征图保存到conv_output中      
        #         x.register_hook(self.save_gradient)  #设置将目标特征图的梯度保存到self.gradients中               
        # return conv_output, x                        #x为最后一层特征图的结果

        conv_output=None
        submodule_list = []
        # print('---------', self.submodule._modules.items())
        submodule_list = list(self.model._modules.items())
        while len(submodule_list) != 0:
            name,module = submodule_list[0]
            if name == "fc":
                break
            if 'conv' in name:
                conv_input=x
            if len(self.target_layer) and name == self.target_layer[0]:
                if len(self.target_layer) == 1:
                    x = module(x)
                    if 'bn2' in name: 
                        x = conv_input+x  
                    x.requires_grad_()
                    conv_output = x
                    x.register_hook(self.save_gradient)
                self.target_layer.pop(0)
                submodule_list.pop(0)
                for elem in list(module._modules.items())[::-1]:
                    submodule_list.insert(0,elem)
            else:
                x = module(x)  
                if 'bn2' in name:  
                    x = conv_input+x  
                submodule_list.pop(0)
        return conv_output, x    

class FeatExtractor():
    """
        Extracts features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model              #用于储存模型
        self.target_layer = target_layer#目标层的名称

    def forward_pass_on_layers(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        submodule_list = []
        # print('---------', self.submodule._modules.items())
        submodule_list = list(self.model._modules.items())
        while len(submodule_list) != 0:
            name,module = submodule_list[0]
            if len(self.target_layer) and name == self.target_layer[0]:
                if len(self.target_layer) == 1:
                    x = module(x)
                    return x
                self.target_layer.pop(0)
                submodule_list.pop(0)
                for elem in list(module._modules.items())[::-1]:
                    submodule_list.insert(0,elem)
            else:
                x = module(x)  
                submodule_list.pop(0)
        return x    

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        return self.forward_pass_on_layers(x)

def get_influence(train_loader, model, criterion, target_layer=None, device=torch.device('cuda')):
    extractor = CamExtractor(model, target_layer) #用于提取特征图与梯度图
    model.eval()
    model.to(device)
    AveCI = 0
    # for i, (input, target) in enumerate(train_loader):
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        conv_output, model_output = extractor.forward_pass(input)
        loss = criterion(model_output, target)
        # print("i:", i, " loss:", torch.mean(loss))
        model.zero_grad()
        loss.backward(retain_graph=True)
        CI = torch.abs(extractor.gradients.data).detach()
        AveCI = (AveCI *i+ torch.mean(CI,dim=(0,2,3)))/(i+1)
        input.cpu()
        target.cpu()
    model.cpu()
    print(AveCI)
    pd.Series(AveCI.cpu().numpy()).to_csv("avgRI.csv")

def get_avg_grad_feat(train_loader, model, criterion, target_layer=None, device=torch.device('cuda')):
    extractor = CamExtractor(model, target_layer) #用于提取特征图与梯度图
    model.eval()
    model.to(device)
    AvgGrad = 0
    AvgAbsGrad = 0
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        conv_output, model_output = extractor.forward_pass(input)
        loss = criterion(model_output, target)
        # print("i:", i, " loss:", torch.mean(loss))
        model.zero_grad()
        loss.backward(retain_graph=True)
        grad = torch.mean(extractor.gradients.data.detach())
        absgrad = torch.mean(torch.abs(extractor.gradients.data.detach()))
        AvgGrad = (AvgGrad *i+ grad)/(i+1)
        AvgAbsGrad = (AvgAbsGrad *i+ absgrad)/(i+1)
        input.cpu()
        target.cpu()
    model.cpu()
    print('AvgGrad:', AvgGrad)
    print('AvgAbsGrad:', AvgAbsGrad)
    # pd.Series(AveGrad.cpu().numpy()).to_csv("avgRI.csv")

def sum(bin,data):
    # num = torch.sum(torch.sign(torch.abs(data)+1e-5),dim=1,keepdim=True)
    min = torch.min(data,dim=1,keepdim=True).values
    max = torch.max(data,dim=1,keepdim=True).values
    stride = (max - min)/bin
    # axis_x = torch.linspace(torch.min(data).item()+stride*0.5, torch.max(data).item()-stride*0.5, bin)
    axis_x = torch.linspace(0.5, bin-0.5, bin).unsqueeze(1)
    axis_y = torch.linspace(0.5, min.size(0)-0.5, min.size(0)).unsqueeze(0)
    axis_x = axis_x.repeat(1,min.size(0))
    axis_y = axis_y.repeat(bin,1)
    pre = 0
    result = None
    for i in range(bin):
        cur = torch.sum(0.5-0.5*torch.sign(data - (min+(i+1)*stride)),dim=1,keepdim=True)
        if result is None:
            result = cur
        else:
            result = torch.cat([result,cur - pre],dim=1) 
        pre = cur
    return result, axis_x, axis_y
    

def get_feat(train_loader, model, target_layer=None, path='/home/hy/wsol/submit', device=torch.device('cuda')):
    extractor = FeatExtractor(model.eval().to(device), target_layer) #用于提取特征图与梯度图
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})
    fig=plt.figure()
    ax1=plt.axes(projection='3d')
    # model.eval()
    # model.to(device)
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        model_output = extractor.forward_pass(input)
        reshaped_model_output = model_output.reshape(model_output.size(0)*model_output.size(1),-1)
        bins, axis_x, axis_y = sum(10,reshaped_model_output)
        # for i in range(model_output.size(0)):
        # Get the test data
        # X, Y, Z = axes3d.get_test_data(0.05)
        
        # Give the first plot only wireframes of the type y = c
        ax1.plot_wireframe(axis_x.T, axis_y.T, bins.detach().cpu(), rstride=10, cstride=0)
        ax1.set_title("Column (x) stride set to 0")
        ax1.set_zlim(0, 8)
        ax1.set_ylim(0, 80)
        plt.tight_layout()    
        input.cpu()
        target.cpu()
        break
    model.cpu()
    plt.savefig('/home/hy/wsol/submit/wireframe.png', dpi=400, bbox_inches = 'tight')
    # pd.Series(AveGrad.cpu().numpy()).to_csv("avgRI.csv")    

if __name__ == '__main__':
    
    pass