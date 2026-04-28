import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 设置缓存目录
# cache_dir = "/lichenghao/huY/assets/common_model"
# os.environ['TIMM_CACHE_DIR'] = cache_dir
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from collections import OrderedDict
import sys
from unicodedata import decimal
import pandas as pd



sys.path.append("/home/wangjzh/adam_optimizer/py/adapoly_optimizer")
import timm
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms, models
from torchvision.utils import make_grid
from vit import vit_base_cifar_patch4_32, vit_cifar_patch4_32, vit_custom_cifar_32, vit_small_cifar_patch4_32, vit_small_cifar_patch4_32_old
from resnet import Bottleneck, CAMResNet, BasicBlock, ResNetOrigin, BasicBlockwores
from imagenet_adv_img import CusImgPair 
import pickle
# from custom_img_pair import CusImgPair 

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer, loss):
        self.model = model              #用于储存模型
        self.target_layer = target_layer#目标层的名称
        self.gradients = None           #最终的梯度图
        self.loss = loss

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
        conv_output = None
        for name,layer in self.model._modules.items():      
            if name == "fc":
                break
            x = layer(x)
            if name == self.target_layer:  
                conv_output = x                      #将目标特征图保存到conv_output中            
                x.register_hook(self.save_gradient)  #设置将目标特征图的梯度保存到self.gradients中               
        return conv_output, x                        #x为最后一层特征图的结果

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        # conv_output, x = self.forward_pass_on_convolutions(x)
        conv_output = x                      #将目标特征图保存到conv_output中            
        conv_output.register_hook(self.save_gradient)
        # x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        # x = self.model.fc(x)
        x = self.model(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer, loss):
        self.model = model
        self.loss = loss
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer, loss) #用于提取特征图与梯度图

    def generate_cam(self, input_image, target_class=None):
        #1.1 前向传播，计算出目标类的最终输出值model_output，以及目标层的特征图的输出conv_output
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if self.loss is None:
            if target_class is None:
                target_class = np.argmax(model_output.data.detach().cpu().numpy())
            print("output category:", np.argmax(model_output.data.detach().cpu().numpy()))
            #one hot编码，令目标类置1
            one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
            one_hot_output[0][target_class] = 1
            # one_hot_output = 1 - one_hot_output
            # 步骤1.2 反向传播， 获取目标类相对于目标层各特征图的梯度
            target = conv_output.data.detach().cpu().numpy()[0]
        # 步骤1.2.1 清零梯度：model.zero_grad()
        self.model.zero_grad()
        # 步骤1.2.2 计算反向传播
        # model_output = model_output[0] #针对hgnet
        if self.loss is None:
            # model_output = nn.Softmax(dim=1)(model_output)
            model_output.backward(gradient=one_hot_output.to(model_output.device), retain_graph=True)
        else:
            torch.sum(model_output).backward(retain_graph=True)
        # 步骤1.2.3 获取目标层各特征图的梯度
        # guided_gradients = torch.abs(torch.clamp(self.extractor.gradients.data,min=0)).detach()      
        guided_gradients = torch.abs(self.extractor.gradients.data).detach()      
        # guided_gradients = self.extractor.gradients.data.detach().cpu().numpy()[0]
        cam = guided_gradients.permute(0,2,3,1).cpu().numpy()[0]
        # 步骤2.1 对每张梯度图求均值，作为与其对应的特征图的权重
        # weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # # 初始化热力图
        # cam = np.ones(target.shape[1:], dtype=np.float32)
        # # 步骤2.2 计算各特征图的加权值
        # for i, w in enumerate(weights):
        #     cam += w * target[i, :, :]
        # #步骤2.3 对热力图进行后处理，即将结果变换到0~255
        # cam = np.maximum(cam, 0)
        # min = [np.min(cam[0]),np.min(cam[1]),np.min(cam[2])]
        # max = [np.max(cam[0]),np.max(cam[1]),np.max(cam[2])]
        min = np.min(cam,axis=(0,1),keepdims=True)
        max = np.max(cam,axis=(0,1),keepdims=True)
        # cam = (cam - min) / (max - min)  # Normalize between 0-1
        # cam = np.clip(cam,a_min=0.5,a_max=None)-0.5
        # cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        # cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
        #                input_image.shape[3]), Image.ANTIALIAS))/255
        return cam, min, max

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        # one_hot.scatter_(1, torch.LongTensor([[ids]]).cuda(), 1.0)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return torch.abs(gradient)

class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)

        for module in self.model.named_modules():
            # self.handlers.append(module[1].register_full_backward_hook(backward_hook)) #pytorch version>1.8.0
            self.handlers.append(module[1].register_backward_hook(backward_hook))

    def generate_influence_cam(self, input_image, target_class=None):
        #1.1 前向传播，计算出目标类的最终输出值model_output，以及目标层的特征图的输出conv_output
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if self.loss is None or self.loss is not None:
            if target_class is None:
                target_class = np.argmax(model_output.data.detach().cpu().numpy())
            print("output category:", np.argmax(model_output.data.detach().cpu().numpy()))
            #one hot编码，令目标类置1
            one_hot_output = torch.LongTensor(1, model_output.size()[-1]).zero_()
            one_hot_output[0][target_class] = 1
            # one_hot_output = 1 - one_hot_output
            # 步骤1.2 反向传播， 获取目标类相对于目标层各特征图的梯度
            target = conv_output.data.detach().cpu().numpy()[0]
        # 步骤1.2.1 清零梯度：model.zero_grad()
        self.model.zero_grad()
        # 步骤1.2.2 计算反向传播
        # model_output = model_output[0] #针对hgnet
        # loss_output = self.loss(model_output, one_hot_output.cuda())
        loss_output = self.loss(model_output, torch.LongTensor([target_class]).cuda())
        grad_factor = 1 * torch.ones_like(loss_output)
        if self.loss is None:
            # model_output = nn.Softmax(dim=1)(model_output)
            model_output.backward(gradient=one_hot_output.to(model_output.device), retain_graph=True)
        else:
            loss_output.backward(gradient=grad_factor.to(model_output.device),retain_graph=True)
        one_hot_output.detach().cpu()
        # 步骤1.2.3 获取目标层各特征图的梯度
        guided_gradients = torch.abs(torch.clamp(self.extractor.gradients.data,min=0)).detach()      
        # guided_gradients = torch.mean(torch.abs(self.extractor.gradients.data),dim=1,keepdim=True).detach()      
        # guided_gradients = self.extractor.gradients.data.detach().cpu().numpy()[0]
        cam = guided_gradients.permute(0,2,3,1).cpu().numpy()[0]
        # 步骤2.1 对每张梯度图求均值，作为与其对应的特征图的权重
        # weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # # 初始化热力图
        # cam = np.ones(target.shape[1:], dtype=np.float32)
        # # 步骤2.2 计算各特征图的加权值
        # for i, w in enumerate(weights):
        #     cam += w * target[i, :, :]
        # #步骤2.3 对热力图进行后处理，即将结果变换到0~255
        # cam = np.maximum(cam, 0)
        # min = [np.min(cam[0]),np.min(cam[1]),np.min(cam[2])]
        # max = [np.max(cam[0]),np.max(cam[1]),np.max(cam[2])]
        min = np.min(cam,axis=(0,1),keepdims=True)
        max = np.max(cam,axis=(0,1),keepdims=True)
        # cam = (cam - min) / (max - min)  # Normalize between 0-1
        # cam = np.clip(cam,a_min=0.5,a_max=None)-0.5
        # cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        # cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
        #                input_image.shape[3]), Image.ANTIALIAS))/255
        return cam, min, max

def get_grad_heatmap(img_path, model , loss=None, target_class=None, exact_feat_name=None, cus_transform=None, device=torch.device('cuda'),label='',save_path=''):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    if(cus_transform == None):
        cus_transform = transforms.Compose([
        # transforms.Grayscale(1),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.225, 0.225, 0.225])
    ])
    input_tensor = cus_transform(img).to(device).unsqueeze(0)

    # # gradient
    input_tensor.requires_grad_()
    grad_Cam = GradCam(model.to(device), exact_feat_name, loss)
    heatmap, min, max = grad_Cam.generate_cam(input_tensor,target_class)

    # # guidedgradbackprop
    # gbp = GuidedBackPropagation(model=model.to(device))
    # probs, ids = gbp.forward(input_tensor)
    # gbp.backward(ids=ids[:, [0]])
    # heatmap = gbp.generate()
    # heatmap = heatmap.permute(0,2,3,1).cpu().numpy()[0]

    # max = np.max(img)
    # max = np.around(max,decimals=3)
    # min = np.min(img)
    # heatmap = heatmap.reshape(heatmap.shape[0],heatmap.shape[1],1)
    # heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    # result = img + 0.8 * np.sum(heatmap,axis=2,keepdims=True)
    # result = transforms.Resize((256,256))(img) * heatmap
    result = heatmap
    # heatmap = np.reshape(heatmap,(heatmap.shape[0]*heatmap.shape[1],heatmap.shape[2]))
    result = np.mean(result,axis=2)
    min_result = np.min(result,axis=(0,1),keepdims=True)
    max_result = np.max(result,axis=(0,1),keepdims=True)
    result = (result - min_result) / (max_result - min_result)  # Normalize between 0-1
    
    
    # result = np.clip(result,a_min=0.9,a_max=None)-0.9
    # result = result *(1/0.1)
    # show rgb image
    # plt.imshow(result)
    # feats = cam[0]
    # size = feats.shape
    # num = size[1]
    df_abs_grad = pd.DataFrame()
    plt.imshow(result[:, :], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    # # show 3 channel image
    # for i in range(3):
    # #     # ax = plt.subplot(6, 6, i + 1)
    #     df_abs_grad = df_abs_grad.append(pd.DataFrame(heatmap[:,:, i]))
    #     ax = plt.subplot(1, 3, i + 1)
    #     # ax = plt.subplot(int(num/8)+1, 8, i + 1)
    #     ax.set_title('Feature {}'.format(i))
    #     ax.axis('off')
    #     ax.set_title(' ')
    #     plt.imshow(result[:, :,i], cmap='gray')
    plt.tight_layout()
    model.cpu()
    input_tensor.cpu()
    name = "%s_ %.4f_%.4f" % (img_path.split('/')[-1].split('.')[0], min_result[0,0], max_result[0,0])
    # name = "geo3_6_1_1conv15_linenet_ %.4f_%.4f_ %.4f_%.4f_ %.4f_%.4f" % (min[0,0,0], max[0,0,0],min[0,0,1], max[0,0,1],min[0,0,2], max[0,0,2])
    plt.savefig(save_path+"heatmap_{}_{}_{}.png".format(name, label, exact_feat_name))
    df_abs_grad.to_csv(save_path+"grad_{}_{}_{}.csv".format(name, label, exact_feat_name))

def get_guidedgrad_heatmap(img_path, model , loss=None, target_class=None, exact_feat_name=None, cus_transform=None, device=torch.device('cuda'),label='',save_path=''):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    if(cus_transform == None):
        cus_transform = transforms.Compose([
        # transforms.Grayscale(1),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.225, 0.225, 0.225])
    ])
    input_tensor = cus_transform(img).to(device).unsqueeze(0)

    # # guidedgradbackprop
    gbp = GuidedBackPropagation(model=model.to(device))
    probs, ids = gbp.forward(input_tensor)
    gbp.backward(ids=ids[:, [0]])
    # gbp.backward(ids=7)
    heatmap = gbp.generate()
    heatmap = heatmap.permute(0,2,3,1).cpu().numpy()[0]

    result = heatmap
    # heatmap = np.reshape(heatmap,(heatmap.shape[0]*heatmap.shape[1],heatmap.shape[2]))
    result = np.mean(result,axis=2)
    min_result = np.min(result,axis=(0,1),keepdims=True)
    max_result = np.max(result,axis=(0,1),keepdims=True)
    result = (result - min_result) / (max_result - min_result)  # Normalize between 0-1
    
    df_abs_grad = pd.DataFrame()
    plt.imshow(result[:, :], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    model.cpu()
    input_tensor.cpu()
    name = "%s_ %.4f_%.4f" % (img_path.split('/')[-1].split('.')[0], min_result[0,0], max_result[0,0])
    # name = "geo3_6_1_1conv15_linenet_ %.4f_%.4f_ %.4f_%.4f_ %.4f_%.4f" % (min[0,0,0], max[0,0,0],min[0,0,1], max[0,0,1],min[0,0,2], max[0,0,2])
    plt.savefig(save_path+"guided_heatmap_{}_{}_{}.png".format(name, label, exact_feat_name))
    df_abs_grad.to_csv(save_path+"grad_{}_{}_{}.csv".format(name, label, exact_feat_name))

def get_influence_heatmap(img_path, model , loss=None, save_path=None, label='', target_class=None, exact_feat_name=None, cus_transform=None, device=torch.device('cuda')):
    img = Image.open(img_path).convert('RGB')
    if(cus_transform == None):
        cus_transform = transforms.Compose([
        # transforms.Grayscale(1),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    input_tensor = cus_transform(img).to(device).unsqueeze(0)
    input_tensor.requires_grad_()
    grad_Cam = GradCam(model.to(device), exact_feat_name, loss)
    heatmap, min, max = grad_Cam.generate_influence_cam(input_tensor,target_class)
    # max = np.max(img)
    # max = np.around(max,decimals=3)
    # min = np.min(img)
    # heatmap = heatmap.reshape(heatmap.shape[0],heatmap.shape[1],1)
    # heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    # result = img + 0.8 * np.sum(heatmap,axis=2,keepdims=True)
    # result = transforms.Resize((256,256))(img) * heatmap
    result = heatmap
    # heatmap = np.reshape(heatmap,(heatmap.shape[0]*heatmap.shape[1],heatmap.shape[2]))
    min_result = np.min(result,axis=(0,1),keepdims=True)
    max_result = np.max(result,axis=(0,1),keepdims=True)
    result = (result - min_result) / (max_result - min_result)  # Normalize between 0-1
    # result = np.clip(result,a_min=0.9,a_max=None)-0.9
    # result = result *(1/0.1)
    # show rgb image
    # plt.imshow(result)
    # feats = cam[0]
    # size = feats.shape
    # num = size[1]
    df_abs_grad = pd.DataFrame()
    # show 3 channel image
    for i in range(1):
    #     # ax = plt.subplot(6, 6, i + 1)
        df_abs_grad = df_abs_grad.append(pd.DataFrame(heatmap[:,:, i]))
        ax = plt.subplot(1, 1, i + 1)
        # ax = plt.subplot(int(num/8)+1, 8, i + 1)
        ax.set_title('Feature {}'.format(i))
        ax.axis('off')
        ax.set_title(' ')
        plt.imshow(result[:, :,i], cmap='gray')
    plt.tight_layout()
    model.cpu()
    input_tensor.cpu()
    name = label+("_ %.4f_%.4f" % (min[0,0,0], max[0,0,0]))
    if save_path is not None:
        plt.savefig(save_path+"heatmap_{}_{}.png".format(name, exact_feat_name))
    else:
        plt.savefig("heatmap_{}_{}.png".format(name, exact_feat_name))
    # df_abs_grad.to_csv("grad_{}_{}.csv".format(name, exact_feat_name))

def cal_data_grad_orthogonality(outer_loader, inner_loader, loss, model, calculator, path,loop_num=100):
    model.eval()
    outer_loop_index=0
    for i, (input, target,_) in enumerate(outer_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        calculator.zero_grad()
        loss.backward()
        calculator.outer_step()
        outer_loop_index+=1
        if outer_loop_index > loop_num:
            break
        inner_loop_index=0
        for j, (inner_input, inner_target,_) in enumerate(inner_loader):
            inner_loop_index+=1
            if j<=i:
                continue
            if inner_loop_index >=loop_num:
                break
            input = inner_input.cuda()
            target = inner_target.cuda()
            output = model(input)
            loss = criterion(output, target)
            calculator.zero_grad()
            loss.backward()
            calculator.inner_step()
            calculator.calculate_orthogonality(account + '/hy/nn_property/submit/nas_%s'%path) #v8mstength2 rand1 

    return 

def cal_grad_mean(model, criterion, dataloader, path, itr=50):
    model.eval()
    mgrads =[]
    for i, (input,_, target) in enumerate(dataloader):
        if i>=50:
            break
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        mgrad=[]
        for param in model.parameters():
            # mgrad.append(torch.log10(torch.mean(torch.abs(param.grad))).item())
            mgrad.append((torch.mean(torch.abs(param.grad))).item())
            param.grad.zero_()
        mgrad = mgrad/np.mean(mgrad)
        mgrads.append(mgrad)
    # df_datas = pd.DataFrame(data=mgrads)
    # mgrads.append(df_datas.mean(axis=0).to_list())
    mean_df_datas = pd.DataFrame(data=mgrads)
    
    mean_df_datas.to_csv(account + '/hy/nn_property/submit/%s_normmgrad.csv'%path)

    return 

def cal_data_grad_orthogonality_v2(outer_loader, inner_loader, criterion, model, path, loop_num=100):
    naos=[]
    total_num = 0
    for i, (input, target) in enumerate(outer_loader):
        if i >= loop_num:
            break
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        init_grads=[]
        for param in model.parameters():
            init_grads.append(param.grad.clone().reshape(-1))
            param.grad.zero_()
        for j, (inner_input, inner_target) in enumerate(inner_loader):
            if j<=i:
                continue
            if j >=loop_num:
                break
            total_num += 1
            input = inner_input.cuda()
            target = inner_target.cuda()
            output = model(input)
            loss = criterion(output, target) 
            loss.backward()
            cur_naos = []
            for k,(param) in enumerate(model.parameters()):
                g = param.grad.reshape(-1)
                # a = torch.sum(F.normalize(torch.abs(g),dim=0)*F.normalize(torch.abs(init_grads[k]),dim=0))
                # b=torch.sum(torch.abs(init_grads[k]*g))/(torch.sum((init_grads[k]*init_grads[k].conj())).sqrt()*torch.sum((g*g.conj())).sqrt())
                nao = torch.sum(torch.abs(init_grads[k]*g))/(torch.sum((init_grads[k]*init_grads[k].conj())).sqrt()*torch.sum((g*g.conj())).sqrt())
                param.grad.zero_()
                cur_naos.append(nao.item())
            naos.append(cur_naos)
    pd.DataFrame(data=naos).mean(axis=0).to_csv(path+'_nao_layer.csv')

def cal_nao_distribution(outer_loader, inner_loader, criterion, model, path, loop_num=100):
    naos=[]
    group_num=10
    for _ in range(group_num):
        naos.append(0)
    model.eval()
    total_num = 0
    avg=0
    for i, (input, target) in enumerate(outer_loader):
        if i >= loop_num:
            break
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        init_grads=[]
        for param in model.parameters():
            init_grads.append(param.grad.clone().reshape(-1))
            param.grad.zero_()
        for j, (inner_input, inner_target) in enumerate(inner_loader):
            if j<=i:
                continue
            if j >=loop_num:
                break
            total_num += 1
            input = inner_input.cuda()
            target = inner_target.cuda()
            output = model(input)
            loss = criterion(output, target) 
            loss.backward()
            nao=0
            num=0
            for k,(param) in enumerate(model.parameters()):
                g = param.grad.reshape(-1)
                next_num = num+g.size()[0]
                # a = torch.sum(F.normalize(torch.abs(g),dim=0)*F.normalize(torch.abs(init_grads[k]),dim=0))
                # b=torch.sum(torch.abs(init_grads[k]*g))/(torch.sum((init_grads[k]*init_grads[k].conj())).sqrt()*torch.sum((g*g.conj())).sqrt())
                # nao = nao*num/next_num+ g.size()[0]/next_num*torch.sum(torch.abs(init_grads[k]*g))/(torch.sum((init_grads[k]*init_grads[k].conj())).sqrt()*torch.sum((g*g.conj())).sqrt())
                nao = nao*num/next_num+ g.size()[0]/next_num*torch.sum(torch.abs(init_grads[k]/(torch.sum((init_grads[k]*init_grads[k].conj())).sqrt()))*torch.abs(g/torch.sum((g*g.conj())).sqrt()))
                param.grad.zero_()
                num = next_num    
            index = 0
            if not torch.isinf(nao).any():
                avg += nao.item()
            while nao>1/group_num and index<9:
                index += 1
                nao -= 1/group_num
            naos[index] += 1
    for i in range(len(naos)):
        naos[i] = naos[i] / total_num
    naos.append(avg/ total_num)
    pd.DataFrame(data=naos).to_csv(path+'_%dbin_naodis.csv'%group_num)

account = '/lichenghao/huY'
if __name__ == '__main__':
    from cifar10_pair_sim import CIFAR10SimPair,CIFAR100LT
    from orth_calculator import OrthCalculator
    
    criterion = nn.CrossEntropyLoss()
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
    ])
    test_data = CIFAR100LT(root="/home/wangjzh/adam_optimizer/data/cifar100_lt_outputs/cifar100-lt-r-100/1perclass",
    train=False,transform=transforms_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    
    # model = ResNetOrigin(BasicBlock, [2, 2, 2, 2],num_classes=100)
    # model = model.cuda()
    # path = 'model_cifar100_res18'
    # cal_nao_distribution(test_loader,test_loader,criterion,model,('/home/wangjzh/adam_optimizer/log/grad/submit/%s'%path))

    # model = ResNetOrigin(Bottleneck, [3, 4, 6, 3],num_classes=100) #3 4 6 3*cifar resnet 50
    # model = model.cuda()
    # path = 'model_cifar100_res50'
    # cal_nao_distribution(test_loader,test_loader,criterion,model,('/home/wangjzh/adam_optimizer/log/grad/submit/%s'%path))

    # model = vit_small_cifar_patch4_32(num_classes=100)
    # model = model.cuda()
    # path = 'model_cifar100_vits'
    # cal_nao_distribution(test_loader,test_loader,criterion,model,('/home/wangjzh/adam_optimizer/log/grad/submit/%s'%path))

    # model = vit_base_cifar_patch4_32(num_classes=100)
    # model = model.cuda()
    # path = 'model_cifar100_vitb'
    # cal_nao_distribution(test_loader,test_loader,criterion,model,('/home/wangjzh/adam_optimizer/log/grad/submit/%s'%path))

    # model = models.vgg16_bn(pretrained=False,num_classes=100) #3 4 6 3*vgg bn
    # model = model.cuda()
    # path = 'model_cifar100_vgg'
    # cal_nao_distribution(test_loader,test_loader,criterion,model,('/home/wangjzh/adam_optimizer/log/grad/submit/%s'%path))
    
    #######
    
    model_configs = [
    # (model_constructor, model_name, optimizer, batch_size, epochs, lr, seed)
    (lambda: ResNetOrigin(BasicBlock, [2, 2, 2, 2], num_classes=100), 'resnet18', 'adam', 256, 200, '0.001', 0),
    (lambda: ResNetOrigin(BasicBlock, [2, 2, 2, 2], num_classes=100), 'resnet18', 'sgd', 256, 200, '0.5', 0),
    (lambda: ResNetOrigin(Bottleneck, [3, 4, 6, 3], num_classes=100), 'resnet50', 'adam', 256, 200, '0.0005', 0),
    (lambda: ResNetOrigin(Bottleneck, [3, 4, 6, 3], num_classes=100), 'resnet50', 'sgd', 256, 200, '0.02', 0),
    (lambda: models.vgg16_bn(num_classes=100), 'vgg16bn', 'adam', 256, 200, '0.0002', 0),
    (lambda: models.vgg16_bn(num_classes=100), 'vgg16bn', 'sgd', 256, 200, '0.05', 0),
    (lambda: vit_small_cifar_patch4_32(num_classes=100), 'vits', 'adam', 256, 200, '5e-05', 0),
    (lambda: vit_small_cifar_patch4_32(num_classes=100), 'vits', 'sgd', 256, 200, '0.05', 0),
    (lambda: vit_base_cifar_patch4_32(num_classes=100), 'vitb', 'adam', 256, 200, '2e-05', 0),
    (lambda: vit_base_cifar_patch4_32(num_classes=100), 'vitb', 'sgd', 256, 200, '0.002', 0),
    ]
    
    for constructor, modelname, optimizer, batch, epoch, lr, seed in model_configs:
        model = constructor().cuda()
        model_path = f"/home/wangjzh/adam_machinism/model/adapoly/cifar_r10_{modelname}_{optimizer}_batch{batch}_lr{lr}_seed{seed}/final.pth"
        log_path = f"/home/wangjzh/adam_optimizer/log/grad/submit/model_cifar_r10_{modelname}_{optimizer}_batch{batch}_{epoch}e_lr{lr}"  # 去掉`.pth`
    
        try:
            model.load_state_dict(torch.load(model_path), strict=False)
            print(f"Loaded model from: {model_path}")
        except FileNotFoundError:
            print(f"Model file not found: {model_path} — skipping")
            continue
    
        cal_nao_distribution(test_loader, test_loader, criterion, model, log_path)

        

    