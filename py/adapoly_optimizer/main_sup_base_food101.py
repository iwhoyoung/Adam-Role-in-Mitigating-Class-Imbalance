from collections import OrderedDict
import glob
import math
import os
import random
import operator
import sys
import time

import timm

from custom_dataset import CustomDatasetDivByFile, CustomDatasetDivByFolder
from food101 import Food101
from swin import swin_t, swin_t_32

sys.path.append("/home/LAB/lufh/adapoly_optimizer/py")
import torchvision
from torchvision.datasets import CIFAR10,CIFAR100
from custom_img_pair import CusImgPair
# from imagenet_adv_img import CusImgPair
from util.grad_utils import get_grad_heatmap, get_guidedgrad_heatmap
from util.influence_utils import get_influence, get_avg_grad_feat, get_feat
# from util.picture_utils import get_avg_feature, get_feature, get_avg_output
from util.visualizer import Visualizer
from resnet import Bottleneck, CAMResNet, BasicBlock, ResNetOrigin, BasicBlockwores
from util.estimate_utils import cal_sim_norm_model, record_param_name
from cifar10_pair_sim import CIFAR10SimPair
from vit import vit_base_cifar_patch4_32, vit_cifar_patch4_32, vit_cifar_patch4_32_depth4, vit_cifar_patch4_256, vit_custom_cifar_32, vit_small_cifar_patch16_224, vit_small_cifar_patch4_32, vit_tiny_cifar_patch4_32
from xadassd_abl import XAdaSSD

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from PIL import Image
import numpy as np

import torch

seed=0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
# 固定优化结果
torch.backends.cudnn.deterministic = True
# 加速库 会导致优化随机不固定
torch.backends.cudnn.benchmark = False

import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import pandas as pd
import torch.nn.functional as F

# hyper-parameter

binary_num = 64
alpha = 1
beta = 1
gamma = 1
use_cuda = True
nThreads = 1
lr = 5e-3
server_mode = True
epoch_num = 200
batch_size = 256
dataset_name = 'food101'
model_name = 'res50'# resnet18vconv13x3 vgg16bn vits
opt_name = 'adau'#sgdwnormandsumadam sgdwredistribution8andnorm
account = "/lichenghao/huY"


# account = "/home/WuHF"
# account = "/xiaobin_phd"
# account = "/bixl_ms"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train1(train_loader, model, criterion, optimizer,mean_loss=100):
    model.train()
    train_loss = []
    js = []
    # 每次取出数量为batch size的数据
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        j=0
        while True:
            classes = model(input)
            loss = criterion(classes, target)
            # print("i:", i, " loss:", torch.mean(loss))

            # 每一轮batch需要设置optimizer.zero_grad
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            train_loss.append(loss.item())  
            j=float(j+1)    
            if loss < mean_loss:
                js.append(j)
                break
    print("j:", np.mean(js))
    return np.mean(train_loss)

def pre(train_loader, model, criterion, optimizer,mean_loss=100):
    model.train()
    train_loss = []
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        classes = model(input)
        loss = criterion(classes, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.pre()
        train_loss.append(loss.item())
        if i>16:
            return np.mean(train_loss)
    return np.mean(train_loss)

def train(train_loader, model, criterion, optimizer,mean_loss=100):
    model.train()
    train_loss = []
    flag = 0
    # input1 = list(train_cov_loader)[0][0].cuda()
    # target1 = list(train_cov_loader)[0][2].cuda()
    # 每次取出数量为batch size的数据
    for i, (input, target) in enumerate(train_loader):
        # if i == 5:
            # break
        # with torch.no_grad():
            # model.eval()
            # classes = model(input1)
            # emt_loss = criterion(classes, target1)  
        # model.train()
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        classes = model(input)
        loss = criterion(classes, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    # optimizer.calculate_mean_std(account + '/adapoly_optimizer/submit/dynamics_%s_%s_%s_batch%d_%de_lr%f_seed%d' % (dataset_name, model_name, opt_name, batch_size, epoch_num, lr, seed))
    return np.mean(train_loss)


def predict(test_loader, model):
    model.eval()
    # batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [losses, top1, top5],
        prefix='Test: ')
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test_cuda():
    # 返回当前设备索引
    print(torch.cuda.current_device())
    # 返回GPU的数量
    print(torch.cuda.device_count())
    # 返回gpu名字，设备索引默认从0开始
    print(torch.cuda.get_device_name(0))
    # cuda是否可用
    print(torch.cuda.is_available())


def adjust_learning_rate(optimizer, epoch):
    cur_lr = lr
    cur_lr *= 0.5 * (1. + math.cos(math.pi * epoch / epoch_num))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

class blockshuffle(nn.Module):
    def __init__(self, num_block=4):
        super(blockshuffle, self).__init__()
        self.num_block = num_block
        
    def get_length(self, length):
        rand = np.random.uniform(size=self.num_block)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        # lengths = []
        # for i in range(int(x.size()[dim]/32)):
        #     lengths.append(32)
        # lengths = tuple(lengths)
        # perm = torch.randperm(self.num_block)
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def forward(self, x):
        # dims = [2,3]
        # random.shuffle(dims)
        # x_strips = self.shuffle_single_dim(x, dims[0])
        # torch.cat([torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])
        dims = [1,2]
        x_strips = self.shuffle_single_dim(x, dims[0])
        torch.cat([torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])
        dims = [2,1]
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat([torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])

class resizedpad(nn.Module):
    def __init__(self, resize_rate=1.15) -> None:
        super(resizedpad, self).__init__()
        self.resize_rate = resize_rate
        
    def forward(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        # resize the input image to random size
        rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        # randomly add padding
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        # resize the image back to img_size
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)

class NoiseInput(nn.Module):
    def __init__(
            self,       
    ):
        super(NoiseInput, self).__init__()

    def forward(self, x):
        size = x.size()
        ones = torch.ones((1,size[1],size[2]),requires_grad=False).to(x.device)
        uniforms = torch.rand((1,size[1],size[2]),requires_grad=False).to(x.device)
        norms = torch.clamp(0.5+torch.randn((1,size[1],size[2]),requires_grad=False).to(x.device),min=0,max=1)
        out = torch.cat([x,ones,uniforms,norms],dim=0)
        return out

if __name__ == '__main__':
    # model = DiscResNet(DiscBasicBlock, [2, 2, 2, 2], num_classes=10, norm_layer=WeReg, disc_layers=(True, True, True, True, True))
    # model = CAMResNet(BasicBlock, [2, 2, 2, 2], num_classes=17, norm_layer=None)
    # model = ResNetOrigin(BasicBlock, [2, 2, 2, 2],num_classes=101) #3 4 6 3
    # model = ResNetOrigin(Bottleneck, [3, 4, 6, 3],num_classes=101) #3 4 6 3
    # model = models.vgg16_bn(pretrained=False,num_classes=101) 
    # model = models.resnet18(pretrained=False,num_classes=101) 
    model = models.resnet50(pretrained=False,num_classes=101) 
    # model = models.vgg16(pretrained=False,num_classes=101) #3 4 6 3
    # model = models.densenet121(num_classes=101) #3 4 6 3
    # model = vit_small_cifar_patch16_224(num_classes=101)
    # model = vit_base_cifar_patch4_32(num_classes=101)
    # model = swin_t(num_classes=101)
    # vit_model_paper = ['vit_small_patch16_224', 'vit_base_patch16_224', 'pit_b_224',
                #    'visformer_small', 'swin_tiny_patch4_window7_224']
    # model = timm.create_model(vit_model_paper[1], pretrained=False)
    # model_pretrain_list = timm.list_models(pretrained=False)
    # print(len(model_pretrain_list), model_pretrain_list[:])
    # model = nn.Sequential(ResNetOrigin(BasicBlock, [2, 3, 1, 1],num_classes=10),
                        #   vit_custom_cifar_32(num_classes=10))
    # model =models.resnet18(num_classes=17)
    # model_flower17_disc_resnet18_no_disc_diff3_nomulti linear_flower17_64_ceil ['cnn','6','1','conv2'] 'layer2','0',
    # a = torch.load(account + '/adapoly_optimizer/model/model_icml24_cifar10_resnet18vconv13x3_sgd_batch1024_200e_lr0.400000_seed0.pth')
    # b = torch.load(account + '/adapoly_optimizer/model/model_icml24_cifar10_resnet18vconv13x3_sgd_batch1024_5i_lr0.400000_seed0.pth')
    # model = DiscResNet(DiscBasicBlock, [2, 2, 2, 2], num_classes=10)
    # model.load_state_dict(torch.load(account + '/adapoly_optimizer/model/model_cifar10_vgg16bn_adassd_grad_smooth_batch256_200e_lr0.010000_seed0.pth'),strict=False)
    # pretrained_state = torch.load(account + '/asset/common_model/densenet121-a639ec97.pth')
    # 修改 key
    # new_state_dict = OrderedDict()
    # for k, v in pretrained_state.items():
    #     if 'denseblock' in k:
    #         param = k.split(".")
    #         k1 = ".".join(param[:-3] + [param[-3] + param[-2]] + [param[-1]])
    #         new_state_dict[k1] = v
    #     else:
    #         new_state_dict[k] = v
    # model.load_state_dict(new_state_dict)
    # model.load_state_dict(torch.load(account + '/asset/common_model/densenet121-a639ec97.pth'),strict=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    #     break
    # get_feature("../asset/dataset/imagenet10/val/n01440764/ILSVRC2012_val_00000293.JPEG",
    #             model, ['layer3','1','conv2'], norm=True)
    # model.load_state_dict(torch.load(account + '/asset/common_model/resnet18-5c106cde.pth'))
    # # #torch.load('../asset/deepbit/model/model_flower17_disc_resnet18_no_disc_diff3_nomulti.mdl')
    # # model.load_state_dict(torch.load(account + '/discConv/model/model_cifar10_regular_base_res18_l2pen0_batch128_size32_150e.pth'))
    # # get_feature(account + "/asset/dataset/17flowers/jpg/0/image_0001.jpg", model, ['layer2','1','conv2'], norm=False)
    # get_grad_heatmap(account + "/asset/dataset/17flowers/jpg/3/image_0314.jpg", model, target_class=None,save_path=account + '/adapoly_optimizer/submit/',
    #             label='%s_%s_%s_batch%d_%de_lr%f_seed%d' % (dataset_name,model_name, opt_name, batch_size, epoch_num, lr,seed)) #imagenet10r/val/n02909870/ILSVRC2012_val_00049238.JPEG 17flowers/jpg/0/image_0040.jpg
    # get_guidedgrad_heatmap(account + "/asset/dataset/17flowers/jpg/3/image_0314.jpg", model, target_class=None,save_path=account + '/adapoly_optimizer/submit/',
    #             label='%s_%s_%s_batch%d_%de_lr%f_seed%d' % (dataset_name,model_name, opt_name, batch_size, epoch_num, lr,seed)) #OpenImages/train/03c7gz/33af4d41ba6bf2d0.jpg

    # transforms_test = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(),transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])#,NoiseInput()
    # train_data = CIFAR10(account + '/asset/dataset/cifar10/', train=True, transform=transforms_test)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
    # ['layer3','1','bn2','bn']
    # get_influence(train_loader,model,nn.CrossEntropyLoss(),target_layer=None)
    # get_avg_grad_feat(train_loader,model,nn.CrossEntropyLoss(),target_layer=['layer2','0','conv1'])
    # get_feat(train_loader,model,target_layer=['layer2','0'])
    # get_avg_feature(train_loader,model,target_layer=['layer4','0','conv1'])
    # get_avg_output(train_loader,model)
    # cal_sim_norm_model(b, a,account + '/adapoly_optimizer/submit/sim_icml24_cifar10_resnet18vconv13x3_sgd_batch1024_5i200e_lr0')# model.cuda().state_dict()
    # record_param_name(model.state_dict(),account + '/adapoly_optimizer/submit/name_%s' % model_name,['running','num'])
    # raise Exception
    # test_cuda()
    if server_mode:
        logger = Visualizer(['train loss', 'test loss', 'acc', 'acc5'], 2, log_path=account + '/adapoly_optimizer/log/',
         name="adapoly_optimizer_food",
         label='%s_%s_%s_batch%d_%de_lr%s_seed%d' % (dataset_name,model_name, opt_name, batch_size, epoch_num, lr,seed)) # Flower17_expblock_resnet_sup
        # model = torch.load('/bixl_ms/simclr/model/model_cifar10_64_esimbin2.mdl')
        # model.load_state_dict(torch.load(account + '/discConv/model/model_flower17_res18_l2pen0001_batch16.pth'))
    else:
        logger = Visualizer(['train loss', 'test loss', 'acc', 'acc5'], 2, log_path='./', name="Flower17_nusu_resnet18_sup",label='top_nusu_nol2pen_batch2')
        # model.load_state_dict(torch.load('../asset/deepbit/model/ImageNet_disc_resnet18_diff_nodisc_37.pth'))
        # model = torch.load('../asset/deepbit/model/model_flower17_diff_scale.mdl')
    # criterion = nn.CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    # criterion = ContrastiveLoss(16)
    # optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=0)#1e-6
    # optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=0)
    # optimizer = SGD(model.parameters(), lr, weight_decay=0)#1e-6
    # optimizer = Adam(model.parameters(), lr, weight_decay=0)#1e-6
    optimizer = XAdaSSD(model.parameters(), lr, weight_decay=0)#1e-6
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr, momentum=0.9,
    #                             weight_decay=0.0005)
    # for p in model.parameters():
    #     p.mask = 1
    # for m in model.modules(): 
    #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) and m.affine:
    #         m.bias.mask=0
    best_loss = 10000.0
    # optimizer = SGD_one_click(model.parameters(), lr, weight_decay=0,betas=(0.99,0.999))#1e-6 0.0001
    scheduler =  torch.optim.lr_scheduler.MultiStepLR(optimizer,[120,160],gamma=0.1)#120,160
    # t1 rotate15
    transforms_c = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.RandomCrop(32,random.randint(0, 16)),
        # transforms.RandomCrop(32,4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # # resizedpad(random.randint(114, 166)*0.01),
        # transforms.RandomRotation(random.randint(0,180)),#15 45
        # blockshuffle(random.randint(1,5)),
        # transforms.Resize((32, 32)),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
        # NoiseInput()
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # NoiseInput()
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    ])
    if use_cuda:
        model = model.cuda()
    # train_data = CustomDatasetDivByFile(account + '/adapoly_optimizer/assets/custdataset1', train=True, transform=transforms_c)#7 4 15 11 6 2 1 3 10 5 0 16 9 12 8 13 14
    # test_data = CustomDatasetDivByFile(account + '/adapoly_optimizer/assets/custdataset1', train=False, transform=transforms_test)#7 4 15 11 6 2 1 3 10 5 0 16 9 12 8 13 14
    # train_data = OpenImage100Pair(account + '/asset/dataset/OpenImages/', train=True, transform=transforms_c)
    # test_data = OpenImage100Pair(account + '/asset/dataset/OpenImages/', train=False, transform=transforms_test)
    # train_data = CIFAR10SimPair(account + '/assets/dataset', train=True, transform=transforms_c)#7 4 15 11 6 2 1 3 10 5 0 16 9 12 8 13 14
    # test_data = CIFAR10SimPair(account + '/assets/dataset', train=False, transform=transforms_test)
    # test_data = CusImgPair(account + '/adapoly_optimizer/adv_sample2', train=False, transform=transforms_test)
    # test_data = CusImgPair(account + '/asset/dataset/adv_imagenet1000/data', train=False, transform=transforms_test)
    # train_data = CIFAR100(account + '/assets/dataset', train=True, transform=transforms_c)
    # test_data = CIFAR100(account + '/assets/dataset', train=False, transform=transforms_test)
    # train_data = Flower17Pair(account + '/asset/dataset/17flowers/jpg', train=True, transform=transforms_c)
    # test_data = Flower17Pair(account + '/asset/dataset/17flowers/jpg', train=False, transform=transforms_test)
    train_data = Food101(root = account + "/assets/dataset/food101", split = "train", transform=transforms_c)
    test_data = Food101(root = account + "/assets/dataset/food101", split = "test", transform=transforms_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=nThreads)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=nThreads)
    # train_cov_loader = torch.utils.data.DataLoader(train_data, batch_size=2048, shuffle=False)
    train_loss=100.
    train_loss = pre(train_loader, model, criterion, optimizer,train_loss)
    for epoch in range(epoch_num):
        # transforms_c = transforms.Compose([
        #     # transforms.RandomResizedCrop(256),
        #     # transforms.RandomCrop(32,4),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32,random.randint(0, 16)),
        #     # transforms.RandomGrayscale(p=0.2),
        #     # transforms.RandomRotation(15),
        #     transforms.ToTensor(),
        #     resizedpad(random.randint(114, 166)*0.01),
        #     transforms.RandomRotation(random.randint(0,180)),#15 45
        #     blockshuffle(random.randint(1,5)),
            
        #     # transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        #     # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
        #     # NoiseInput()
        #     # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        # ])
        # train_data = CIFAR10SimPair(account + '/assets/dataset', train=True, transform=transforms_c)
        # adjust_learning_rate(optimizer, epoch)
        train_loss = train(train_loader, model, criterion, optimizer,train_loss)
        # optimizer.epoch_step(train_loss)
        if epoch==0 or epoch==epoch_num-1 or (epoch+1) % 10 == 0:
            acc, acc5, test_loss = predict(test_loader, model)
        # torch.save(model.state_dict(), account + '/adapoly_optimizer/model/model_%s_%s_%s_batch%d_5i_lr%f_seed%d.pth' % (dataset_name, model_name, opt_name, batch_size, lr, seed))
        torch.save(model.state_dict(), account + '/adapoly_optimizer/model/model_%s_%s_%s_batch%d_%de_lr%s_seed%d.pth' % (dataset_name, model_name, opt_name, batch_size, epoch_num, lr, seed))
        # torch.save(model.state_dict(), account + '/adapoly_optimizer/model/model_%s_%s_%s_batch%d_%de_lr%f_seed0.pth' % (dataset_name, model_name, opt_name, batch_size, epoch_num, lr))
        logger.record([train_loss, test_loss, acc.item(), acc5.item()])
        logger.log()
        logger.write_log()
        scheduler.step()
    # acc, acc5, test_loss = predict(test_loader, model)
    # raise Exception()
    logger.save_data()
    logger.plot_loss()
    if server_mode:
        pass
        torch.save(model.state_dict(), account + '/adapoly_optimizer/model/model_%s_%s_%s_batch%d_%de_lr%s_seed%d.pth' % (dataset_name, model_name, opt_name, batch_size, epoch_num, lr, seed))
        # torch.save(model, account + '/simclr/model/model_flower17_discrete.mdl')
        # pd.DataFrame().to_csv(account + '/simclr/submit/' + ('fl17acc_%.5f' % (acc)))
        # mAPs.to_csv('/bixl_ms/deepbit/submit/' + str(mAP))
    else:
        torch.save(model.state_dict(), '../asset/deepbit/model/model_flower17_nusu_nol2pen_batch2.pth')
        # torch.save(model, '../asset/deepbit/model/model_flower17_diff_same_output.mdl')
        print('mAP:', acc)
