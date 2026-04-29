from collections import OrderedDict
import glob
import math
import os
import random
import operator
import sys
import time
from datetime import datetime
import time
import timm
import pickle
from custom_dataset import CustomDatasetDivByFile, CustomDatasetDivByFolder
from swin import swin_t_32
import matplotlib.pyplot as plt
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
from cifar10_pair_sim import CIFAR10SimPair,CIFAR10LT,CIFAR100LT
from vit import vit_base_cifar_patch4_32, vit_cifar_patch4_32, vit_cifar_patch4_32_depth4, vit_cifar_patch4_256, vit_custom_cifar_32, vit_small_cifar_patch4_32, vit_tiny_cifar_patch4_32,vit_small_cifar_patch16_224,vit_base_cifar_patch16_224,vit_small_cifar_patch16_224,vit_base_cifar_patch16_224
from adassd_gamma import Adam_bn,Adam_ini,Adam_Sbn,Sgd_m,SGD_Sbn
from torchvision import models
from torch.utils.data import Subset

import argparse  # 新增：导入参数解析库
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
parser = argparse.ArgumentParser(description='训练参数配置')
parser.add_argument('--binary_num', type=int, default=64, help='二进制位数')
parser.add_argument('--alpha', type=float, default=1.0, help='alpha超参数')
parser.add_argument('--beta', type=float, default=1.0, help='beta超参数')
parser.add_argument('--gamma', type=float, default=1.0, help='gamma超参数')
parser.add_argument('--use_cuda', action='store_true', default=True, help='是否使用CUDA')
parser.add_argument('--nThreads', type=int, default=16, help='数据加载线程数')
parser.add_argument('--lr', type=float, default=5e0, help='学习率')
parser.add_argument('--server_mode', action='store_true', default=True, help='是否服务器模式')
parser.add_argument('--epoch_num', type=int, default=200, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=256, help='批量大小')
parser.add_argument('--dataset_name', type=str, default='cifar100', help='数据集名称')
parser.add_argument('--datapath', type=str, default='adam_optimizer/data/cifar100_lt_outputs/cifar100-lt-r-10', help='数据集名称')
parser.add_argument('--model_name', type=str, default='vits', help='模型名称')
parser.add_argument('--opt_name', type=str, default='adainit', help='优化器名称')
parser.add_argument('--account', type=str, default='/home/wangjzh', help='账户路径')
parser.add_argument('--cuda_visible_devices', type=str, default='7', help='可见的CUDA设备')
parser.add_argument('--seed', type=int, default='0', help='可见的CUDA设备')
args = parser.parse_args()
seed=args.seed
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
from torch.utils.data import DataLoader
# hyper-parameter

from collections import defaultdict
from collections import Counter

# 新增：创建参数解析器并定义参数

binary_num = args.binary_num
alpha = args.alpha
beta = args.beta
gamma = args.gamma
use_cuda = args.use_cuda
nThreads = args.nThreads
lr = args.lr
server_mode = args.server_mode
epoch_num = args.epoch_num
batch_size = args.batch_size
dataset_name = args.dataset_name
model_name = args.model_name
opt_name = args.opt_name
datapath=args.datapath
account = "/home/wangjzh"
# binary_num = 64
# alpha = 1
# beta = 1
# gamma = 1
# use_cuda = True
# nThreads = 1
# lr = 5e0
# server_mode = True
# epoch_num = 200
# batch_size = 256
# dataset_name = 'cifar10'
# model_name = 'vits'# res18vconv13x3 vgg16bn vits
# opt_name = 'adainit'#sgdwnormandsumadam sgdwredistribution8andnorm adacertaintywoadd
# account = "/lichenghao/huY"


# account = "/home/WuHF"
# account = "/xiaobin_phd"
# account = "/bixl_ms"
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

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
    # spend = 0
    class_loss_train = defaultdict(lambda: [0.0, 0])  # [sum_loss, count]
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
        # print(classes)
        # print(target)
        per_sample_loss = F.cross_entropy(classes, target, reduction='none')
        for t, l in zip(target, per_sample_loss):
            class_idx = t.item()
            class_loss_train[class_idx][0] += l.item()  # 累加类别loss
            class_loss_train[class_idx][1] += 1         # 累加类别样本数
        # start = time.perf_counter()       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # end = time.perf_counter()
        # spend = spend + (end - start)
        # print(f"{i} 执行时间: {spend:.6f}秒")
        train_loss.append(loss.item())
    #optimizer.calculate_mean_std(account + '/adapoly_optimizer/submit/dynamics_%s_%s_%s_batch%d_%de_lr%f_seed%d' % (dataset_name, model_name, opt_name, batch_size, epoch_num, lr, seed))
    class_avg_train = {k: v[0]/v[1] if v[1] else 0.0 for k, v in class_loss_train.items()}
    return np.mean(train_loss), class_avg_train 


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
    class_loss_test = defaultdict(lambda: [0.0, 0])
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()
            output = model(input)
            # print(output)
            # print(target)
            loss = criterion(output, target)

            # measure accuracy and record loss
            per_sample_loss = F.cross_entropy(output, target, reduction='none')
            for t, l in zip(target, per_sample_loss):
                class_idx = t.item()
                class_loss_test[class_idx][0] += l.item()  # 累加类别loss
                class_loss_test[class_idx][1] += 1         # 累加类别样本数
            acc1, acc5 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            progress.display(i)
            
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    class_avg_test = {k: v[0]/v[1] if v[1] else 0.0 for k, v in class_loss_test.items()}
    return top1.avg, top5.avg, losses.avg, class_avg_test  # 返回整体指标和类别loss


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


def analyze_class_distribution(dataset):
    """
    分析数据集中的类别分布
    
    参数:
    dataset: 数据集实例，可以是原始数据集或 Subset
    
    返回:
    dict: 每个类别的样本数量统计
    """
    # 检查数据集类型
    if hasattr(dataset, 'targets'):
        # 原始数据集
        class_counts = Counter(dataset.targets)
    elif hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        # Subset 对象
        parent_dataset = dataset.dataset
        indices = dataset.indices
        
        # 获取子集中的标签
        subset_targets = [parent_dataset.targets[i] for i in indices]
        class_counts = Counter(subset_targets)
    else:
        raise ValueError("不支持的数据集类型")
    
    # 获取类别总数
    num_classes = len(class_counts)
    
    # 获取样本总数
    total_samples = len(dataset)
    
    # 计算最大、最小和平均样本数
    max_samples = max(class_counts.values())
    min_samples = min(class_counts.values())
    avg_samples = total_samples / num_classes
    
    # 计算不平衡比率 (最大类别样本数 / 最小类别样本数)
    imbalance_ratio = max_samples / min_samples
    
    # 获取样本数最多的类别和样本数最少的类别
    most_common_classes = class_counts.most_common(5)  # 前5个样本数最多的类别
    least_common_classes = class_counts.most_common()[-5:]  # 后5个样本数最少的类别
    
    # 计算每个类别的样本比例
    class_ratios = {cls: count / total_samples for cls, count in class_counts.items()}
    
    # 计算累积分布
    sorted_counts = sorted(class_counts.values(), reverse=True)
    cumulative_distribution = np.cumsum(sorted_counts) / total_samples
    
    # 结果汇总
    stats = {
        'num_classes': num_classes,
        'total_samples': total_samples,
        'samples_per_class': dict(class_counts),
        'max_samples': max_samples,
        'min_samples': min_samples,
        'avg_samples': avg_samples,
        'imbalance_ratio': imbalance_ratio,
        'most_common_classes': most_common_classes,
        'least_common_classes': least_common_classes,
        'class_ratios': class_ratios,
        'cumulative_distribution': cumulative_distribution
    }
    
    return stats

def print_class_distribution_stats(stats, dataset):
    """
    打印类别分布统计信息
    
    参数:
        stats (dict): 类别分布统计信息
        dataset (ImageNetLT): ImageNetLT 数据集实例
    """
    print(f"数据集类别分布统计:")
    print(f"  - 总类别数: {stats['num_classes']}")
    print(f"  - 总样本数: {stats['total_samples']}")
    print(f"  - 样本数最多的类别: {stats['max_samples']}")
    print(f"  - 样本数最少的类别: {stats['min_samples']}")
    print(f"  - 平均每个类别样本数: {stats['avg_samples']:.2f}")
    print(f"  - 不平衡比率 (最大/最小): {stats['imbalance_ratio']:.2f}")
    
    print("\n样本数最多的5个类别:")
    for cls_id, count in stats['most_common_classes']:
        class_name = dataset.classes[cls_id] if hasattr(dataset, 'classes') else f"Class {cls_id}"
        print(f"  - {class_name} (ID: {cls_id}): {count} samples")
    
    print("\n样本数最少的5个类别:")
    for cls_id, count in reversed(stats['least_common_classes']):
        class_name = dataset.classes[cls_id] if hasattr(dataset, 'classes') else f"Class {cls_id}"
        print(f"  - {class_name} (ID: {cls_id}): {count} samples")
    
    # 绘制累积分布曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(stats['cumulative_distribution'])
        plt.title('类别样本累积分布')
        plt.xlabel('类别数量 (按样本数从多到少排序)')
        plt.ylabel('累积样本比例')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("\n[警告] Matplotlib未安装，无法绘制累积分布曲线。")
        print("可以使用以下命令安装: pip install matplotlib")
def select_top_n_classes(dataset, n, label_list=[], plot=True, save_path=None, figsize=(12, 6)):
    """
    选择数据集中样本数量最多的前n个类别（或指定类别），并绘制/保存类别频率柱状图
    
    参数:
    dataset (ImageNetLT): ImageNetLT 数据集实例
    n (int): 需要保留的类别数量（当label_list为空时生效）
    label_list (list): 指定保留的类别列表（优先级高于n）
    plot (bool): 是否显示图表
    save_path (str): 图像保存路径（如"plots/class_distribution.png"），为None则不保存
    figsize (tuple): 图表尺寸
    
    返回:
    tuple: (Subset, list) 包含筛选后子集和保留的类别列表
    """
    if n <= 0 and not label_list:
        raise ValueError("n 必须是正整数（当label_list为空时）")
    
    # 统计每个类别的样本数量
    class_counts = Counter(dataset.targets)
    # 按频率从高到低排序所有类别
    sorted_classes = [cls for cls, _ in class_counts.most_common()]
    print(sorted_classes)
    sorted_counts = [class_counts[cls] for cls in sorted_classes]
    
    # 确定保留的类别
    if len(label_list) > 0:
        # 检查指定类别是否都存在于数据集中
        invalid_labels = [cls for cls in label_list if cls not in class_counts]
        if invalid_labels:
            raise ValueError(f"指定的类别不存在于数据集中: {invalid_labels}")
        top_n_classes = label_list
    else:
        # 取前n个类别（若总类别数少于n则取全部）
        top_n_classes = sorted_classes[:min(n, len(sorted_classes))]
        
    
    # 收集属于保留类别的样本索引
    selected_indices = []
    class_counts = {cls: 0 for cls in top_n_classes}  # 初始化每类的计数器

    for idx, target in enumerate(dataset.targets):
        if target in top_n_classes:
            if class_counts[target] < 10:
                selected_indices.append(idx)
                class_counts[target] += 1
    print(class_counts)
    
    # 绘制类别频率柱状图
    if plot or save_path is not None:
        plt.figure(figsize=figsize)
        # 绘制所有类别的频率
        bars = plt.bar(
            range(len(sorted_counts)), 
            sorted_counts, 
            color=['#1f77b4' if cls in top_n_classes else '#7f7f7f' for cls in sorted_classes]
        )
        # 添加类别标签和标题
        plt.xlabel('类别（按频率排序）', fontsize=12)
        plt.ylabel('样本数量', fontsize=12)
        plt.title(f'类别频率分布（保留类别：{len(top_n_classes)}个）', fontsize=14)
        
        # 标记x轴为类别ID（自动调整间隔避免拥挤）
        tick_interval = max(1, len(sorted_classes) // 20)  # 最多显示20个标签
        plt.xticks(
            range(0, len(sorted_classes), tick_interval),
            [sorted_classes[i] for i in range(0, len(sorted_classes), tick_interval)],
            rotation=45, 
            ha='right'
        )
        
        # 添加网格线
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 突出显示保留的类别
        for i, bar in enumerate(bars):
            if sorted_classes[i] in top_n_classes:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
        
        plt.tight_layout()
        
        # 保存图像（如果指定了路径）
        if save_path is not None:
            # 创建保存目录（如果不存在）
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 类别分布图表已保存至: {save_path}")
        
        # 显示图像（如果需要）
        if plot:
            plt.show()
        else:
            plt.close()  # 不显示时关闭图表，释放内存
    
    # 创建子集
    subset = Subset(dataset, selected_indices)
    return subset, top_n_classes

def select_top_n_classes_lmdb(dataset, n, label_list=[], plot=True, save_path=None, figsize=(12, 6), cache_path=None):
    if n <= 0 and not label_list:
        raise ValueError("n 必须是正整数（当 label_list 为空时）")
    
    if cache_path and os.path.exists(cache_path):
        print(f"🔖 从缓存加载标签: {cache_path}")
        with open(cache_path, 'rb') as f:
            all_targets = pickle.load(f)
    else:
        print("🔍 正在扫描数据集以统计每个类别的样本数量...")
        all_targets = []
        for idx in range(len(dataset)):
            _, target = dataset[idx]
            all_targets.append(target)
        print(f"✅ 扫描完成，共计样本数: {len(all_targets)}")

        # 写入缓存
        if cache_path:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(all_targets, f)
            print(f"💾 标签缓存已保存到: {cache_path}")

    # ======= 统计与筛选 =======
    class_counts = Counter(all_targets)
    sorted_classes = [cls for cls, _ in class_counts.most_common()]
    sorted_counts = [class_counts[cls] for cls in sorted_classes]

    if len(label_list) > 0:
        invalid_labels = [cls for cls in label_list if cls not in class_counts]
        if invalid_labels:
            raise ValueError(f"指定的类别不存在于数据集中: {invalid_labels}")
        top_n_classes = label_list
    else:
        top_n_classes = sorted_classes[:min(n, len(sorted_classes))]

    selected_indices = []
    selected_counts = {cls: 0 for cls in top_n_classes}
    for idx, target in enumerate(all_targets):
        if target in top_n_classes:
            if selected_counts[target] < 10:
                selected_indices.append(idx)
                selected_counts[target] += 1

    #print(f"📦 每类最多 10 个样本，实际选择情况: {selected_counts}")

   
    subset = Subset(dataset, selected_indices)
    return subset, top_n_classes

class ImageNetLT_LMDB(Dataset):
    def __init__(self, root, version='imagenetlt_lt', train=True, transform=None, target_transform=None):
        import io
        import lmdb

        split = 'train' if train else 'test'
        self.path = os.path.join(root, version, f'{split}.lmdb')
        self.env = lmdb.open(self.path, readonly=True, lock=False, readahead=False, meminit=False)
        self.transform = transform
        self.target_transform = target_transform
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
        self._io = io

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            item = pickle.loads(txn.get(str(index).encode()))
        image = Image.open(self._io.BytesIO(item['image'])).convert('RGB')
        target = item['label']
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

if __name__ == '__main__':
    if model_name=='vgg16bn':
        if dataset_name=='imagenet': model = models.vgg16_bn(pretrained=False, num_classes=1000)
        else: model = models.vgg16_bn(pretrained=False, num_classes=100)
    elif model_name=='vgg16':
        if dataset_name=='imagenet': model = models.vgg16(pretrained=False, num_classes=1000)
        else: model = models.vgg16(pretrained=False, num_classes=100)
    elif model_name=='resnet18':
        if dataset_name=='imagenet':
            model = models.resnet18(num_classes=1000,pretrained=False)
            #model = models.resnet18(num_classes=1000)
        else:
            model = ResNetOrigin(BasicBlock, [1,1,1,1], num_classes=100)
    elif model_name=='resnet50':
        # if dataset_name=='imagenet':model = ResNetOrigin(Bottleneck, [3, 4, 6, 3], num_classes=1000)
        if dataset_name=='imagenet':model = models.resnet50(num_classes=1000,pretrained=False)
        else:model = ResNetOrigin(Bottleneck, [3, 4, 6, 3], num_classes=100)
        
    elif model_name=='vits':
        if dataset_name=='imagenet':model = vit_small_cifar_patch16_224(num_classes=1000)
        else:model = vit_small_cifar_patch4_32(num_classes=100)
    elif model_name=='vitb':
        if dataset_name=='imagenet':model = vit_base_cifar_patch16_224(num_classes=1000)
        else:model = vit_base_cifar_patch4_32(num_classes=100)

    init_model_dir = os.path.join(account, 'adam_machinism', 'model', 'adapoly', 'init_model')
    init_model_path = os.path.join(init_model_dir, f"init_{dataset_name}_{model_name}_seed{seed}_ngpu{torch.cuda.device_count()}_1,1,1,1.pth")
    os.makedirs(init_model_dir, exist_ok=True)
    if torch.cuda.device_count() > 1:
        log(f"🔧 使用 {torch.cuda.device_count()} 块 GPU 进行并行训练")
        model = nn.DataParallel(model)
    model = model.cuda()
    # if os.path.exists(init_model_path):
    #     print(f"[初始化] 发现相同参数的初始化模型，加载: {init_model_path}")
    #     model.load_state_dict(torch.load(init_model_path))
    # else:
    #     print(f"[初始化] 未找到初始化模型，保存当前初始化状态到: {init_model_path}")
    #     torch.save(model.state_dict(), init_model_path)
    print(torch.cuda.device_count())
    if dataset_name=='imagenet':
        n_lenth=1000
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # 添加15度的随机旋转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        n_lenth=100
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
        ])
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
        ])
    
    if server_mode:
        balance_param = datapath.split("cifar100-lt-")[-1] if "cifar100-lt-" in datapath else "unknown"
        class_metrics = [f"class_{i}_{balance_param}_{t}"  for i in range(n_lenth) for t in ['train', 'test']]
        all_metrics = class_metrics + ['train loss', 'test loss', 'acc', 'acc5']
        logger = Visualizer(
            all_metrics, 2,
            log_path=os.path.join(account, 'adam_optimizer/log/adapoly'),
            name=f"adau_optimizer_100{dataset_name}_{model_name}_{opt_name}_batch{batch_size}_{epoch_num}e_lr{lr}_seed{seed}_v2",
            label=f"{dataset_name}_{model_name}_{opt_name}_batch{batch_size}_{epoch_num}e_lr{lr}_seed{seed}_v2"
        )
    else:
        logger = Visualizer(['train loss', 'test loss', 'acc', 'acc5'], 2, log_path='./', name="Flower17_nusu_resnet18_sup", label='top_nusu_nol2pen_batch2')



    criterion = nn.CrossEntropyLoss()

    if opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=0)
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=0)
    elif opt_name == 'sgd_m':
        optimizer = Sgd_m(model.parameters(), lr, weight_decay=0)
    elif opt_name =='adam_bn':
        optimizer =Adam_bn(model.parameters(), lr, weight_decay=0)
    elif opt_name =='adam_ini':
        optimizer =Adam_ini(model.parameters(), lr, weight_decay=0)
    elif opt_name =='adam_sbn':
        optimizer =Adam_Sbn(model.parameters(), lr, weight_decay=0)
    elif opt_name =='sgd_sbn':
        optimizer =SGD_Sbn(model.parameters(), lr, weight_decay=0)
    elif opt_name =='rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr, weight_decay=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

    

    model.train()

    print(account + '/' + datapath)
    if dataset_name=='imagenet':
        # train_data_root='/home/wangjzh/adam_optimizer/data/imagenet_new/data/ILSVRC2014_DET_train'
        # train_txt = '/home/wangjzh/adam_optimizer/data/imagenet_new/label/ImageNet_LT_train.txt'
        # test_txt = '/home/wangjzh/adam_optimizer/data/imagenet_new/label/ImageNet_LT_val.txt'
        # test_data_root='/home/wangjzh/adam_optimizer/data/imagenet_new/data/ILSVRC2014_DET_val'
        # train_dataset = imagenetLT_Dataset(train_data_root, train_txt, transform_train)
        # test_dataset = imagenetLT_Dataset(data_root, test_txt, transform_test)
        train_dataset = ImageNetLT_LMDB(root=account + '/' + datapath, version="imagenetlt_lt", train=True, transform=transform_train)
        print(len(train_dataset))
        train_dataset,label_list = select_top_n_classes_lmdb(train_dataset, 1000,save_path='train_data.png',cache_path='train_cache.pkl')
        test_dataset = ImageNetLT_LMDB(root=account + '/' + datapath, version="imagenetlt_lt", train=False, transform=transform_test)
        test_dataset,label_list = select_top_n_classes_lmdb(test_dataset, 1000,label_list,save_path='test_data.png',cache_path='test_cache.pkl')

        #分析类别分布
        # stats = analyze_class_distribution(train_dataset)
        # print("<<<<<<<<<<<<<<<训练集")
        # #打印统计信息
        # print_class_distribution_stats(stats, train_dataset)

        # stats = analyze_class_distribution(test_dataset)
        # print("<<<<<<<<<<<<<<<测试集")
        # #打印统计信息
        # print_class_distribution_stats(stats, test_dataset)
    
        log(f"训练集样本数: {len(train_dataset)}")
        log(f"测试集样本数: {len(test_dataset)}")
    
    else:
        train_dataset = CIFAR100LT(account + '/' + datapath, train=True, transform=transform_train)
        test_dataset = CIFAR100LT(account + '/' + datapath, train=False, transform=transform_test)
    # train_dataset = ImageNetLT(root=account + '/' + datapath, version="imagenetlt_lt", train=True, transform=transform_train)
    # # test_dataset = ImageNetLT(root=account + '/' + datapath, version="imagenetlt_lt", train=False, transform=transform_test)
    # ImageNetLTDatasetWithIndex.generate_index_file(
    # image_dir=account + '/' +'adam_optimizer/data/imagenet/imagenetlt',  # 你的训练图片目录
    # label_map_path=account + '/' +'adam_optimizer/data/imagenet/imagenetlt/class_map.txt',
    # output_json_path=account + '/' +'adam_optimizer/data/imagenet/imagenetlt_test_index.json'
    # )
    # test_dataset = ImageNetLTDatasetWithIndex(
    # index_file=account + '/' +'adam_optimizer/data/imagenet/imagenetlt_test_index.json',
    # transform=transform_test
    # )
    # # train_data = ImageNetLT(account + '/' + datapath, train=True, transform=transforms_c)
    # # test_data = ImageNetLT(account + '/' + datapath, train=False, transform=transforms_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nThreads,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=nThreads,pin_memory=True)

    # 🔸 模型保存目录（不同模型/优化器组合）
    save_dir = os.path.join(account, 'adam_machinism', 'model', 'adapoly',
                            f"{dataset_name}_{model_name}_{opt_name}_batch{batch_size}_lr{lr}_seed{seed}_1,1,1,1")
    os.makedirs(save_dir, exist_ok=True)

    train_loss = 100.
    # 🔹 保存初始模型并记录其性能
    init_model_eval_path = os.path.join(save_dir, 'epoch000.pth')
    torch.save(model.state_dict(), init_model_eval_path)
    print(f"[保存模型] 初始模型已保存到: {init_model_eval_path}")
    
    # # 🔹 初始模型评估
    acc, acc5, test_loss, test_class_loss = predict(test_loader, model)
    acc, acc5, train_loss, train_class_loss = predict(train_loader, model)
    train_class_loss = {i: 0.0 for i in range(n_lenth)}  # 可设为0或空，初始时不训练
    
    log_values = []
    for class_idx in range(n_lenth):
        log_values.append(train_class_loss.get(class_idx, 0.0))  # 初始为0
        log_values.append(test_class_loss.get(class_idx, 0.0))
    log_values.extend([train_loss, test_loss, acc.item(), acc5.item()])
    
    logger.record(log_values)
    logger.log()
    logger.write_log()
    for epoch in range(epoch_num):
        train_loss, train_class_loss = train(train_loader, model, criterion, optimizer, train_loss)
        #print(train_class_loss)
        if (epoch + 1) % 1 == 0:
            acc, acc5, test_loss, test_class_loss = predict(test_loader, model)

            log_values = []
            for class_idx in range(n_lenth):
                log_values.append(train_class_loss.get(class_idx, 0.0))
                log_values.append(test_class_loss.get(class_idx, 0.0))
    
            log_values.extend([train_loss, test_loss, acc.item(), acc5.item()])
            logger.record(log_values)
            logger.log()
            logger.write_log()
            
        scheduler.step()

        # 🔸 每隔 10 个 epoch 保存一次模型
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epoch_num:
            epoch_model_path = os.path.join(save_dir, f'epoch{epoch+1:03d}.pth')
            torch.save(model.state_dict(), epoch_model_path)
            logger.save_data()
            print(f"[保存模型] 第 {epoch + 1} 轮模型已保存到: {epoch_model_path}")

    logger.save_data()
    logger.plot_loss()

    # 🔸 最终模型保存
    final_model_path = os.path.join(save_dir, 'final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"[保存模型] 最终模型已保存到: {final_model_path}")
