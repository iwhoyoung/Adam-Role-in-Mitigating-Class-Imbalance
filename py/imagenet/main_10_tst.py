import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import csv
from collections import Counter
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from torch.utils.data import Subset
from vit import vit_base_cifar_patch4_32, vit_cifar_patch4_32, vit_cifar_patch4_32_depth4, vit_cifar_patch4_256, vit_custom_cifar_32, vit_small_cifar_patch4_32, vit_tiny_cifar_patch4_32,vit_small_cifar_patch16_224,vit_base_cifar_patch16_224,vit_small_cifar_patch16_224,vit_base_cifar_patch16_224
from adassd_gamma import Adam_bn,Adam_ini
try:
    from timm.data import TimmDatasetTar
except ImportError:
    # for higher version of timm
    from timm.data import ImageDataset as TimmDatasetTar
    

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with Category Loss Tracking')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', default=None, type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--enable_wandb', action='store_true', help="enable wandb logging")
parser.add_argument('--output_dir', type=str, default='output', help="output dir")
parser.add_argument('--category-loss', default=True, help="track and save category-wise loss")
parser.add_argument('--optimizer', default=None, type=str)
parser.add_argument('--scheduler', default=None, type=str)

best_acc1 = 0


class CategoryLossTracker:
    """跟踪每个类别的损失"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.category_loss = [0.0] * self.num_classes
        self.category_count = [0] * self.num_classes
        
    def update(self, targets, losses):
        """
        更新类别损失
        targets: 批量样本的类别标签
        losses: 批量样本的损失
        """
        for target, loss in zip(targets, losses):
            target = int(target)
            self.category_loss[target] += loss.item()
            self.category_count[target] += 1
            
    def get_avg_loss(self):
        """获取每个类别的平均损失"""
        avg_losses = []
        for i in range(self.num_classes):
            if self.category_count[i] > 0:
                avg_losses.append(self.category_loss[i] / self.category_count[i])
            else:
                avg_losses.append(0.0)  # 避免除零错误
        return avg_losses

    def get_grouped_avg_loss(self, group_map, num_groups=10):
        """
        获取每组（如10组）的加权平均损失，group_map[i] 表示类别i属于哪一组
        """
        group_loss = [0.0] * num_groups
        group_count = [0] * num_groups

        for cls in range(self.num_classes):
            group = group_map[cls]
            loss = self.category_loss[cls]
            count = self.category_count[cls]
            group_loss[group] += loss
            group_count[group] += count

        group_avg = []
        for i in range(num_groups):
            if group_count[i] > 0:
                group_avg.append(group_loss[i] / group_count[i])
            else:
                group_avg.append(0.0)
        return group_avg


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
        
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
    # create log
    # only enable wandb log in main process
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        if args.enable_wandb:
            try:
                import wandb
                wandb.init(project="pytorch_imagenet")
            except ImportError:
                raise ImportError("Please install wandb to enable wandb logging")
            
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model_name =args.arch
        if model_name=='vgg16bn':
            model = models.vgg16_bn(num_classes=1000)
        elif model_name=='resnet18':
            model = models.resnet18(num_classes=1000)  
        elif model_name=='resnet50':
            model = models.resnet50(num_classes=1000)  
        elif model_name=='vits':
            model = vit_small_cifar_patch16_224(num_classes=1000)
        elif model_name=='vitb':
            model = vit_base_cifar_patch16_224(num_classes=1000)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # 获取类别数量
    if hasattr(model, 'fc'):
        num_classes = model.fc.out_features
    elif hasattr(model, 'classifier'):
        num_classes = model.classifier.out_features
    else:
        # 默认设置为1000，如果无法获取实际类别数
        num_classes = 1000
        print("Warning: Could not determine number of classes, defaulting to 1000")

    class_freqs = [1300 / (i + 1) for i in range(num_classes)]  # 类频递减
    sorted_classes = sorted(range(num_classes), key=lambda i: -class_freqs[i])  # 高频在前

    # 每组100类，总共10组，生成 group_map
    group_map = [0] * num_classes  # group_map[i] 表示第i类属于哪一组
    mmap=[[0,1],[1,3],[3,7],[7,14],[14,27],[27,52],[52,100],[100,194],[194,391],[391,1000]]
    for group_id in range(10):
        start = mmap[group_id][0]
        end = mmap[group_id][1]
        for cls_id in sorted_classes[start:end]:
            print(group_id)
            group_map[cls_id] = group_id
        break
    import sys
    #sys.exit()
    # 创建类别损失跟踪器
    train_category_tracker = CategoryLossTracker(num_classes)
    val_category_tracker = CategoryLossTracker(num_classes)
    
    # 准备CSV文件
    # if args.category_loss and (not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)):
    #     os.makedirs(args.output_dir, exist_ok=True)
    #     train_csv_path = os.path.join(args.output_dir, 'train_category_loss_'+str(args.lr)+'_'+str(args.batch_size)+'_'+str(args.optimizer)+'_'+str(args.arch)+str(args.scheduler)+'.csv')
    #     val_csv_path = os.path.join(args.output_dir, 'val_category_loss_'+str(args.lr)+'_'+str(args.batch_size)+'_'+str(args.optimizer)+'_'+str(args.arch)+str(args.scheduler)+'.csv')
        
    #     # 写入CSV表头
    #     with open(train_csv_path, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         #header = ['epoch'] + [f'class_{i}' for i in range(num_classes)]
    #         header = ['epoch'] + [f'group_{i}' for i in range(10)]
    #         writer.writerow(header)
            
    #     with open(val_csv_path, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         header = ['epoch'] + [f'class_{i}' for i in range(num_classes)]
    #         writer.writerow(header)
    if args.category_loss and (not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)):
        os.makedirs(args.output_dir, exist_ok=True)
        # 训练集CSV路径（包含总loss和acc）
        train_csv_path = os.path.join(args.output_dir, 'train_category_loss_'+str(args.lr)+'_'+str(args.batch_size)+'_'+str(args.optimizer)+'_'+str(args.arch)+str(args.scheduler)+'_'+str(args.epochs)+'_'+'.csv')
        # 验证集CSV路径（包含总loss和acc）
        val_csv_path = os.path.join(args.output_dir, 'val_category_loss_'+str(args.lr)+'_'+str(args.batch_size)+'_'+str(args.optimizer)+'_'+str(args.arch)+str(args.scheduler)+'_'+str(args.epochs)+'_'+'.csv')
        
        # 写入CSV表头（新增总loss、acc1、acc5）
        with open(train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 表头：epoch + 总loss + acc1 + acc5 + 各组损失
            header = ['epoch', 'total_loss', 'acc1', 'acc5'] + [f'group_{i}' for i in range(10)]
            writer.writerow(header)
            
        with open(val_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 表头：epoch + 总loss + acc1 + acc5 + 各类损失
            header = ['epoch', 'total_loss', 'acc1', 'acc5'] + [f'class_{i}' for i in range(num_classes)]
            writer.writerow(header)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    opt_name=args.optimizer
    lr = args.lr
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=0)
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=0)
    elif opt_name =='adam_bn':
        optimizer =Adam_bn(model.parameters(), lr, weight_decay=0)
    elif opt_name =='adam_ini':
        optimizer =Adam_ini(model.parameters(), lr, weight_decay=0)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.scheduler=='cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[60,80],gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        # train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        train_dataset = datasets.FakeData(128116, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        if os.path.exists(os.path.join(args.data, 'train')):
            traindir = os.path.join(args.data, 'train')
            valdir = os.path.join(args.data, 'val')
        
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            dataset_tar_t = TimmDatasetTar
            
            train_dataset = dataset_tar_t(
                os.path.join(args.data, 'train.tar'), 
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            val_dataset = dataset_tar_t(
                os.path.join(args.data, 'val.tar'), 
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
    all_labels = []
    for data, label in train_dataset:  # 假设每个样本是 (数据, 标签) 元组
        all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    
    ## 方式2：用 DataLoader 批量提取（适用于大数据集，更高效）
    # dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    # all_labels = []
    # for batch_data, batch_labels in dataloader:
    #     all_labels.extend(batch_labels.numpy().tolist())  # 转为列表并合并
    
    # 2. 统计类别数量
    class_counts = Counter(all_labels)
    
    # 3. 输出结果
    print("每类数据的数量统计（Counter）：")
    print(class_counts)  # 格式：{类别0: 数量0, 类别1: 数量1, ...}
    
    # （可选）转为 DataFrame 更清晰
    class_stats = pd.DataFrame({
        "类别": list(class_counts.keys()),
        "数量": list(class_counts.values()),
        "占比(%)": [round(count / sum(class_counts.values()) * 100, 2) for count in class_counts.values()]
    })
    print("\n带占比的统计表格：")
    print(class_stats)
    import sys
    sys.exit()
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
    print('start_loader')

    from timm.data.loader import create_loader
    print(args.batch_size)
    try:
        # only works if gpu present on machine
        train_loader = create_loader( train_dataset, (3, 224, 224), args.batch_size,is_training=True,pin_memory=True)
    except:
        train_loader = create_loader( train_dataset,(3, 224, 224),args.batch_size, use_prefetcher=False,is_training=True,pin_memory=True)
        
    try:
        # only works if gpu present on machine
        val_loader = create_loader(val_dataset,  (3, 224, 224), args.batch_size)
    except:
        val_loader = create_loader( val_dataset, (3, 224, 224), args.batch_size, use_prefetcher=False)

    if args.evaluate:
        epoch=0
        validate(train_loader, model, criterion, args, train_category_tracker)
        train_losses = train_category_tracker.get_avg_loss()
        #val_losses = val_category_tracker.get_avg_loss()
        
        with open(train_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch] + train_losses)
        return
   #########
    if args.distributed:
        train_sampler.set_epoch(epoch)
    # 重置类别损失跟踪器
    train_category_tracker.reset()
    val_category_tracker.reset()
    
    # train for one epoch
    #train_loss, train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, device, args, train_category_tracker)

    # evaluate on validation set
    valid_loss, acc1, acc5 = validate(train_loader, model, criterion, args, val_category_tracker)
    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if args.category_loss and (not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)):
        # 执行验证（用train_loader暂代，建议后续改为val_loader）
        valid_loss, acc1, acc5 = validate(train_loader, model, criterion, args, val_category_tracker)
        val_losses_grouped = val_category_tracker.get_grouped_avg_loss(group_map)
        # 写入：epoch=-1 + valid_loss + acc1.item() + acc5.item() + 10组损失（共15列）
        with open(train_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([-1, valid_loss, acc1.item(), acc5.item()] + val_losses_grouped)
                
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # 重置类别损失跟踪器
        train_category_tracker.reset()
        val_category_tracker.reset()
        
        # train for one epoch
        train_loss, train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, device, args, train_category_tracker)

        # evaluate on validation set
        #valid_loss, acc1, acc5 = validate(val_loader, model, criterion, args, val_category_tracker)
        
        scheduler.step()
        acc1=train_acc1
        acc5=train_acc5
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            # 保存类别损失到CSV
            # if args.category_loss:
            #     print('start class loss')
            #     train_losses_grouped = train_category_tracker.get_grouped_avg_loss(group_map)
            #     with open(train_csv_path, 'a', newline='') as f:
            #         writer = csv.writer(f)
            #         writer.writerow([epoch] + train_losses_grouped)
            if args.category_loss:
                print('start class loss')
                train_losses_grouped = train_category_tracker.get_grouped_avg_loss(group_map)
                # 写入数据：epoch + 总loss + acc1 + acc5 + 各组损失
                with open(train_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_loss, train_acc1.item(), train_acc5.item()] + train_losses_grouped)


                    
                # with open(val_csv_path, 'a', newline='') as f:
                #     writer = csv.writer(f)
                #     writer.writerow([epoch] + val_losses)
                    
                print(f"Epoch {epoch}: Category losses saved to CSV")
            
            # only enable wandb log in main process
            if args.enable_wandb:
                wandb.log({
                    "train/loss": train_loss,
                    "train/acc1": train_acc1,
                    "train/acc5": train_acc5,
                    "val/loss": valid_loss,
                    "val/acc1": acc1,
                    "val/acc5": valid_acc5,
                })
            path='output_'+str(args.lr)+'_'+str(args.batch_size)+'_'+str(args.optimizer)+'_'+str(args.arch)+str(args.scheduler)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, save_dir='output_'+str(args.lr)+'_'+str(args.batch_size)+'_'+str(args.optimizer)+'_'+str(args.arch)+str(args.scheduler))
            
    print("Training finished")


def train(train_loader, model, criterion, optimizer, epoch, device, args, category_tracker):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        
        # 计算每个样本的损失（用于类别统计）
        if args.category_loss:
            # 获取每个样本的损失
            sample_losses = nn.functional.cross_entropy(output, target, reduction='none')
            # 更新类别损失跟踪器
            category_tracker.update(target, sample_losses)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
    
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args, category_tracker):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)
                
                # 计算每个样本的损失（用于类别统计）
                if args.category_loss:
                    # 获取每个样本的损失
                    sample_losses = nn.functional.cross_entropy(output, target, reduction='none')
                    # 更新类别损失跟踪器
                    category_tracker.update(target, sample_losses)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, save_dir='output', filename='checkpoint.pth.tar'):
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'model_best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
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
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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


if __name__ == '__main__':
    main()