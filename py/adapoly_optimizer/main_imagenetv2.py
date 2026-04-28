import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from collections import defaultdict
from util.visualizer import Visualizer
from cifar10_pair_sim import ImageNetLT, ImageNetLTDatasetWithIndex
from vit import vit_small_cifar_patch4_32, vit_base_cifar_patch4_32,vit_small_cifar_patch16_224,vit_base_cifar_patch16_224
from resnet import ResNetOrigin, BasicBlock, Bottleneck
from datetime import datetime
import time

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
# AMP
from torch.cuda.amp import autocast, GradScaler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cifar100')
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--model_name', type=str, default='vits')
    parser.add_argument('--opt_name', type=str, default='adam')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epoch_num', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--server_mode', action='store_true', default=False)
    parser.add_argument('--cuda_visible_devices', type=str, default='0')
    parser.add_argument('--account', type=str, default='/home/wangjzh')
    return parser.parse_args()

def accuracy(output, target, topk=(1,)):
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

def train(train_loader, model, criterion, optimizer, scaler):
    model.train()
    total_loss = []
    class_loss = defaultdict(lambda: [0.0, 0])
    log(f"  🔁 开始训练 batch，共 {len(train_loader)} 个 batch")
    for step, (input, target) in enumerate(train_loader):
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            output = model(input)
            loss = criterion(output, target)
            per_sample_loss = F.cross_entropy(output, target, reduction='none')
        for t, l in zip(target, per_sample_loss):
            class_loss[t.item()][0] += l.item()
            class_loss[t.item()][1] += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss.append(loss.item())
        if step % 20 == 0:
            log(f"    [Batch {step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
    avg_loss = np.mean(total_loss)
    class_avg = {k: v[0] / v[1] if v[1] else 0 for k, v in class_loss.items()}
    return avg_loss, class_avg


def predict(test_loader, model, criterion):
    model.eval()
    losses, top1s, top5s = [], [], []
    class_loss = defaultdict(lambda: [0.0, 0])
    log(f"  🔍 开始测试 batch，共 {len(test_loader)} 个 batch")
    with torch.no_grad():
        for step, (input, target) in enumerate(test_loader):
            input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
            with autocast():
                output = model(input)
                loss = criterion(output, target)
            per_sample_loss = F.cross_entropy(output, target, reduction='none')
            for t, l in zip(target, per_sample_loss):
                class_loss[t.item()][0] += l.item()
                class_loss[t.item()][1] += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 3))
            losses.append(loss.item())
            top1s.append(acc1.item())
            top5s.append(acc5.item())
            if step % 20 == 0:
                log(f"    [Batch {step+1}/{len(test_loader)}] Test Loss: {loss.item():.4f}, Top1: {acc1.item():.2f}%, Top3: {acc5.item():.2f}%")
    avg_loss = np.mean(losses)
    avg_top1 = np.mean(top1s)
    avg_top5 = np.mean(top5s)
    class_avg = {k: v[0] / v[1] if v[1] else 0 for k, v in class_loss.items()}
    return avg_top1, avg_top5, avg_loss, class_avg

import csv
from collections import Counter

def save_class_loss_to_csv(epoch_stats, class_order, save_path='train_log.csv'):
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 表头
        header = []
        for cid in class_order:
            header.append(f"class_{cid}_imagenet_train")
            header.append(f"class_{cid}_imagenet_test")
        header += ["train loss", "test loss", "acc", "acc5"]
        writer.writerow(header)

        # 每一轮数据
        for stat in epoch_stats:
            row = []
            d = {cid: (t, s) for cid, t, s in stat['class_losses']}
            for cid in class_order:
                t, s = d.get(cid, (0.0, 0.0))
                row.extend([f"{t:.4f}", f"{s:.4f}"])
            row.extend([f"{stat['train_loss']:.4f}", f"{stat['test_loss']:.4f}",
                        f"{stat['acc']:.2f}", f"{stat['acc5']:.2f}"])
            writer.writerow(row)

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    log(f"使用 GPU: {args.cuda_visible_devices}")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # 🗂️ 构造保存路径（支持 dataset/model/lr_bs/时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        "imagenet_logs",
        args.dataset_name,
        args.model_name,
        args.opt_name,
        f"bs{args.batch_size}_lr{args.lr}",
        timestamp
    )
    model_save_path = os.path.join(save_dir, "checkpoints")
    log_save_path = os.path.join(save_dir, "log")
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    log(f"📁 模型和日志将保存在: {save_dir}")

    # ✅ 初始化数据增强与模型
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = {
        'vits': lambda: vit_small_cifar_patch16_224(num_classes=1000),
        'vitb': lambda: vit_base_cifar_patch16_224(num_classes=1000),
        'resnet18': lambda: ResNetOrigin(BasicBlock, [2, 2, 2, 2], num_classes=1000),
        'resnet50': lambda: ResNetOrigin(Bottleneck, [3, 4, 6, 3], num_classes=1000),
        'vgg16bn': lambda: models.vgg16_bn(pretrained=False, num_classes=1000)
    }[args.model_name]()
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        log(f"🔧 使用 {torch.cuda.device_count()} 块 GPU 进行并行训练")
        model = nn.DataParallel(model)
    log(f"✅ 模型初始化完成: {args.model_name}")

    # ✅ 优化器、调度器、混合精度
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num)
    scaler = GradScaler()

    # ✅ 加载数据集
    train_dataset = ImageNetLT(
        root=os.path.join(args.account, args.datapath),
        version="imagenetlt_lt", train=True, transform=transform_train)
    test_dataset = ImageNetLT(
        root=os.path.join(args.account, args.datapath),
        version="imagenetlt_lt", train=False, transform=transform_test)

    log(f"训练集样本数: {len(train_dataset)}")
    log(f"测试集样本数: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.nThreads, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.nThreads, pin_memory=True, persistent_workers=True)

    logger = Visualizer(['train loss', 'test loss', 'acc', 'acc5'], 2,
                        log_path=log_save_path, name="log", label='run')

    # ✅ 类别频率排序
    from collections import Counter
    class_freq = Counter(train_dataset.targets)
    sorted_classes = [cid for cid, _ in sorted(class_freq.items(), key=lambda x: -x[1])]
    epoch_stats = []

    # ✅ 训练循环
    log("🚀 开始训练循环...")
    for epoch in range(args.epoch_num):
        start = time.time()
        log(f"[Epoch {epoch+1}/{args.epoch_num}] 开始训练")

        train_loss, train_class_loss = train(train_loader, model, criterion, optimizer, scaler)
        acc1, acc5, test_loss, test_class_loss = predict(test_loader, model, criterion)

        log(f"[Epoch {epoch+1}] ✅ 训练完成 - Loss: {train_loss:.4f}")
        log(f"[Epoch {epoch+1}] 🧪 测试结果 - Top1 Acc: {acc1:.2f}%, Top5 Acc: {acc5:.2f}%, Loss: {test_loss:.4f}")
        logger.record([train_loss, test_loss, acc1, acc5])
        logger.log()
        logger.write_log()
        scheduler.step()
        log(f"[Epoch {epoch+1}] ⏱️ Epoch耗时: {time.time() - start:.2f} 秒")

        # ✅ 记录类别损失
        class_losses = []
        for cid in sorted_classes:
            train_l = train_class_loss.get(cid, 0.0)
            test_l = test_class_loss.get(cid, 0.0)
            class_losses.append((cid, train_l, test_l))

        epoch_stats.append({
            "class_losses": class_losses,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "acc": acc1,
            "acc5": acc5
        })

        # ✅ 模型保存
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(model_save_path, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            log(f"[Epoch {epoch+1}] 💾 模型已保存: {model_path}")

    # ✅ 保存最终模型
    final_model_path = os.path.join(model_save_path, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    log(f"[✅] 训练结束，最终模型已保存为 {final_model_path}")

    # ✅ 保存 CSV 日志
    save_class_loss_to_csv(epoch_stats, sorted_classes,
                           save_path=os.path.join(log_save_path, "train_log.csv"))
    log(f"📄 所有 epoch 的类损失与指标已保存到 {os.path.join(log_save_path, 'train_log.csv')}")

if __name__ == '__main__':
    main()
