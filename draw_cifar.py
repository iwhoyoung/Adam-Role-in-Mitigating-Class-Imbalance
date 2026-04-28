import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def find_columns(df, suffix):
    #return [col for col in df.columns if col.startswith(suffix)]
    return [col for col in df.columns][:]

def plot_series(df1, df2, columns, labels, ylabel, title, output_dir):
    plt.figure(figsize=(10, 6))
    for col in columns:
        plt.plot(df1[col].values, label=f"{labels[0]} {col}")
        plt.plot(df2[col].values, label=f"{labels[1]} {col}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()

import matplotlib.cm as cm
import numpy as np

import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

def process_loss_csv(input_csv_path,model_name):
    # 1. 读取输入CSV
    df1 = pd.read_csv(input_csv_path)
    df2 = pd.read_csv(f"/home/wangjzh/adam_optimizer/py/imagenet/output/train_ini_category_loss_0.5_512_sgd_{model_name}.csv")
    df = pd.concat([df2,df1], ignore_index=True)
    print(f"成功读取文件: {input_csv_path}")
    print(f"数据形状: {df.shape}，包含 {df.shape[0]} 个epoch，{df.shape[1]-1} 个类别")

    # 2. 计算每个类别的样本量 (1300/i，i从1到1000)
    num_classes = 1000
    n_i = [int(1300 / i) for i in range(1, num_classes + 1)]  # 索引0对应第1类，索引999对应第1000类

    # 3. 计算总样本量和每组目标样本量
    total_samples = sum(n_i)
    #target_per_group = total_samples / 10
    target_per_group = 810
    print(f"总样本量: {total_samples:.2f}，每组目标样本量: {target_per_group:.2f}")

    # 4. 按样本量从高到低（即原始顺序）划分10组
    groups = []  # 存储每组包含的类别索引（0-based）
    current_sum = 0.0
    current_group = []

    for idx in range(num_classes):
        current_sum += n_i[idx]
        current_group.append(idx)
        
        # 前9组达到目标时划分，最后一组包含剩余所有
        if len(groups) < 9 and current_sum >= target_per_group:
            groups.append(current_group)
            print(f"组{len(groups)}: 包含{len(current_group)}个类别，总样本量{current_sum:.2f}")
            current_sum = 0.0
            current_group = []

    # 添加最后一组
    groups.append(current_group)
    print(f"组10: 包含{len(current_group)}个类别，总样本量{sum(n_i[idx] for idx in current_group):.2f}")

    # 5. 计算每组的加权平均loss（权重为样本量）
    new_df = pd.DataFrame({"epoch": df["epoch"]})  # 保留epoch列

    for group_idx, group in enumerate(groups, 1):
        # 获取组内所有类别的列名（假设类别列名为class_1到class_1000）
        class_cols = [f"class_{idx }" for idx in group]  # idx是0-based，对应class idx+1
        
        # 组内每个类别的样本量（权重）
        weights = [n_i[idx] for idx in group]
        
        # 计算加权和：每个epoch下，组内所有类别的loss乘以权重后求和
        weighted_sum = df[class_cols].mul(weights).sum(axis=1)
        
        # 组内总样本量（用于归一化）
        group_total = sum(weights)
        
        # 加权平均loss
        new_df[f"group_{group_idx}_loss"] = weighted_sum / group_total

    # 6. 保存结果到新CSV
    output_csv = input_csv_path.replace(".csv", "_10.csv")
    new_df.to_csv(output_csv, index=False)
    print(f"结果已保存至: {output_csv}")
    return output_csv

# 示例调用（请替换为你的CSV路径）
# process_loss_csv("your_loss_data.csv")

def plot_top5_last5(df1,suffix, label1, label2, ylabel, output_dir, csv1_name, model_name,opt_name,tmp_r=10):
    cols1 = find_columns(df1, suffix)
    #cols2 = find_columns(df2, suffix)
    #cols2=cols1
    common_cols = list(set(cols1))

    if len(common_cols) < 10:
        print(f"Not enough columns with suffix '{suffix}' to plot top 5 and last 5 classes.")
        return

    # 提取并按 class index 排序
    def extract_class_idx(col_name):
        try:
            return int(col_name.split('_')[1])
        except:
            return -1
    
    common_cols_sorted = sorted(common_cols, key=extract_class_idx)[1:]
    # 支持任意选择方式：等间隔 or 自定义
    # selected_indices = np.linspace(0, 9, 5, dtype=int)
    selected_indices = np.linspace(0, 99, 10, dtype=int)
    #selected_indices = [0]
    selected_cols = [f'class_{i}_r-'+str(tmp_r)+'_train' for i in selected_indices]
    # selected_cols = [f'group_{i}' for i in selected_indices]
    #print(selected_cols)
    class_indices = [extract_class_idx(col) for col in selected_cols]
    #print(class_indices)
    colors = [cm.viridis(idx / 100) for idx in class_indices]  # 颜色映射基于 class index

    # 统一 Y 轴范围
    all_values = []
    for col in selected_cols:
        all_values.extend(df1[col].values)
    #print(df1)
    
    y_min, y_max = -0.2,10

    def plot(df, filename_suffix):
        plt.figure(figsize=(11, 8))
        for i, col in enumerate(selected_cols):
            # 获取原始数据
            values = df[col].values[:].copy()  # 使用copy()避免修改原数据
            
            # 构建对应的test列名（假设命名规则是将_train替换为_test）
            test_col = col.replace("_train", "_test")
            
            # 如果test列存在，则用其第一个step数据替换
            if test_col in df.columns:
                values[0] = df[test_col].values[0]
            
            # 绘制修改后的数据
            plt.plot(values, color=colors[i], linewidth=10)
            
        ax = plt.gca()
        # 设置X轴主刻度间隔为2
        x_major_locator = ticker.MultipleLocator(50)  # 主刻度间隔为2
        ax.xaxis.set_major_locator(x_major_locator)
        
        # 设置Y轴主刻度间隔为0.5
        y_major_locator = ticker.MultipleLocator(2)  # 主刻度间隔为0.5
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xlabel("Epoch", fontsize=50)
        plt.ylabel("Train Loss", fontsize=50)
        
        plt.xticks(fontsize=50) #16
        plt.yticks(fontsize=50)
        plt.ylim(y_min, y_max)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        filename = f"{model_name}_{filename_suffix}_top5_last5_{suffix}_liner_v2.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        #print(f"Saved plot to {filepath}")

    # 分别画 SGD 和 Adam 图，统一 y 轴范围
    plot(df1, opt_name)
    #plot(df2, 'sgd')

def plot_accuracy(csv1, csv2, output_dir):
    # Load data
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # # 1. Loss curves for columns ending with 'r-100_train'
    # loss_cols1 = set(find_columns(df1, 'r-100_train'))
    # loss_cols2 = set(find_columns(df2, 'r-100_train'))
    # common_loss = sorted(list(loss_cols1 & loss_cols2))
    # if common_loss:
    #     plot_series(df1, df2, common_loss, ['Adam', 'SGD'], 'Loss', 'Loss Convergence', output_dir)
    # else:
    #     print("No matching 'r-100_train' columns found.")

    # 2. Accuracy (acc) columns
    acc_cols1 = set(find_columns(df1, 'acc'))
    acc_cols2 = set(find_columns(df2, 'acc'))
    common_acc = sorted(list(acc_cols1 & acc_cols2))
    if common_acc:
        plot_series(df1, df2, common_acc, ['Adam', 'SGD'], 'Accuracy', 'Accuracy Curve', output_dir)
    else:
        print("No matching 'acc' columns found.")

def plot_loss_and_accuracy(csv1, output_dir,model_name,opt_name,tmp_r=10):
    # Load data
    df1 = pd.read_csv(csv1)[:]
    #df2 = pd.read_csv(csv2)

    os.makedirs(output_dir, exist_ok=True)

    # Get base name without extension
    csv1_name = os.path.splitext(os.path.basename(csv1))[0]

    # Plot acc
    #plot_top5_last5(df1, df2, 'acc', 'Adam', 'SGD', 'Accuracy', output_dir, csv1_name)

    # Plot acc5
    #plot_top5_last5(df1, df2, 'acc5', 'Adam', 'SGD', 'Top-5 Accuracy', output_dir, csv1_name)

    # Plot loss (optional: uncomment if needed)
    #plot_top5_last5(df1,  '_train', 'Adam', 'SGD', 'Loss', output_dir, csv1_name,model_name,opt_name)
    plot_top5_last5(df1,  'group', 'Adam', 'SGD', 'Loss', output_dir, csv1_name,model_name,opt_name,tmp_r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot top-5 and last-5 class curves.')
    parser.add_argument('--csv1', default='log/select_result/adapolycifar_r10_resnet18_adam_batch256_200e_lr0.001_seed0.csv', help='Path to first CSV file')
    parser.add_argument('--csv2', default='log/select_result/adapolycifar_r10_resnet18_sgd_batch256_200e_lr0.5_seed0.csv', help='Path to second CSV file')
    parser.add_argument('--output_dir', default='plots/acc_imagenet', help='Directory to save output figures')
    args = parser.parse_args()
    [
    {"name":"vgg16bn","sgd": 0.1,  "adam": 0.0005, "adam_bn": 1e-5,   "adam_ini": 0.05,   "adam_sbn": 0.005},
    {"name":"resnet18","sgd": 0.5,  "adam": 0.001,  "adam_bn": 0.0005, "adam_ini": 0.05,   "adam_sbn": 0.01},
    {"name":"resnet50","sgd": 0.5,  "adam": 0.001,  "adam_bn": 0.0001, "adam_ini": 0.5,   "adam_sbn": 0.01},
    {"name":"vitb","sgd": 0.05, "adam": 5e-5,   "adam_bn": 5e-6,  "adam_ini": 0.005,   "adam_sbn": 0.0005},
    {"name":"vits","sgd": 0.05, "adam": 5e-5,   "adam_bn": 1e-5,  "adam_ini": 0.005,   "adam_sbn": 0.001},
    ]
    
    cifar_BEST_LRS = {
    # "resnet18_cifar_r10": {"sgd": 1.0,  "adam": 0.001,  "adam_bn": 0.0005, "adam_ini": 0.05,   "adam_sbn": 0.01,   "sgd_m": 1.0},
    "resnet50_cifar_r10": {"sgd": 1.0,  "adam": 0.001,  "adam_bn": 0.0001, "adam_ini": 0.5,   "adam_sbn": 0.005,   "sgd_m": 0.5},
    # "vgg16bn_cifar_r10":  {"sgd": 0.1,  "adam": 0.0005, "adam_bn": 1e-5,   "adam_ini": 0.01,   "adam_sbn": 0.001,   "sgd_m": 0.05},
    "vitb_cifar_r10":     {"sgd": 0.05, "adam": 5e-5,   "adam_bn": 5e-6,  "adam_ini": 0.005,   "adam_sbn": 0.0005,   "sgd_m": 0.05},
    # "vits_cifar_r10":     {"sgd": 0.05, "adam": 5e-5,   "adam_bn": 1e-5,  "adam_ini": 0.005,   "adam_sbn": 0.001,   "sgd_m": 0.05},
    }
    # cifar_BEST_LRS = {
    # "resnet18_cifar_r10": {"rmsprop": 5e-4},
    # "vits_cifar_r10":     {"rmsprop": 5e-5},
    # }
    imagenet_best_lrs =  {
    "resnet18": {"sgd": 0.5,  "adam": 0.001,  "adam_bn": 0.0001, "adam_ini": 0.05,   "adam_sbn": 0.005},
    "resnet50": {"sgd": 0.1,"adam": 0.001,  "adam_bn": 0.0001, "adam_ini": 0.05,   "adam_sbn": 0.005},
    "vgg16bn":  {"sgd": 0.1,  "adam": 0.0001,  "adam_bn": 2e-5, "adam_ini": 0.01,   "adam_sbn": 0.002},
    "vitb":     {"sgd": 0.1, "adam": 5e-4,   "adam_bn": 5e-5,  "adam_ini": 0.01,   "adam_sbn": 0.005},
    "vits":     {"sgd": 0.1,"adam": 5e-4,   "adam_bn": 0.0001,  "adam_ini": 0.01,   "adam_sbn": 0.005},
    }

    # best_lrs =  {
    # "resnet18_imagenet": {"adam": 0.001,"adam_bn":1e-5},
    # }
    # csv_path = f"/home/wangjzh/adam_optimizer/py/imagenet/output/train_category_loss_0.005_51210.csv"
    # plot_loss_and_accuracy(csv_path, args.output_dir,"resnet18_imagenet" , "adam")
    #model_name='resnet18'
    #optimizer='adam_bn'
    #csv_path = f"/home/wangjzh/adam_optimizer/py/imagenet/output/train_category_loss_0.0005_512_adam_bn_resnet18cosine200.csv"
    #csv_path = process_loss_csv(csv_path,model_name)
    #plot_loss_and_accuracy(csv_path, args.output_dir, model_name, optimizer)
    # model_name='resnet18'
    # optimizer='adam_bn_cosine'
    # csv_path = f"/home/wangjzh/adam_optimizer/py/imagenet/output/train_category_loss_0.0005_512_adam_bn_resnet18cosine.csv"
    #csv_path = process_loss_csv(csv_path,model_name)
    #plot_loss_and_accuracy(csv_path, args.output_dir, model_name, optimizer)

    # for model_name, optim_dict in imagenet_best_lrs.items():
    #     for optimizer, lr in optim_dict.items():
    #         model_short = model_name.split("_")[0]  # 如 resnet18、vgg、vitb
    #         if model_short!='vgg16bn':continue
    #         csv_path = f"/home/wangjzh/adam_optimizer/py/imagenet/output/train_category_loss_{lr}_512_{optimizer}_{model_name}cosine_200_v2.csv"
    #         #csv_path = process_loss_csv(csv_path,model_name)
    #         plot_loss_and_accuracy(csv_path, args.output_dir, model_name, optimizer)

    best_lrs =  {
    "resnet18_cifar_r100": {"adam": 0.001},
    #"vgg16bn_cifar_r100":  {"adam": 0.0005},
    }
    
    for model_name, optim_dict in cifar_BEST_LRS.items():
        #if model_name=='resnet18_cifar_r10' or model_name=='vits_cifar_r10':
        #if True:
        for optimizer, lr in optim_dict.items():
            # if optimizer!="sgd_m":
                # continue
            model_short = model_name.split("_")[0]  # 如 resnet18、vgg、vitb
            # csv_path = f"log/adapolycifar_r10_{model_short}_{optimizer}_batch256_200e_lr{lr}_seed0.csv"
            # plot_loss_and_accuracy(csv_path,'plots/acc_r10', model_name, optimizer)
            # csv_path = f"log/adapolycifar_r10_{model_short}_{optimizer}_batch256_200e_lr{lr}_seed0.csv"
            # plot_loss_and_accuracy(csv_path,'plots/acc_r10', model_name, optimizer)
            #print(model_short)

            try:
                csv_path = f"log/adapolycifar_r10_{model_short}_{optimizer}_batch256_200e_lr{lr}_seed0.csv"
                print(csv_path)
                plot_loss_and_accuracy(csv_path, 'plots/acc_r10_1119', model_name, optimizer,10)
            except:
                print('error:',csv_path)
                csv_path = f"log/adapolycifar_r10_{model_short}_{optimizer}_batch256_200e_lr{lr}_seed7.csv"
                print(csv_path)
                plot_loss_and_accuracy(csv_path, 'plots/acc_r10_1119', model_name, optimizer,10)
            # try:
            #     csv_path = f"log/adapolycifar_r-20_{model_short}_{optimizer}_batch256_200e_lr{lr}_seed0.csv"
            #     plot_loss_and_accuracy(csv_path, 'plots/1012_acc_r20', model_name, optimizer,20)
            # except:
            #     csv_path = f"log/adapolycifar_r-20_{model_short}_{optimizer}_batch256_200e_lr{lr}_seed7.csv"
            #     plot_loss_and_accuracy(csv_path, 'plots/1012_acc_r20', model_name, optimizer,20)
                # try:
                #     try:
                #         csv_path = f"log/adapolycifar_r-50_{model_short}_{optimizer}_batch256_200e_lr{lr}_seed0.csv"
                #         plot_loss_and_accuracy(csv_path, 'plots/0909_acc_r50', model_name, optimizer,50)
                #     except:
                #         csv_path = f"log/adapolycifar_r-50_{model_short}_{optimizer}_batch256_200e_lr{lr}_seed7.csv"
                #         plot_loss_and_accuracy(csv_path, 'plots/0909_acc_r50', model_name, optimizer,50)
                # except:
                #     print('error:',csv_path)
                # print(csv_path)
                # try:
                #     try:
                #         csv_path = f"log/adapolycifar_r-100_{model_short}_{optimizer}_batch256_200e_lr{lr}_seed0.csv"
                #         plot_loss_and_accuracy(csv_path, 'plots/0909_acc_r100', model_name, optimizer,100)
                #     except:
                #         csv_path = f"log/adapolycifar_r-100_{model_short}_{optimizer}_batch256_200e_lr{lr}_seed7.csv"
                #         plot_loss_and_accuracy(csv_path, 'plots/0909_acc_r100', model_name, optimizer,100)
                # except:
                #     print('error:',csv_path)
                # print(csv_path)



    
