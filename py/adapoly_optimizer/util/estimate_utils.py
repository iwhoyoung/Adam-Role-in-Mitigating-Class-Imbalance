import operator

# deepbit performance index
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd

def get_fpr_at_95_recall(labels, scores):
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    temp = zip(labels, scores)
    # operator.itemgetter(1)按照第二个元素的次序对元组进行排序，reverse=True是逆序，即按照从大到小的顺序排列
    # sorted_scores.sort(key=operator.itemgetter(1), reverse=True)
    sorted_scores = sorted(temp, key=operator.itemgetter(1), reverse=True)

    # Compute error rate
    # n_match表示测试集正样本数目
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_point * n_match
    tp = 0
    count = 0
    for label, score in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
        if tp >= n_thresh:
            break
    return float(count - tp) / (len(sorted_scores) - n_match)


# calculate mean and standard deviate of dataset
def get_mean_and_std(train_data):
    """
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    """
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    print("mean:", list(mean.numpy())," std:",list(std.numpy()))
    return list(mean.numpy()), list(std.numpy())


def gen_performance_plot(data, x_label, y_label, title):
    """
    generate a 2-D figure
    :param data: the data to show
    :param x_label: x_axis name
    :param y_label: y_axis name
    :param title: the title
    :return:
    """
    plt.plot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.grid()
    # plt.legend()
    plt.savefig("performance.jpg")
    plt.clf()

def cal_sim_norm_model(model_a, model_b,path):
    names=[]
    sims=[]
    norm_values1=[]
    norm_values2=[]
    for (key_a,value_a),(key_b,value_b) in zip(model_a.items(), model_b.items()):
        if 'num' in key_b or 'bn' in key_b or 'downsample.1' in key_b:
            continue
        names.append(key_b)
        vector_a = value_a.reshape(-1)
        vector_b = value_b.reshape(-1)
        # sim = torch.abs(torch.sum(F.normalize(vector_a,dim=0)*F.normalize(vector_b,dim=0)))
        norm_value1 = torch.sum(vector_a**2).sqrt()
        norm_value2 = torch.sum(vector_b**2).sqrt()
        sim = torch.sum(vector_a/norm_value1 * (vector_b/norm_value2))
        # sim = torch.sum(F.normalize(vector_a,dim=0)*F.normalize(vector_b,dim=0))
        sims.append(sim.item())
        norm_values1.append(norm_value1.item())
        norm_values2.append(norm_value2.item())
    df_datas = pd.DataFrame()
    df_datas[0]=names
    df_datas[1]=sims
    df_datas[2]=norm_values1
    df_datas[3]=norm_values2
    df_datas.to_csv(path + '.csv')
    return sims

def get_param(model, path, key):
    model = model.state_dict()
    names=[]
    df_datas = pd.DataFrame()
    for (key_a,value_a) in model.items():
        if not key in key_a:
            continue
        names.append(key_a)
        vector_a = value_a.reshape(-1)
        # sim = torch.abs(torch.sum(F.normalize(vector_a,dim=0)*F.normalize(vector_b,dim=0)))
        df_params = df_params.append(pd.Series(vector_a.cpu().numpy()), ignore_index=True)
    df_datas.to_csv(path + '.csv')
    return df_params

def record_param_name(model_a, path, filter_strs=[]):
    names=[]
    is_continue = False
    for (key_a,value_a) in model_a.items():
        is_continue = False
        for filter_str in filter_strs:
            if filter_str in key_a:
                is_continue = True
        if is_continue:
            continue
        names.append(key_a)
    df_datas = pd.DataFrame()
    df_datas[0]=names
    df_datas.to_csv(path + '.csv')
    return names

def cal_norm_layer(model, layer_index=None, layer_name=None, path='./norm_weights'):
    # for i,(key,value) in enumerate(model.items()):
    i=-1
    for key,value in model.items():
        if 'num' in key or 'bn' in key or 'downsample.1' in key:
            continue
        i = i+1
        if layer_name == key or layer_index==i:
            vector = value.reshape(-1)
            norm_vector = F.normalize(vector,dim=0)
            df_datas = pd.DataFrame()
            # a = norm_vector.numpy()
            df_datas[0]=norm_vector.cpu().numpy()
            if layer_index !=None:
                df_datas.to_csv(path + ('_%d'%layer_index)+ '.csv')
            else:
                df_datas.to_csv(path + ('_%s'%layer_name)+ '.csv')
            return norm_vector

if __name__ == '__main__':
    pass