import random

import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR10, PhotoTour
from random import shuffle


def data_pair_generate(dataset, file_path):
    df_data_pair = pd.DataFrame()
    l = list(range(len(dataset)))
    shuffle(l)
    for i in range(len(dataset)):
        df_data_pair = df_data_pair.append(pd.Series([i, i, 1]), ignore_index=True)
        df_data_pair = df_data_pair.append(pd.Series([i, l[i], 0]), ignore_index=True)
        df_data_pair = df_data_pair.astype(int)
    df_data_pair.to_csv(file_path, )


def data_pair_generate_2(dataset, file_path):
    df_data_pair = pd.DataFrame()
    df_data_pair[0] = pd.Series(list(range(len(dataset))))
    df_data_pair[1] = pd.Series(list(range(len(dataset))))
    df_data_pair[2] = 1
    df_data_pair_2 = pd.DataFrame()
    df_data_pair_2[0] = pd.Series(list(range(len(dataset))))
    l = list(range(len(dataset)))
    shuffle(l)
    df_data_pair_2[1] = pd.Series(l)
    df_data_pair_2[2] = 0
    df_data_pair = pd.concat([df_data_pair, df_data_pair_2]).astype(int)
    df_data_pair.to_csv(file_path, index=None)


def data_triplet_generate(dataset, file_path):
    df_data_pair = pd.DataFrame()
    df_data_pair[0] = pd.Series(list(range(len(dataset))))
    df_data_pair[1] = pd.Series(list(range(len(dataset))))
    l = list(range(len(dataset)))
    shuffle(l)
    df_data_pair[2] = pd.Series(l)
    df_data_pair.to_csv(file_path, index=None)


if __name__ == '__main__':
    # data = CIFAR10('../asset/deepbit/cifar10/', download=False, transform=None)
    data = PhotoTour('/bixl_ms/hy/deepbit/dataset/photo_tour/', 'liberty', train=True, download=False, transform=None)
    data_pair_generate_2(data, '../asset/deepbit/brown_dataset/phototour_pairs.csv')
