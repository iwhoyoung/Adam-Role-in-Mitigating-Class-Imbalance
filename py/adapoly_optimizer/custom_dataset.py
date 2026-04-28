import glob
import random
import pandas as pd
import torch

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class CustomDatasetDivByFolder(Dataset):
    def __init__(self, root: str, train=True, subset=None, transform=None, onehot=False):
        self.train = train
        self.transform = transform
        self.targets = []
        self.data = []
        label_type=[0,2,0,1,3]
        if root is not None:
            self.class_data = glob.glob(root + "/*")
            label_index=-1
            for i in range(len(self.class_data)):
                pair_data = glob.glob(self.class_data[i] + "/*")
                if len(pair_data) == 0:
                    continue
                label_index += 1
                if subset is not None and label_index != subset:
                    continue  
                print(self.class_data[i])          
                if train:
                    length = int(0.8 * len(pair_data))
                    offset = 0
                else:
                    length = len(pair_data)
                    offset = int(0.8 * len(pair_data))
                for j in range(offset, length):
                    if onehot:
                        label = torch.zeros(17)
                        label[label_index] = 1
                    else:
                        label = label_type[label_index]
                    self.targets.append(label)
                    # self.targets.append(torch.randint(0,17,[1]))
                    self.data.append(pair_data[j])

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')
        # plt.imshow(img)
        # plt.show()
        target = self.targets[index]
        if self.transform is not None:
            img_1 = self.transform(img)
            # img_2 = self.transform(img)
            # a = img_1.transpose(2, 0)
            # b = a.transpose(0, 1)
            # plt.imshow(b)
            # plt.show()
            if self.train:
                return img_1, target
            else:
                return img_1, target

    def __len__(self):
        return len(self.data)

class CustomDatasetDivByFile(Dataset):
    def __init__(self, root: str, train=True, transform=None, onehot=False, test_class=None, prefix=''):
        self.train = train
        self.transform = transform
        self.prefix = prefix
        self.targets = []
        if root is not None:
            targets = pd.read_csv(root+'/target_40.csv')#target_long_tail
            if train:
                self.targets = targets.loc[targets['2'] == 0]
                self.targets.reset_index(drop=True, inplace=True)
            elif test_class is None:
                self.targets = targets.loc[targets['2'] == 1]
                self.targets.reset_index(drop=True, inplace=True)
            else:
                self.targets = targets.loc[(targets['1'] == test_class)&(targets['2'] == 1)]
                self.targets.reset_index(drop=True, inplace=True)


    def __getitem__(self, index):
        img = Image.open(self.prefix+self.targets['0'][index]).convert('RGB')
        target = self.targets['1'][index]
        
        if self.transform is not None:
            img_1 = self.transform(img)
            if self.train:
                return img_1, target
            else:
                return img_1, target

    def dec2bin(self, decimal):
        decimal = torch.floor(decimal*255)
        size = decimal.size()
        input = decimal.reshape(size[0],size[1]*size[2])
        bin = torch.zeros([8*size[0],size[1]*size[2]])
        for i in range(size[0]):
            for j in range(size[1]*size[2]):
                cur_bin = []
                cur_dec = input[i][j]
                while cur_dec > 1:
                    cur_bin.append(cur_dec % 2)
                    cur_dec = int(cur_dec / 2)
                cur_bin.append(cur_dec % 2)
                for k in range(8):
                    if len(cur_bin)>0:
                        bin[i*8+7-k][j] = cur_bin.pop()
        bin = 2*(bin-0.5)
        return bin.reshape(8*size[0],size[1],size[2])

    def __len__(self):
        return self.targets.shape[0]