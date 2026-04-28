import glob
import random

import pandas as pd
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class CusImgPair(Dataset):
    def __init__(self, root: str, train=True, transform=None):
        self.train = train
        self.transform = transform
        self.targets = []
        self.data = []
        self.root = root
        if root is not None:
            if train:
                self.class_data = glob.glob(root + "/train/*")
            else:
                self.class_data = glob.glob(root + "/images/*")
            self.data = self.class_data
            self.targets = pd.read_csv(root+'/labels.csv')
            # self.data.append(pair_data[j])

    def __getitem__(self, index):
        img = Image.open(self.root+'/images/'+self.targets['filename'][index]).convert('RGB')
        target = self.targets['label'][index]
        if self.transform is not None:
            img_1 = self.transform(img)
            # img_1 = self.transform(img.rotate(90))
            # img_1 = self.transform(img.transpose(Image.ROTATE_90))
            if self.train:
                img_2 = self.transform(img)
                return img_1, img_2, target
            else:
                return img_1, target, self.targets['filename'][index]

    def __len__(self):
        return len(self.data)
