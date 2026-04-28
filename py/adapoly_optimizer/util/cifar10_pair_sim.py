import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CIFAR10SimPair(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10SimPair, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # a = transforms.ToTensor()(img).transpose(2, 0)
        # b = a.transpose(0, 1)
        # plt.imshow(b)
        # plt.show()
        if self.train:
            if self.transform is not None:
                img_1 = self.transform(img)
                img_2 = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            # a = img_1.transpose(2, 0)
            # b = a.transpose(0, 1)
            # plt.imshow(b)
            # plt.show()
            # a = img_2.transpose(2, 0)
            # b = a.transpose(0, 1)
            # plt.imshow(b)
            # plt.show()
            return img_1, img_2, target
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.data)

from torch.utils.data import Dataset

class CIFAR100LT(Dataset):
    def __init__(self, root, version='r-10', train=True, transform=None, target_transform=None):
        self.root = root
        self.version = version
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []

        self._load_data()

        # 加载 fine label 名称（100 类）
        meta_path = os.path.join(self.root, f"cifar100-lt-{self.version}", "meta")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as infile:
                self.classes = pickle.load(infile, encoding="bytes")[b"fine_label_names"]
                self.classes = [x.decode("utf-8") for x in self.classes]
        else:
            self.classes = [str(i) for i in range(100)]

    def _load_data(self):
        base_path = os.path.join(self.root)
        file_name = "train" if self.train else "test"
        path = os.path.join(base_path, file_name)
    
        with open(path, "rb") as f:
            entry = pickle.load(f, encoding="bytes")
            self.data = entry[b"data"]
            self.targets = entry[b"fine_labels"]
    
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.train:
            if self.transform is not None:
                img_1 = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img_1, target
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.data)

