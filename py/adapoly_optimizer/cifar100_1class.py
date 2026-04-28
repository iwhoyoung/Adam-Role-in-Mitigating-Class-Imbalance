import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CIFAR100LT(Dataset):
    def __init__(self, root, version='r-10', train=True, transform=None, target_transform=None, subset_path=None):
        self.root = root
        self.version = version
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []

        if subset_path is not None:
            self._load_subset(subset_path)
        else:
            self._load_data()

        meta_path = os.path.join(self.root, f"cifar100-lt-{self.version}", "meta")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as infile:
                self.classes = pickle.load(infile, encoding="bytes")[b"fine_label_names"]
                self.classes = [x.decode("utf-8") for x in self.classes]
        else:
            self.classes = [str(i) for i in range(100)]

    def _load_data(self):
        file_name = "train" if self.train else "test"
        path = os.path.join(self.root, file_name)
        with open(path, "rb") as f:
            entry = pickle.load(f, encoding="bytes")
            self.data = entry[b"data"]
            self.targets = entry[b"fine_labels"]
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    def _load_subset(self, subset_path):
        with open(subset_path, "rb") as f:
            entry = pickle.load(f, encoding="bytes")
            self.data = entry[b"data"]
            self.targets = entry[b"fine_labels"]
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

from collections import defaultdict
import shutil
def generate_cifar100lt_mini_test(original_test_path, output_path):
    with open(original_test_path, 'rb') as f:
        entry = pickle.load(f, encoding='bytes')

    data = entry[b'data']                      # shape (10000, 3072)
    fine_labels = entry[b'fine_labels']        # list of int
    coarse_labels = entry[b'coarse_labels']    # list of int
    filenames = entry[b'filenames']            # list of bytes

    seen_classes = set()
    indices_to_keep = []

    for idx, label in enumerate(fine_labels):
        if label not in seen_classes:
            indices_to_keep.append(idx)
            seen_classes.add(label)
        if len(seen_classes) == 100:
            break

    # 筛选出对应的数据
    mini_data = data[indices_to_keep]
    mini_fine_labels = [fine_labels[i] for i in indices_to_keep]
    mini_coarse_labels = [coarse_labels[i] for i in indices_to_keep]
    mini_filenames = [filenames[i] for i in indices_to_keep]

    # 构造新的字典
    mini_dict = {
        b'data': mini_data,
        b'fine_labels': mini_fine_labels,
        b'coarse_labels': mini_coarse_labels,
        b'filenames': mini_filenames,
    }

    # 保存为新的 test-like 文件
    with open(output_path, 'wb') as f:
        pickle.dump(mini_dict, f)

    print(f"Saved mini test dataset with 1 image per class to: {output_path}")
# 初始化 test 数据集

# dataset = CIFAR100LT(root="/home/wangjzh/adam_optimizer/data/cifar100_lt_outputs/cifar100-lt-r-100", version='r-100', train=False)

# # 提取每类一张图并保存
# generate_cifar100lt_mini_test(
#     original_test_path="/home/wangjzh/adam_optimizer/data/cifar100_lt_outputs/cifar100-lt-r-100/test",      # 你原始 test 的路径
#     output_path="/home/wangjzh/adam_optimizer/data/cifar100_lt_outputs/cifar100-lt-r-100/test-1perclass"         # 新文件路径
# )
