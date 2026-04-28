import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
from PIL import Image
import os
import pickle
import shutil
import math
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import json

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
                # img_2 = self.transform(img)
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
            return img_1, target
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.data)
        

class CIFAR10LT(Dataset):
    def __init__(self, root, version='r-10', train=True, transform=None, target_transform=None):
        self.root = root
        self.version = version
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []

        self._load_data()

        meta_path = os.path.join(self.root, "batches.meta")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as infile:
                self.classes = pickle.load(infile, encoding="bytes")[b"label_names"]
                self.classes = [x.decode("utf-8") for x in self.classes]
        else:
            self.classes = [str(i) for i in range(10)]

    def _load_data(self):
        base_path = os.path.join(self.root)
        if self.train:
            batch_files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            batch_files = ["test_batch"]

        for batch_name in batch_files:
            path = os.path.join(base_path, batch_name)
            with open(path, "rb") as f:
                entry = pickle.load(f, encoding="bytes")
                self.data.append(entry[b"data"])
                self.targets.extend(entry[b"labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, H, W, C)

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
                # img_2 = self.transform(img)
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
            return img_1, target
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.data)


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


class ImageNetLT(Dataset):
    def __init__(self, root, version='imagenetlt', train=True, transform=None, target_transform=None, max_samples=1300):
        self.root = root
        self.version = version
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.max_samples = max_samples

        self.data = []
        self.targets = []

        # 自动加载或构建数据
        self._prepare()

        # 加载类别名
        meta_path = os.path.join(self.root, self.version, "meta", "fine_label_names.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.classes = pickle.load(f)
        else:
            self.classes = [str(i) for i in range(max(self.targets) + 1)]

    def _prepare(self):
        version_dir = os.path.join(self.root, self.version)
        os.makedirs(version_dir, exist_ok=True)

        file_name = "train_balance.pkl" if self.train else "test.pkl"
        cache_path = os.path.join(version_dir, file_name)
        # print(f"⚙️ 第一次构建数据缓存: {file_name}")

        # if self.train:
        #     self._build_heavy_tail_train(version_dir)
        # else:
        #     self._load_test_as_is(version_dir)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                entry = pickle.load(f)
                self.data = entry["data"]
                self.targets = entry["targets"]
        else:
            print(f"⚙️ 第一次构建数据缓存: {file_name}")

            if self.train:
                self._build_heavy_tail_train(version_dir)
            else:
                self._load_test_as_is(version_dir)

    def _build_heavy_tail_train(self, version_dir):
        src_dir = os.path.join(self.root, "imagenetlt")
        src_map = os.path.join(src_dir, "class_map.txt")
        dst_map = os.path.join(version_dir, "class_map_balance.txt")
    
        # 统计类样本
        class_to_imgs = defaultdict(list)
        with open(src_map, 'r') as f:
            for line in f:
                filename, class_id = line.strip().split()
                class_to_imgs[int(class_id)].append(filename)
    
        print(f"📊 共有 {len(class_to_imgs)} 个类")
    
        # 按类频排序 + ⎡1300 / k⎤ 采样
        sorted_classes = sorted(class_to_imgs.items(), key=lambda x: len(x[1]), reverse=True)
        new_class_map = []
    
        for rank, (class_id, files) in enumerate(sorted_classes, 1):
            #n = math.ceil(self.max_samples / rank)
            n = 10
            selected = files[:n]
            for fname in selected:
                # 不复制，仅记录原路径
                img_path = os.path.join(src_dir, fname)
                new_class_map.append((img_path, class_id))
    
        print(f"✅ 采样完成，共保留 {len(new_class_map)} 张图片")
    
        # 保存新的 class_map.txt（使用完整路径）
        with open(dst_map, 'w') as f:
            for img_path, cid in new_class_map:
                f.write(f"{img_path} {cid}\n")
    
        # 保存 pkl
        paths = [img_path for img_path, _ in new_class_map]
        labels = [int(cid) for _, cid in new_class_map]
        with open(os.path.join(version_dir, "train_balance.pkl"), 'wb') as f:
            pickle.dump({"data": paths, "targets": labels}, f)
    
        self.data, self.targets = paths, labels
    
        # 保存类别名
        meta_dir = os.path.join(version_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        class_names = [str(i) for i in range(max(labels) + 1)]
        with open(os.path.join(meta_dir, "fine_label_names.pkl"), "wb") as f:
            pickle.dump(class_names, f)

    def _load_test_as_is(self, version_dir):
        test_dir = os.path.join(self.root, "imagenet_test")
        test_map = os.path.join(test_dir, "class_map.txt")
        paths, labels = [], []

        with open(test_map, 'r') as f:
            
            for line in f:
                fname, cid = line.strip().split()
                paths.append(os.path.join(test_dir, fname))
                labels.append(int(cid))
        with open(os.path.join(version_dir, "test.pkl"), "wb") as f:
            pickle.dump({"data": paths, "targets": labels}, f)

        self.data, self.targets = paths, labels

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
# import lmdb
# import io

# class ImageNetLT_LMDB(Dataset):
#     def __init__(self, root, version='imagenetlt', train=True, transform=None, target_transform=None):
#         self.root = root
#         self.version = version
#         self.train = train
#         self.transform = transform
#         self.target_transform = target_transform

#         self.data = []
#         self.targets = []

#         version_dir = os.path.join(self.root, self.version)
#         lmdb_name = "train.lmdb" if self.train else "test.lmdb"
#         lmdb_path = os.path.join(version_dir, lmdb_name)

#         if os.path.exists(lmdb_path):
#             print(f"🔧 使用 LMDB 读取: {lmdb_name}")
#             self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
#             with self.env.begin() as txn:
#                 self.length = pickle.loads(txn.get(b'__len__'))
#         else:
#             raise FileNotFoundError(f"LMDB 文件不存在: {lmdb_path}，请先运行 build_lmdb()")

#         # 加载类别名
#         meta_path = os.path.join(self.root, self.version, "meta", "fine_label_names.pkl")
#         if os.path.exists(meta_path):
#             with open(meta_path, "rb") as f:
#                 self.classes = pickle.load(f)
#         else:
#             self.classes = None  # 可选: fallback

#     def __getitem__(self, index):
#         with self.env.begin() as txn:
#             byteflow = txn.get(str(index).encode())
#         item = pickle.loads(byteflow)
#         img = Image.open(io.BytesIO(item['image'])).convert("RGB")
#         target = item['label']

#         if self.transform:
#             img = self.transform(img)
#         if self.target_transform:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return self.length
