import os
import pickle
import lmdb
from PIL import Image
import io
from tqdm import tqdm

def build_lmdb(pkl_path, lmdb_path):
    print(f"⚙️ 构建 LMDB 数据库: {lmdb_path}")
    with open(pkl_path, 'rb') as f:
        entry = pickle.load(f)
        paths = entry['data']
        targets = entry['targets']

    map_size = len(paths) * 3 * 1024 * 1024  # 约 3MB 每张图片估算，可根据数据规模调整

    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, (img_path, label) in enumerate(tqdm(zip(paths, targets), total=len(paths))):
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            item = pickle.dumps({'image': img_bytes, 'label': label})
            txn.put(str(idx).encode(), item)
        txn.put(b'__len__', pickle.dumps(len(paths)))

    print(f"✅ LMDB 构建完成: 共 {len(paths)} 条记录")
account = "/home/wangjzh"
datapath='adam_optimizer/data/imagenet'
# 用法示例
root = account + '/' + datapath
version = 'imagenetlt_lt'

train_pkl = os.path.join(root, version, 'train.pkl')
train_lmdb = os.path.join(root, version, 'train.lmdb')
build_lmdb(train_pkl, train_lmdb)

test_pkl = os.path.join(root, version, 'test.pkl')
test_lmdb = os.path.join(root, version, 'test.lmdb')
build_lmdb(test_pkl, test_lmdb)
