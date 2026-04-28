import tarfile
import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torch
from torchvision import transforms
from multiprocessing import Pool, cpu_count
import io
import random
random.seed(42)

# ---------- 配置 ----------

image_size = 224
max_per_class_base = 1300
num_workers = max(cpu_count() // 2, 2)  # 可调节并行度
image_exts = ('.jpg', '.jpeg', '.png')
import math
# ------------- 处理单个子 tar -------------
def process_single_class(args):
    member, class_idx, outer_tar_path,output_dir = args
    class_name = os.path.splitext(os.path.basename(member.name))[0]
    save_dir = os.path.join(output_dir, class_name)
    os.makedirs(save_dir, exist_ok=True)

    try:
        with tarfile.open(outer_tar_path, 'r') as outer_tar:
            inner_fileobj = outer_tar.extractfile(member)
            if inner_fileobj is None:
                return
            inner_bytes = io.BytesIO(inner_fileobj.read())

        with tarfile.open(fileobj=inner_bytes, mode='r') as inner_tar:
            image_members = [m for m in inner_tar.getmembers()
                             if m.isfile() and m.name.lower().endswith(image_exts)]

            # k = max(int(max_per_class_base / (class_idx + 1)), 1)
            k = max(math.ceil(max_per_class_base / (class_idx + 1)), 2)
            sampled_members = random.sample(image_members, min(k, len(image_members)))

            for m in sampled_members:
                img_f = inner_tar.extractfile(m)
                if img_f is None:
                    continue
                try:
                    img = Image.open(img_f).convert('RGB')
                except UnidentifiedImageError:
                    continue

                # 保存为 JPEG
                img_name = os.path.basename(m.name)
                img_save_path = os.path.join(save_dir, img_name)
                img.save(img_save_path)

    except Exception as e:
        print(f"[ERROR] Class {class_name}: {e}")

# ------------- 主函数 -------------

def main():
    outer_tar_path = '/home/wangjzh/adam_optimizer/data/imagenet2012/train.tar'          # 或 'data/val.tar'
    output_dir = '/home/wangjzh/adam_optimizer/data/imagenet2012_ceil/train_image_round'  # 或 'data/val_preprocessed'
    with tarfile.open(outer_tar_path, 'r') as outer:
        members = [m for m in outer.getmembers() ]

    args_list = [(m, i, outer_tar_path,output_dir) for i, m in enumerate(sorted(members, key=lambda m: m.name))]


    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_single_class, args_list), total=len(args_list)))

    print(f"✅ Done. Saved to {output_dir}")
    outer_tar_path = '/home/wangjzh/adam_optimizer/data/imagenet2012/val.tar'          # 或 'data/val.tar'
    output_dir = '/home/wangjzh/adam_optimizer/data/imagenet2012_ceil/val_image'  # 或 'data/val_preprocessed
    with tarfile.open(outer_tar_path, 'r') as outer:
        members = [m for m in outer.getmembers() ]

    args_list = [(m, i, outer_tar_path,output_dir) for i, m in enumerate(sorted(members, key=lambda m: m.name))]


    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_single_class, args_list), total=len(args_list)))

    print(f"✅ Done. Saved to {output_dir}")
if __name__ == '__main__':
    main()