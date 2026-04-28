import os
import tarfile
import pickle
import numpy as np
from tqdm import tqdm


def extract_cifar10_tar(tar_path, extract_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted to {extract_path}")


def load_data_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch[b'data'], batch[b'labels'], batch[b'filenames']


def get_imbalanced_indices(labels, cls_num=10, imb_factor=0.01, imb_type='exp'):
    data_length = len(labels)
    img_max = data_length / cls_num
    img_num_per_cls = []

    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    else:
        raise NotImplementedError(f"imb_type {imb_type} not supported")

    targets_np = np.array(labels)
    new_indices = []

    for cls_idx, img_num in enumerate(img_num_per_cls):
        idx = np.where(targets_np == cls_idx)[0]
        np.random.shuffle(idx)
        selected = idx[:img_num]
        new_indices.extend(selected.tolist())

    return sorted(new_indices)


def save_pickle_batch(save_path, data, labels, filenames, batch_label="training batch"):
    with open(save_path, "wb") as f:
        pickle.dump({
            b'data': data,
            b'labels': labels,
            b'filenames': filenames,
            b'batch_label': batch_label.encode("utf-8")
        }, f)
    print(f"Saved batch to {save_path}")


def generate_imbalanced_dataset(tar_path, output_root, imb_factors):
    extract_path = "./cifar10_extracted"
    extract_cifar10_tar(tar_path, extract_path)

    all_data = []
    all_labels = []
    all_filenames = []

    for i in range(1, 6):
        batch_path = os.path.join(extract_path, "cifar-10-batches-py", f"data_batch_{i}")
        data, labels, filenames = load_data_batch(batch_path)
        all_data.append(data)
        all_labels.extend(labels)
        all_filenames.extend(filenames)

    all_data = np.vstack(all_data)

    for factor in imb_factors:
        print(f"\nProcessing imbalance factor: 1/{int(1/factor)}")
        save_dir = os.path.join(output_root, f"cifar10-lt-r-{int(1/factor)}")
        os.makedirs(save_dir, exist_ok=True)

        indices = get_imbalanced_indices(all_labels, imb_factor=factor)
        data_lt = all_data[indices]
        labels_lt = [all_labels[i] for i in indices]
        filenames_lt = [all_filenames[i] for i in indices]

        # Split into 5 batches like original format
        num_samples = len(data_lt)
        samples_per_batch = num_samples // 5
        for i in range(5):
            start = i * samples_per_batch
            end = (i + 1) * samples_per_batch if i < 4 else num_samples
            batch_data = data_lt[start:end]
            batch_labels = labels_lt[start:end]
            batch_filenames = filenames_lt[start:end]
            save_path = os.path.join(save_dir, f"data_batch_{i + 1}")
            save_pickle_batch(save_path, batch_data, batch_labels, batch_filenames)

        # Copy original test_batch unchanged
        original_test_path = os.path.join(extract_path, "cifar-10-batches-py", "test_batch")
        target_test_path = os.path.join(save_dir, "test_batch")
        with open(original_test_path, "rb") as src, open(target_test_path, "wb") as dst:
            dst.write(src.read())
        print(f"Copied original test_batch to {target_test_path}")


if __name__ == "__main__":
    tar_path = "cifar-100-python.tar.gz"  # 🔁 修改为本地路径
    output_root = "./cifar100_lt_outputs"
    imb_factors = [1/10, 1/20, 1/50, 1/100, 1/200]  # 多个因子保存
    generate_imbalanced_dataset(tar_path, output_root, imb_factors)
