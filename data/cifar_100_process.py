import os
import tarfile
import pickle
import numpy as np


def extract_cifar100_tar(tar_path, extract_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted CIFAR-100 to {extract_path}")


def load_cifar100_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch


def get_imbalanced_indices(labels, cls_num=100, imb_factor=0.01, imb_type='exp'):
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


def save_cifar100_format(save_path, data, fine_labels, coarse_labels, filenames, batch_label):
    with open(save_path, "wb") as f:
        pickle.dump({
            b'data': data,
            b'fine_labels': fine_labels,
            b'coarse_labels': coarse_labels,
            b'filenames': filenames,
            b'batch_label': batch_label.encode("utf-8")
        }, f)
    print(f"Saved CIFAR-100 style batch to {save_path}")


def generate_imbalanced_cifar100(tar_path, output_root, imb_factors):
    extract_path = "./cifar100_extracted"
    extract_cifar100_tar(tar_path, extract_path)

    # Load full train and test sets
    train_data = load_cifar100_batch(os.path.join(extract_path, "cifar-100-python", "train"))
    test_data = load_cifar100_batch(os.path.join(extract_path, "cifar-100-python", "test"))

    all_data = train_data[b'data']
    all_fine_labels = train_data[b'fine_labels']
    all_coarse_labels = train_data[b'coarse_labels']
    all_filenames = train_data[b'filenames']

    for factor in imb_factors:
        print(f"\nProcessing imbalance factor: 1/{int(1/factor)}")
        save_dir = os.path.join(output_root, f"cifar100-lt-r-{int(1/factor)}")
        os.makedirs(save_dir, exist_ok=True)

        indices = get_imbalanced_indices(all_fine_labels, cls_num=100, imb_factor=factor)
        data_lt = all_data[indices]
        fine_labels_lt = [all_fine_labels[i] for i in indices]
        coarse_labels_lt = [all_coarse_labels[i] for i in indices]
        filenames_lt = [all_filenames[i] for i in indices]

        # Save imbalanced training set in CIFAR-100 format
        train_save_path = os.path.join(save_dir, "train")
        save_cifar100_format(
            train_save_path,
            data_lt,
            fine_labels_lt,
            coarse_labels_lt,
            filenames_lt,
            batch_label="training batch 1 of 1"
        )

        # Copy original test set unchanged
        test_save_path = os.path.join(save_dir, "test")
        with open(os.path.join(extract_path, "cifar-100-python", "test"), "rb") as src, \
             open(test_save_path, "wb") as dst:
            dst.write(src.read())
        print(f"Copied original test batch to {test_save_path}")


if __name__ == "__main__":
    tar_path = "cifar-100-python.tar.gz"  # 本地路径
    output_root = "./cifar100_lt_outputs"
    imb_factors = [1/10, 1/20, 1/50, 1/100, 1/200]
    generate_imbalanced_cifar100(tar_path, output_root, imb_factors)
