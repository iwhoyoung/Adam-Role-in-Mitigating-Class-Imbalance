import subprocess
import itertools
import os
from multiprocessing import Pool

# 设置优化器和学习率的搜索空间
optimizers = ["adam", "sgd"]
learning_rates = [1e-4, 5e-5, 1e-5, 1e-2, 5e-3]

# 设置可用 GPU ID 列表（根据你的机器配置修改）
available_gpus = [0, 1, 2, 3]  # 多卡设置

gpu_count = len(available_gpus)

# 可选：结果输出目录
output_dir = "grid_search_results"
os.makedirs(output_dir, exist_ok=True)

# 构建任务列表并分配 GPU ID
param_grid = list(itertools.product(optimizers, learning_rates))
tasks_with_gpu = [(optimizer, lr, available_gpus[i % gpu_count]) for i, (optimizer, lr) in enumerate(param_grid)]

def run_training(params):
    optimizer, lr, gpu_id = params
    print(f"\n[>>] Running optimizer={optimizer}, lr={lr} on GPU {gpu_id}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        subprocess.run([
            "python", "train.py",
            "--optimizer", optimizer,
            "--lr", str(lr)
        ], check=True, env=env)
    except subprocess.CalledProcessError:
        print(f"[!!] Failed for optimizer={optimizer}, lr={lr}")
        return

    # 移动生成的图像文件到输出目录
    plot_filename = f"{optimizer}_lr{lr}_group_loss.png"
    csv_filename = f"group_loss_{optimizer}_lr{lr}.csv"

    for fname in [plot_filename, csv_filename]:
        src = os.path.join("plots" if fname.endswith(".png") else ".", fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src):
            os.rename(src, dst)
            print(f"[✓] Saved {fname} to {output_dir}")
        else:
            print(f"[!] File not found: {src}")

if __name__ == "__main__":
    print(f"[INFO] Running grid search with {gpu_count} GPUs in parallel...")
    with Pool(processes=gpu_count) as pool:
        pool.map(run_training, tasks_with_gpu)